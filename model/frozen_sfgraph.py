"""Frozen-encoder spatial-frequency coherence graph for medical AD.

The CLIP image and text encoders remain completely frozen. A lightweight head
operates after the image encoder and combines fixed text semantics, stationary
wavelet statistics, and local graph residuals.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


_SPATIAL_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _shift_to_neighbor(tensor, row_offset, col_offset):
    """Return the neighbor at (row + offset, col + offset) and a valid mask."""
    shifted = torch.roll(
        tensor,
        shifts=(-row_offset, -col_offset),
        dims=(-2, -1),
    )
    mask = tensor.new_ones((1, 1, tensor.shape[-2], tensor.shape[-1]))
    if row_offset > 0:
        mask[..., -row_offset:, :] = 0
    elif row_offset < 0:
        mask[..., :-row_offset, :] = 0
    if col_offset > 0:
        mask[..., -col_offset:] = 0
    elif col_offset < 0:
        mask[..., :-col_offset] = 0
    return shifted, mask


def _standardize_spatial(values, eps=1e-6):
    mean = values.mean(dim=(-2, -1), keepdim=True)
    variance = (values - mean).square().mean(dim=(-2, -1), keepdim=True)
    return (values - mean) / variance.add(eps).sqrt()


def _relative_log_energy(values, eps=1e-6):
    reduce_dims = tuple(range(1, values.ndim))
    scale = values.mean(dim=reduce_dims, keepdim=True).clamp_min(eps)
    return torch.log1p(values / scale)


class StationaryHaarWavelet(nn.Module):
    """A fixed, resolution-preserving 2D Haar analysis transform."""

    def __init__(self):
        super().__init__()
        filters = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[-0.5, -0.5], [0.5, 0.5]],
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("filters", filters.unsqueeze(1), persistent=False)

    def forward(self, feature_grid):
        if feature_grid.ndim != 4:
            raise ValueError("feature_grid must have shape [B, C, H, W]")
        channels = feature_grid.shape[1]
        filters = self.filters.to(
            device=feature_grid.device,
            dtype=feature_grid.dtype,
        ).repeat(channels, 1, 1, 1)
        padded = F.pad(feature_grid, (0, 1, 0, 1), mode="reflect")
        bands = F.conv2d(padded, filters, groups=channels)
        bands = bands.view(
            feature_grid.shape[0],
            channels,
            4,
            feature_grid.shape[-2],
            feature_grid.shape[-1],
        )
        return bands.permute(0, 2, 1, 3, 4).contiguous()


class SpatialFrequencyCoherenceHead(nn.Module):
    """A small residual scorer over a deterministic spatial-frequency graph."""

    def __init__(
        self,
        embedding_dim=768,
        hidden_dim=32,
        text_temperature=10.0,
        low_frequency_temperature=0.2,
        high_frequency_temperature=1.0,
        semantic_graph_temperature=0.1,
        max_correction=4.0,
    ):
        super().__init__()
        if embedding_dim < 1 or hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if (
            min(
                text_temperature,
                low_frequency_temperature,
                high_frequency_temperature,
                semantic_graph_temperature,
                max_correction,
            )
            <= 0
        ):
            raise ValueError("temperatures and max_correction must be positive")

        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.text_temperature = float(text_temperature)
        self.low_frequency_temperature = float(low_frequency_temperature)
        self.high_frequency_temperature = float(high_frequency_temperature)
        self.semantic_graph_temperature = float(semantic_graph_temperature)
        self.max_correction = float(max_correction)
        self.wavelet = StationaryHaarWavelet()

        self.residual_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.correction_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 8, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # Start exactly from the frozen CLIP normal/abnormal margin. The graph
        # head then learns only a residual correction.
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    @staticmethod
    def _validate_text_embeddings(text_embeddings, batch_size, embedding_dim):
        if text_embeddings.ndim == 2:
            expected = (embedding_dim, 2)
            if tuple(text_embeddings.shape) != expected:
                raise ValueError(
                    f"text_embeddings must have shape {expected}, "
                    f"got {tuple(text_embeddings.shape)}"
                )
        elif text_embeddings.ndim == 3:
            expected = (batch_size, embedding_dim, 2)
            if tuple(text_embeddings.shape) != expected:
                raise ValueError(
                    f"text_embeddings must have shape {expected}, "
                    f"got {tuple(text_embeddings.shape)}"
                )
        else:
            raise ValueError("text_embeddings must have shape [D, 2] or [B, D, 2]")

    @staticmethod
    def _band_scales(band_scales, patch_features):
        if band_scales is None:
            return patch_features.new_ones((1, 3, 1, 1, 1))
        scales = torch.as_tensor(
            band_scales,
            device=patch_features.device,
            dtype=patch_features.dtype,
        )
        if scales.ndim == 1:
            scales = scales.view(1, 3, 1, 1, 1)
        elif scales.ndim == 2:
            scales = scales.view(scales.shape[0], 3, 1, 1, 1)
        else:
            raise ValueError("band_scales must have shape [3] or [B, 3]")
        if scales.shape[1] != 3 or scales.shape[0] not in (
            1,
            patch_features.shape[0],
        ):
            raise ValueError("band_scales must have shape [3] or [B, 3]")
        return scales.clamp(0.0, 2.0)

    def _semantic_margin(self, patch_features, text_embeddings):
        normalized_features = F.normalize(patch_features, dim=-1)
        if text_embeddings.ndim == 2:
            normalized_text = F.normalize(text_embeddings, dim=0)
            logits = normalized_features @ normalized_text
        else:
            normalized_text = F.normalize(text_embeddings, dim=1)
            logits = torch.matmul(normalized_features, normalized_text)
        logits = logits * self.text_temperature
        margin = logits[..., 1] - logits[..., 0]
        probability = torch.softmax(logits, dim=-1)[..., 1]
        return normalized_features, margin, probability

    def _graph_aggregate(
        self,
        semantic_grid,
        low_grid,
        frequency_descriptor,
        anomaly_probability,
    ):
        neighbor_records = []
        neighbor_anomaly_sum = torch.zeros_like(anomaly_probability)
        neighbor_weight_sum = torch.zeros_like(anomaly_probability)

        for row_offset, col_offset in _SPATIAL_OFFSETS:
            neighbor_semantic, valid = _shift_to_neighbor(
                semantic_grid,
                row_offset,
                col_offset,
            )
            neighbor_low, _ = _shift_to_neighbor(
                low_grid,
                row_offset,
                col_offset,
            )
            neighbor_frequency, _ = _shift_to_neighbor(
                frequency_descriptor,
                row_offset,
                col_offset,
            )
            neighbor_anomaly, _ = _shift_to_neighbor(
                anomaly_probability,
                row_offset,
                col_offset,
            )

            low_distance = (
                1.0 - (low_grid * neighbor_low).sum(dim=1, keepdim=True)
            ).clamp_min(0.0)
            frequency_distance = (
                (frequency_descriptor - neighbor_frequency)
                .square()
                .mean(dim=1, keepdim=True)
            )
            spatial_frequency_weight = (
                torch.exp(
                    -low_distance / self.low_frequency_temperature
                    - frequency_distance / self.high_frequency_temperature
                )
                * valid
            )

            neighbor_anomaly_sum = (
                neighbor_anomaly_sum + spatial_frequency_weight * neighbor_anomaly
            )
            neighbor_weight_sum = neighbor_weight_sum + spatial_frequency_weight
            neighbor_records.append(
                (
                    neighbor_semantic,
                    neighbor_frequency,
                    neighbor_anomaly,
                    spatial_frequency_weight,
                )
            )

        # The consensus excludes the center patch. A false isolated anomaly
        # therefore cannot create its own lesion support through a self-loop.
        neighbor_anomaly_consensus = (
            neighbor_anomaly_sum / neighbor_weight_sum.clamp_min(1e-8)
        )
        lesion_preservation_gate = (
            anomaly_probability * neighbor_anomaly_consensus
        ).clamp(0.0, 1.0)
        isolated_anomaly_evidence = (
            anomaly_probability * (1.0 - neighbor_anomaly_consensus)
        ).clamp(0.0, 1.0)

        semantic_sum = semantic_grid.clone()
        frequency_sum = frequency_descriptor.clone()
        weight_sum = torch.ones_like(anomaly_probability)
        for (
            neighbor_semantic,
            neighbor_frequency,
            neighbor_anomaly,
            spatial_frequency_weight,
        ) in neighbor_records:
            semantic_affinity = torch.exp(
                -(anomaly_probability - neighbor_anomaly).square()
                / self.semantic_graph_temperature
            )
            # This directed gate only suppresses cross-semantic messages when
            # the center patch is supported as a lesion by its neighborhood.
            # Isolated high-frequency responses keep receiving correction from
            # their structurally compatible neighbors.
            preservation_factor = (
                1.0
                - lesion_preservation_gate
                + lesion_preservation_gate * semantic_affinity
            )
            weight = spatial_frequency_weight * preservation_factor
            semantic_sum = semantic_sum + weight * neighbor_semantic
            frequency_sum = frequency_sum + weight * neighbor_frequency
            weight_sum = weight_sum + weight

        return (
            semantic_sum / weight_sum,
            frequency_sum / weight_sum,
            neighbor_anomaly_consensus,
            lesion_preservation_gate,
            isolated_anomaly_evidence,
        )

    def forward(
        self,
        patch_features,
        text_embeddings,
        band_scales=None,
        return_aux=False,
    ):
        if patch_features.ndim != 3:
            raise ValueError("patch_features must have shape [B, N, D]")
        batch_size, num_patches, embedding_dim = patch_features.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"expected embedding dimension {self.embedding_dim}, "
                f"got {embedding_dim}"
            )
        grid_size = math.isqrt(num_patches)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"patch count {num_patches} is not a square grid")
        self._validate_text_embeddings(text_embeddings, batch_size, embedding_dim)

        normalized_features, semantic_margin, anomaly_probability = (
            self._semantic_margin(patch_features, text_embeddings)
        )
        semantic_grid = normalized_features.transpose(1, 2).reshape(
            batch_size,
            embedding_dim,
            grid_size,
            grid_size,
        )
        margin_grid = semantic_margin.view(batch_size, 1, grid_size, grid_size)
        anomaly_grid = anomaly_probability.view(
            batch_size,
            1,
            grid_size,
            grid_size,
        )

        bands = self.wavelet(semantic_grid)
        low_band = bands[:, 0]
        high_bands = bands[:, 1:] * self._band_scales(band_scales, patch_features)
        low_grid = F.normalize(low_band, dim=1)

        low_energy = low_band.square().mean(dim=1, keepdim=True).add(1e-8).sqrt()
        high_energy = high_bands.square().mean(dim=2).add(1e-8).sqrt()
        # A shared scale across all three directional bands preserves their
        # relative energy. Per-band standardization would cancel the training
        # intervention that rescales one selected high-frequency band.
        frequency_descriptor = _relative_log_energy(high_energy)
        standardized_low_energy = _standardize_spatial(torch.log1p(low_energy))

        (
            graph_semantic,
            graph_frequency,
            graph_anomaly,
            lesion_preservation_gate,
            isolated_anomaly_evidence,
        ) = self._graph_aggregate(
            semantic_grid,
            low_grid,
            frequency_descriptor,
            anomaly_grid,
        )
        semantic_residual_vector = semantic_grid - graph_semantic
        semantic_residual = (
            1.0 - F.cosine_similarity(semantic_grid, graph_semantic, dim=1).unsqueeze(1)
        ).clamp_min(0.0)
        frequency_residual = (
            (frequency_descriptor - graph_frequency)
            .square()
            .mean(dim=1, keepdim=True)
            .add(1e-8)
            .sqrt()
        )

        projected_residual = self.residual_projection(
            semantic_residual_vector.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        scalar_features = torch.cat(
            [
                anomaly_grid,
                graph_anomaly,
                lesion_preservation_gate,
                isolated_anomaly_evidence,
                semantic_residual,
                frequency_residual,
                standardized_low_energy,
                frequency_descriptor.mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        correction_input = torch.cat(
            [projected_residual, scalar_features],
            dim=1,
        ).permute(0, 2, 3, 1)
        raw_correction = self.correction_head(correction_input).permute(0, 3, 1, 2)
        correction = self.max_correction * torch.tanh(
            raw_correction / self.max_correction
        )
        logits = margin_grid + correction

        if not return_aux:
            return logits
        return logits, {
            "semantic_margin": margin_grid,
            "semantic_probability": anomaly_grid,
            "neighbor_anomaly": graph_anomaly,
            "lesion_preservation_gate": lesion_preservation_gate,
            "isolated_anomaly_evidence": isolated_anomaly_evidence,
            "semantic_residual": semantic_residual,
            "frequency_residual": frequency_residual,
            "low_frequency_energy": standardized_low_energy,
            "high_frequency_energy": frequency_descriptor,
            "correction": correction,
        }


class LayerProbabilityFusion(nn.Module):
    """Fuse layer evidence in probability space with non-negative weights."""

    def __init__(self, num_layers, minimum_uniform_mass=0.2, eps=1e-6):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0 <= minimum_uniform_mass < 1:
            raise ValueError("minimum_uniform_mass must lie in [0, 1)")
        self.num_layers = int(num_layers)
        self.minimum_uniform_mass = float(minimum_uniform_mass)
        self.eps = float(eps)
        self.weight_logits = nn.Parameter(torch.zeros(self.num_layers))

    def weights(self):
        layer_floor = self.minimum_uniform_mass / self.num_layers
        free_mass = 1.0 - self.minimum_uniform_mass
        return layer_floor + free_mass * torch.softmax(
            self.weight_logits,
            dim=0,
        )

    def forward(self, per_layer_logits):
        if per_layer_logits.ndim != 5 or per_layer_logits.shape[2] != 1:
            raise ValueError(
                "per_layer_logits must have shape [B, L, 1, H, W]"
            )
        if per_layer_logits.shape[1] != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} feature levels, "
                f"got {per_layer_logits.shape[1]}"
            )
        weights = self.weights().to(dtype=per_layer_logits.dtype)
        probability = torch.sigmoid(per_layer_logits)
        fused_probability = (
            probability * weights.view(1, -1, 1, 1, 1)
        ).sum(dim=1)
        return torch.logit(fused_probability.clamp(self.eps, 1.0 - self.eps))


class ExtentAwareImagePool(nn.Module):
    """Combine global semantics with focal and diffuse patch evidence."""

    def __init__(
        self,
        embedding_dim,
        text_temperature=10.0,
        topk_ratio=0.05,
        gem_power=3.0,
        initial_cls_weight=0.5,
        initial_topk_weight=0.5,
        minimum_evidence_weight=0.1,
        eps=1e-6,
    ):
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if text_temperature <= 0 or gem_power <= 0:
            raise ValueError("text_temperature and gem_power must be positive")
        if not 0 < topk_ratio <= 1:
            raise ValueError("topk_ratio must lie in (0, 1]")
        if not 0 <= minimum_evidence_weight < 0.5:
            raise ValueError("minimum_evidence_weight must lie in [0, 0.5)")
        lower = minimum_evidence_weight
        upper = 1.0 - minimum_evidence_weight
        if not lower < initial_cls_weight < upper:
            raise ValueError(
                "initial_cls_weight must lie strictly inside the constrained range"
            )
        if not lower < initial_topk_weight < upper:
            raise ValueError(
                "initial_topk_weight must lie strictly inside the constrained range"
            )

        self.embedding_dim = int(embedding_dim)
        self.text_temperature = float(text_temperature)
        self.topk_ratio = float(topk_ratio)
        self.gem_power = float(gem_power)
        self.minimum_evidence_weight = float(minimum_evidence_weight)
        self.eps = float(eps)
        self.cls_mix_logit = nn.Parameter(
            torch.tensor(self._inverse_bounded_weight(initial_cls_weight))
        )
        self.topk_mix_logit = nn.Parameter(
            torch.tensor(self._inverse_bounded_weight(initial_topk_weight))
        )

    def _inverse_bounded_weight(self, value):
        floor = self.minimum_evidence_weight
        unit_value = (float(value) - floor) / (1.0 - 2.0 * floor)
        return math.log(unit_value / (1.0 - unit_value))

    def _bounded_weight(self, parameter):
        floor = self.minimum_evidence_weight
        return floor + (1.0 - 2.0 * floor) * torch.sigmoid(parameter)

    def _global_probability(self, global_features, text_embeddings):
        if global_features.ndim != 2 or global_features.shape[1] != self.embedding_dim:
            raise ValueError(
                f"global_features must have shape [B, {self.embedding_dim}]"
            )
        SpatialFrequencyCoherenceHead._validate_text_embeddings(
            text_embeddings,
            global_features.shape[0],
            self.embedding_dim,
        )
        normalized_global = F.normalize(global_features, dim=-1)
        if text_embeddings.ndim == 2:
            normalized_text = F.normalize(text_embeddings, dim=0)
            logits = normalized_global @ normalized_text
        else:
            normalized_text = F.normalize(text_embeddings, dim=1)
            logits = torch.matmul(normalized_global.unsqueeze(1), normalized_text)
            logits = logits.squeeze(1)
        return torch.softmax(logits * self.text_temperature, dim=-1)[:, 1]

    def forward(
        self,
        per_layer_logits,
        global_features,
        text_embeddings,
        layer_weights,
        return_aux=False,
    ):
        if per_layer_logits.ndim != 5 or per_layer_logits.shape[2] != 1:
            raise ValueError(
                "per_layer_logits must have shape [B, L, 1, H, W]"
            )
        if layer_weights.ndim != 1 or layer_weights.numel() != per_layer_logits.shape[1]:
            raise ValueError("layer_weights must contain one value per feature level")

        patch_probability = torch.sigmoid(per_layer_logits).flatten(start_dim=3)
        topk_count = max(1, math.ceil(patch_probability.shape[-1] * self.topk_ratio))
        topk_probability = patch_probability.topk(topk_count, dim=-1).values.mean(dim=-1)
        gem_probability = patch_probability.pow(self.gem_power).mean(dim=-1)
        gem_probability = gem_probability.clamp_min(
            torch.finfo(gem_probability.dtype).tiny
        ).pow(1.0 / self.gem_power)

        topk_weight = self._bounded_weight(self.topk_mix_logit)
        per_layer_local = (
            topk_weight * topk_probability
            + (1.0 - topk_weight) * gem_probability
        )
        normalized_layer_weights = layer_weights.to(
            device=per_layer_logits.device,
            dtype=per_layer_logits.dtype,
        )
        normalized_layer_weights = normalized_layer_weights / normalized_layer_weights.sum()
        local_probability = (
            per_layer_local * normalized_layer_weights.view(1, -1, 1)
        ).sum(dim=1).squeeze(1)

        cls_probability = self._global_probability(global_features, text_embeddings)
        cls_weight = self._bounded_weight(self.cls_mix_logit)
        image_probability = (
            cls_weight * cls_probability
            + (1.0 - cls_weight) * local_probability
        )
        image_logits = torch.logit(
            image_probability.clamp(self.eps, 1.0 - self.eps)
        )
        if not return_aux:
            return image_logits
        return image_logits, {
            "image_probability": image_probability,
            "cls_probability": cls_probability,
            "local_probability": local_probability,
            "per_layer_topk_probability": topk_probability.squeeze(-1),
            "per_layer_gem_probability": gem_probability.squeeze(-1),
            "cls_weight": cls_weight,
            "topk_weight": topk_weight,
        }


class FrozenSFGraphModel(nn.Module):
    """Frozen CLIP feature extractor followed by the trainable SF graph head."""

    def __init__(
        self,
        clip_model,
        feature_layers=(6, 12, 18, 24),
        hidden_dim=32,
        text_temperature=10.0,
        low_frequency_temperature=0.2,
        high_frequency_temperature=1.0,
        semantic_graph_temperature=0.1,
        max_correction=4.0,
        topk_ratio=0.05,
        gem_power=3.0,
        initial_cls_pool_weight=0.5,
        initial_topk_pool_weight=0.5,
    ):
        super().__init__()
        feature_layers = tuple(int(layer) for layer in feature_layers)
        if not feature_layers or min(feature_layers) < 1:
            raise ValueError("feature_layers must contain positive layer indices")
        if len(set(feature_layers)) != len(feature_layers):
            raise ValueError("feature_layers must not contain duplicates")
        if tuple(sorted(feature_layers)) != feature_layers:
            raise ValueError("feature_layers must be strictly increasing")
        available_layers = len(clip_model.visual.transformer.resblocks)
        if max(feature_layers) > available_layers:
            raise ValueError(
                f"feature layer {max(feature_layers)} exceeds the CLIP visual depth "
                f"{available_layers}"
            )
        self.clip_model = clip_model
        self.feature_layers = feature_layers
        embedding_dim = int(clip_model.visual.proj.shape[1])
        self.head = SpatialFrequencyCoherenceHead(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            text_temperature=text_temperature,
            low_frequency_temperature=low_frequency_temperature,
            high_frequency_temperature=high_frequency_temperature,
            semantic_graph_temperature=semantic_graph_temperature,
            max_correction=max_correction,
        )
        self.layer_fusion = LayerProbabilityFusion(len(self.feature_layers))
        self.image_pool = ExtentAwareImagePool(
            embedding_dim=embedding_dim,
            text_temperature=text_temperature,
            topk_ratio=topk_ratio,
            gem_power=gem_power,
            initial_cls_weight=initial_cls_pool_weight,
            initial_topk_weight=initial_topk_pool_weight,
        )
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        self.clip_model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        self.head.train(mode)
        self.layer_fusion.train(mode)
        self.image_pool.train(mode)
        return self

    @torch.no_grad()
    def encode_patches(self, image):
        global_features, token_levels = self.clip_model.encode_image(
            image,
            list(self.feature_layers),
        )
        if len(token_levels) != len(self.feature_layers):
            raise RuntimeError(
                "CLIP returned an unexpected number of feature levels: "
                f"expected {len(self.feature_layers)}, got {len(token_levels)}"
            )
        patch_levels = []
        for tokens in token_levels:
            patch_tokens = self.clip_model.visual._global_pool(tokens)[1]
            patch_tokens = self.clip_model.visual.ln_post(patch_tokens)
            if self.clip_model.visual.proj is not None:
                patch_tokens = patch_tokens @ self.clip_model.visual.proj
            patch_levels.append(F.normalize(patch_tokens.float(), dim=-1))
        return F.normalize(global_features.float(), dim=-1), tuple(patch_levels)

    def _forward_feature_levels(
        self,
        patch_levels,
        text_embeddings,
        band_scales=None,
        return_aux=False,
    ):
        level_outputs = [
            self.head(
                patch_features,
                text_embeddings,
                band_scales=band_scales,
                return_aux=return_aux,
            )
            for patch_features in patch_levels
        ]
        level_logits = torch.stack(
            [output[0] if return_aux else output for output in level_outputs],
            dim=1,
        )
        fused_logits = self.layer_fusion(level_logits)
        if not return_aux:
            return fused_logits, level_logits

        layer_weights = self.layer_fusion.weights().to(dtype=level_logits.dtype)
        auxiliary = {
            name: (
                torch.stack(
                    [level_output[1][name] for level_output in level_outputs],
                    dim=1,
                )
                * layer_weights.view(1, -1, 1, 1, 1)
            ).sum(dim=1)
            for name in level_outputs[0][1]
        }
        auxiliary["per_layer_logits"] = level_logits
        auxiliary["layer_weights"] = layer_weights
        return fused_logits, level_logits, auxiliary

    def forward(
        self,
        image,
        text_embeddings,
        band_scales=None,
        return_aux=False,
        return_image_logits=False,
        return_band_perturbation=False,
        band_scale_range=(0.5, 1.5),
    ):
        global_features, patch_levels = self.encode_patches(image)
        feature_output = self._forward_feature_levels(
            patch_levels,
            text_embeddings,
            band_scales=band_scales,
            return_aux=return_aux,
        )
        if return_aux:
            base_logits, per_layer_logits, auxiliary = feature_output
        else:
            base_logits, per_layer_logits = feature_output

        image_logits = None
        if return_image_logits:
            image_logits = self.image_pool(
                per_layer_logits,
                global_features,
                text_embeddings,
                self.layer_fusion.weights(),
            )

        if return_aux:
            if image_logits is not None:
                auxiliary["image_logits"] = image_logits
            output = (base_logits, auxiliary)
        elif image_logits is not None:
            output = (base_logits, image_logits)
        else:
            output = base_logits
        if not return_band_perturbation:
            return output

        minimum, maximum = band_scale_range
        if not 0.0 <= minimum <= maximum <= 2.0:
            raise ValueError("band scale range must lie in [0, 2]")
        band_index = int(torch.randint(0, 3, (), device=image.device))
        scale = float(torch.empty((), device=image.device).uniform_(minimum, maximum))
        scales = patch_levels[0].new_ones(3)
        scales[band_index] = scale
        perturbed_logits, _ = self._forward_feature_levels(
            patch_levels,
            text_embeddings,
            band_scales=scales,
        )
        return (
            output,
            perturbed_logits,
            {
                "band_index": band_index,
                "band_scale": scale,
                "base_logits": base_logits,
            },
        )

    def trainable_parameters(self):
        yield from self.head.parameters()
        yield from self.layer_fusion.parameters()
        yield from self.image_pool.parameters()

    def head_state_dict(self):
        return {
            "graph_head": self.head.state_dict(),
            "layer_fusion": self.layer_fusion.state_dict(),
            "image_pool": self.image_pool.state_dict(),
        }

    def load_head_state_dict(self, state_dict, strict=True):
        expected = {"graph_head", "layer_fusion", "image_pool"}
        if set(state_dict) != expected:
            raise ValueError(
                "V3 trainable state must contain graph_head, layer_fusion, "
                "and image_pool"
            )
        return {
            "graph_head": self.head.load_state_dict(
                state_dict["graph_head"],
                strict=strict,
            ),
            "layer_fusion": self.layer_fusion.load_state_dict(
                state_dict["layer_fusion"],
                strict=strict,
            ),
            "image_pool": self.image_pool.load_state_dict(
                state_dict["image_pool"],
                strict=strict,
            ),
        }

    @torch.no_grad()
    def pooling_state(self):
        return {
            "layer_weights": self.layer_fusion.weights().detach().cpu().tolist(),
            "cls_weight": float(self.image_pool._bounded_weight(
                self.image_pool.cls_mix_logit
            ).detach().cpu()),
            "topk_weight": float(self.image_pool._bounded_weight(
                self.image_pool.topk_mix_logit
            ).detach().cpu()),
        }

    def architecture_config(self):
        return {
            "feature_layers": list(self.feature_layers),
            "layer_fusion": "bounded_softmax_probability",
            "minimum_uniform_layer_mass": self.layer_fusion.minimum_uniform_mass,
            "image_pool": "cls_topk_gem_probability",
            "topk_ratio": self.image_pool.topk_ratio,
            "gem_power": self.image_pool.gem_power,
            "minimum_evidence_weight": self.image_pool.minimum_evidence_weight,
            "embedding_dim": self.head.embedding_dim,
            "hidden_dim": self.head.hidden_dim,
            "text_temperature": self.head.text_temperature,
            "low_frequency_temperature": self.head.low_frequency_temperature,
            "high_frequency_temperature": self.head.high_frequency_temperature,
            "semantic_graph_temperature": self.head.semantic_graph_temperature,
            "max_correction": self.head.max_correction,
            "encoder_frozen": True,
        }
