"""Frozen multi-layer CLIP with modality-conditioned graph spectral fusion."""

import math

import torch
from torch import nn
import torch.nn.functional as F


_CARDINAL_OFFSETS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _shift_to_neighbor(tensor, row_offset, col_offset):
    shifted = torch.roll(
        tensor,
        shifts=(-row_offset, -col_offset),
        dims=(-2, -1),
    )
    valid = tensor.new_ones((1, 1, tensor.shape[-2], tensor.shape[-1]))
    if row_offset > 0:
        valid[..., -row_offset:, :] = 0
    elif row_offset < 0:
        valid[..., :-row_offset, :] = 0
    if col_offset > 0:
        valid[..., -col_offset:] = 0
    elif col_offset < 0:
        valid[..., :-col_offset] = 0
    return shifted, valid


class FixedGraphLaplacian(nn.Module):
    """Fixed four-neighbor random-walk Laplacian on a CLIP patch grid."""

    def __init__(self, affinity_temperature=0.2, eps=1e-8):
        super().__init__()
        if affinity_temperature <= 0:
            raise ValueError("affinity_temperature must be positive")
        self.affinity_temperature = float(affinity_temperature)
        self.eps = float(eps)

    def _edge_weights(self, feature_grid):
        records = []
        for row_offset, col_offset in _CARDINAL_OFFSETS:
            neighbor, valid = _shift_to_neighbor(
                feature_grid,
                row_offset,
                col_offset,
            )
            cosine = (feature_grid * neighbor).sum(dim=1, keepdim=True)
            weight = torch.exp(
                (cosine.clamp(-1.0, 1.0) - 1.0) / self.affinity_temperature
            )
            records.append((row_offset, col_offset, weight * valid))
        return records

    def _neighbor_average(self, signal, edge_weights):
        weighted_sum = torch.zeros_like(signal)
        weight_sum = torch.zeros_like(signal)
        for row_offset, col_offset, weight in edge_weights:
            neighbor, _ = _shift_to_neighbor(signal, row_offset, col_offset)
            weighted_sum = weighted_sum + weight * neighbor
            weight_sum = weight_sum + weight
        return weighted_sum / weight_sum.clamp_min(self.eps)

    def forward(self, signal, feature_grid):
        if signal.ndim != 4 or signal.shape[1] != 1:
            raise ValueError("signal must have shape [B, 1, H, W]")
        if feature_grid.ndim != 4:
            raise ValueError("feature_grid must have shape [B, D, H, W]")
        if (
            signal.shape[0] != feature_grid.shape[0]
            or signal.shape[-2:] != (feature_grid.shape[-2:])
        ):
            raise ValueError("signal and feature_grid batch/grid shapes must match")
        if min(signal.shape[-2:]) < 2:
            raise ValueError("graph Laplacian requires at least a 2x2 patch grid")

        edge_weights = self._edge_weights(feature_grid)
        first_order = signal - self._neighbor_average(signal, edge_weights)
        second_order = first_order - self._neighbor_average(
            first_order,
            edge_weights,
        )
        return torch.stack((signal, first_order, second_order), dim=1)


class ModalityConditionedSpectralFusion(nn.Module):
    """Preserve semantic margins and add bounded signed spectral residuals."""

    def __init__(
        self,
        embedding_dim,
        num_layers,
        num_orders=3,
        minimum_uniform_mass=0.2,
        max_spectral_coefficient=1.0,
    ):
        super().__init__()
        if embedding_dim < 1 or num_layers < 1 or num_orders < 2:
            raise ValueError(
                "embedding_dim/layer counts must be positive and num_orders "
                "must include at least one residual order"
            )
        if not 0 <= minimum_uniform_mass < 1:
            raise ValueError("minimum_uniform_mass must lie in [0, 1)")
        if max_spectral_coefficient <= 0:
            raise ValueError("max_spectral_coefficient must be positive")
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.num_orders = int(num_orders)
        self.minimum_uniform_mass = float(minimum_uniform_mass)
        self.max_spectral_coefficient = float(max_spectral_coefficient)
        self.layer_conditioner = nn.Linear(
            self.embedding_dim,
            self.num_layers,
        )
        self.residual_conditioner = nn.Linear(
            self.embedding_dim,
            self.num_layers * (self.num_orders - 1),
        )
        # V4.1 starts from the uniform four-layer CLIP semantic baseline.
        # Laplacian corrections are exactly zero before optimization.
        for conditioner in (self.layer_conditioner, self.residual_conditioner):
            nn.init.zeros_(conditioner.weight)
            nn.init.zeros_(conditioner.bias)

    def _validate_modality_embedding(self, modality_embeddings, batch_size):
        if modality_embeddings.ndim == 1:
            expected = (self.embedding_dim,)
            if tuple(modality_embeddings.shape) != expected:
                raise ValueError(
                    f"modality_embeddings must have shape {expected}, "
                    f"got {tuple(modality_embeddings.shape)}"
                )
            modality = modality_embeddings.unsqueeze(0).expand(batch_size, -1)
        elif modality_embeddings.ndim == 2:
            expected = (batch_size, self.embedding_dim)
            if tuple(modality_embeddings.shape) != expected:
                raise ValueError(
                    f"modality_embeddings must have shape {expected}, "
                    f"got {tuple(modality_embeddings.shape)}"
                )
            modality = modality_embeddings
        else:
            raise ValueError("modality_embeddings must have shape [D] or [B, D]")
        return F.normalize(modality, dim=-1)

    def conditioning_parameters(self, modality_embeddings, batch_size):
        modality = self._validate_modality_embedding(
            modality_embeddings,
            batch_size,
        )
        learned_layers = torch.softmax(self.layer_conditioner(modality), dim=-1)
        layer_weights = self.minimum_uniform_mass / self.num_layers
        layer_weights = (
            layer_weights + (1.0 - self.minimum_uniform_mass) * learned_layers
        )
        residual_coefficients = self.max_spectral_coefficient * torch.tanh(
            self.residual_conditioner(modality)
        )
        residual_coefficients = residual_coefficients.view(
            batch_size,
            self.num_layers,
            self.num_orders - 1,
        )
        return layer_weights, residual_coefficients

    def forward(self, spectral_bases, modality_embeddings):
        if spectral_bases.ndim != 6:
            raise ValueError("spectral_bases must have shape [B, L, K, 1, H, W]")
        batch_size, num_layers, num_orders = spectral_bases.shape[:3]
        if (num_layers, num_orders) != (self.num_layers, self.num_orders):
            raise ValueError(
                "spectral_bases layer/order dimensions do not match the fusion"
            )
        layer_weights, residual_coefficients = self.conditioning_parameters(
            modality_embeddings,
            batch_size,
        )
        layer_weights = layer_weights.to(dtype=spectral_bases.dtype)
        residual_coefficients = residual_coefficients.to(dtype=spectral_bases.dtype)
        base_margin = spectral_bases[:, :, 0]
        spectral_residual = (
            spectral_bases[:, :, 1:]
            * residual_coefficients.view(
                batch_size,
                num_layers,
                num_orders - 1,
                1,
                1,
                1,
            )
        ).sum(dim=2)
        corrected_layers = base_margin + spectral_residual
        fused = (
            corrected_layers * layer_weights.view(batch_size, num_layers, 1, 1, 1)
        ).sum(dim=1)
        return fused, layer_weights, residual_coefficients


class FrozenModalityGraphSpectralModel(nn.Module):
    """Four frozen CLIP levels and one modality-conditioned spectral operator."""

    def __init__(
        self,
        clip_model,
        feature_layers=(6, 12, 18, 24),
        text_temperature=10.0,
        laplacian_temperature=0.2,
        spectral_uniform_mass=0.2,
        max_spectral_coefficient=1.0,
        readout_temperature=1.0,
    ):
        super().__init__()
        feature_layers = tuple(int(layer) for layer in feature_layers)
        if not feature_layers or min(feature_layers) < 1:
            raise ValueError("feature_layers must contain positive layer indices")
        if len(set(feature_layers)) != len(feature_layers):
            raise ValueError("feature_layers must not contain duplicates")
        if tuple(sorted(feature_layers)) != feature_layers:
            raise ValueError("feature_layers must be strictly increasing")
        if text_temperature <= 0 or readout_temperature <= 0:
            raise ValueError("text/readout temperatures must be positive")
        available_layers = len(clip_model.visual.transformer.resblocks)
        if max(feature_layers) > available_layers:
            raise ValueError(
                f"feature layer {max(feature_layers)} exceeds the CLIP visual depth "
                f"{available_layers}"
            )

        self.clip_model = clip_model
        self.feature_layers = feature_layers
        self.text_temperature = float(text_temperature)
        self.readout_temperature = float(readout_temperature)
        embedding_dim = int(clip_model.visual.proj.shape[1])
        self.embedding_dim = embedding_dim
        self.laplacian = FixedGraphLaplacian(laplacian_temperature)
        self.spectral_fusion = ModalityConditionedSpectralFusion(
            embedding_dim=embedding_dim,
            num_layers=len(feature_layers),
            num_orders=3,
            minimum_uniform_mass=spectral_uniform_mass,
            max_spectral_coefficient=max_spectral_coefficient,
        )

        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        self.clip_model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        self.spectral_fusion.train(mode)
        return self

    @torch.no_grad()
    def encode_image(self, image):
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

    def _text_margin(self, features, text_embeddings):
        batch_size = features.shape[0]
        if text_embeddings.ndim == 2:
            expected = (self.embedding_dim, 2)
            if tuple(text_embeddings.shape) != expected:
                raise ValueError(
                    f"text_embeddings must have shape {expected}, "
                    f"got {tuple(text_embeddings.shape)}"
                )
            normalized_text = F.normalize(text_embeddings, dim=0)
            logits = features @ normalized_text
        elif text_embeddings.ndim == 3:
            expected = (batch_size, self.embedding_dim, 2)
            if tuple(text_embeddings.shape) != expected:
                raise ValueError(
                    f"text_embeddings must have shape {expected}, "
                    f"got {tuple(text_embeddings.shape)}"
                )
            normalized_text = F.normalize(text_embeddings, dim=1)
            if features.ndim == 2:
                logits = torch.matmul(
                    features.unsqueeze(1),
                    normalized_text,
                ).squeeze(1)
            else:
                logits = torch.matmul(features, normalized_text)
        else:
            raise ValueError("text_embeddings must have shape [D, 2] or [B, D, 2]")
        return self.text_temperature * (logits[..., 1] - logits[..., 0])

    def _spectral_bases(self, patch_levels, text_embeddings):
        bases = []
        grid_shape = None
        for patch_features in patch_levels:
            batch_size, num_patches, _ = patch_features.shape
            grid_size = math.isqrt(num_patches)
            if grid_size * grid_size != num_patches:
                raise ValueError(f"patch count {num_patches} is not a square grid")
            if grid_shape is None:
                grid_shape = (grid_size, grid_size)
            elif grid_shape != (grid_size, grid_size):
                raise ValueError("all feature levels must use the same patch grid")

            margin = self._text_margin(patch_features, text_embeddings)
            margin = margin.view(batch_size, 1, grid_size, grid_size)
            feature_grid = patch_features.transpose(1, 2).reshape(
                batch_size,
                self.embedding_dim,
                grid_size,
                grid_size,
            )
            bases.append(self.laplacian(margin, feature_grid))
        return torch.stack(bases, dim=1)

    def _fixed_image_readout(self, patch_logits, global_features, text_embeddings):
        flat_logits = patch_logits.flatten(start_dim=1)
        attention = torch.softmax(
            flat_logits / self.readout_temperature,
            dim=1,
        )
        local_logits = (attention * flat_logits).sum(dim=1)
        cls_logits = self._text_margin(global_features, text_embeddings)
        return 0.5 * (cls_logits + local_logits), cls_logits, local_logits

    def forward(
        self,
        image,
        text_embeddings,
        modality_embeddings,
        return_image_logits=False,
        return_aux=False,
    ):
        global_features, patch_levels = self.encode_image(image)
        spectral_bases = self._spectral_bases(patch_levels, text_embeddings)
        patch_logits, layer_weights, residual_coefficients = self.spectral_fusion(
            spectral_bases,
            modality_embeddings,
        )
        image_logits = cls_logits = local_logits = None
        if return_image_logits or return_aux:
            image_logits, cls_logits, local_logits = self._fixed_image_readout(
                patch_logits,
                global_features,
                text_embeddings,
            )

        if return_aux:
            auxiliary = {
                "layer_weights": layer_weights,
                "residual_coefficients": residual_coefficients,
                "spectral_bases": spectral_bases,
                "cls_logits": cls_logits,
                "local_logits": local_logits,
                "image_logits": image_logits,
            }
            return patch_logits, auxiliary
        if return_image_logits:
            return patch_logits, image_logits
        return patch_logits

    def trainable_parameters(self):
        yield from self.spectral_fusion.parameters()

    def head_state_dict(self):
        return {"spectral_fusion": self.spectral_fusion.state_dict()}

    def load_head_state_dict(self, state_dict, strict=True):
        if set(state_dict) != {"spectral_fusion"}:
            raise ValueError("trainable state must contain only spectral_fusion")
        return self.spectral_fusion.load_state_dict(
            state_dict["spectral_fusion"],
            strict=strict,
        )

    @torch.no_grad()
    def conditioning_state(self, modality_embeddings):
        layer_weights, residual_coefficients = (
            self.spectral_fusion.conditioning_parameters(
                modality_embeddings,
                batch_size=1,
            )
        )
        return {
            "layer_weights": layer_weights[0].detach().cpu().tolist(),
            "residual_coefficients": (residual_coefficients[0].detach().cpu().tolist()),
        }

    def architecture_config(self):
        return {
            "feature_layers": list(self.feature_layers),
            "spectral_orders": [0, 1, 2],
            "laplacian_graph": "four_neighbor_feature_affinity_random_walk",
            "laplacian_temperature": self.laplacian.affinity_temperature,
            "spectral_fusion": "modality_conditioned_bounded_signed_residual",
            "spectral_uniform_mass": (self.spectral_fusion.minimum_uniform_mass),
            "max_spectral_coefficient": (self.spectral_fusion.max_spectral_coefficient),
            "modality_conditioning": "fixed_template_v1",
            "image_readout": "fixed_attention_cls_mean",
            "readout_temperature": self.readout_temperature,
            "image_cls_weight": 0.5,
            "embedding_dim": self.embedding_dim,
            "text_temperature": self.text_temperature,
            "encoder_frozen": True,
        }
