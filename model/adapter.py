import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import PatchGraphBlock, SimpleAdapter, SimpleProj


class AdaptedCLIP(nn.Module):
    def __init__(
        self,
        clip_model,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        levels: list = [6, 12, 18, 24],
        relu: bool = True,
        enable_patch_graph: bool = True,
        patch_graph_k: int = 8,
        patch_graph_alpha: float = 0.7,
        patch_graph_residual_weight: float = 0.2,
        patch_graph_use_spatial: bool = True,
        patch_graph_feature_temperature: float = 0.2,
        patch_graph_anomaly_temperature: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.text_adapt_until = text_adapt_until
        self.image_adapt_until = image_adapt_until
        self.t_w = text_adapt_weight
        self.i_w = image_adapt_weight
        self.levels = levels
        self.enable_patch_graph = enable_patch_graph

        layer_adapters = nn.ModuleList(
            [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
        )
        seg_proj = nn.ModuleList(
            [SimpleProj(1024, 768, relu) for _ in range(len(levels))]
        )
        det_proj = SimpleProj(1024, 768, relu)
        patch_graph = (
            PatchGraphBlock(
                dim=768,
                k=patch_graph_k,
                alpha=patch_graph_alpha,
                residual_weight=patch_graph_residual_weight,
                use_spatial=patch_graph_use_spatial,
                feature_temperature=patch_graph_feature_temperature,
                anomaly_temperature=patch_graph_anomaly_temperature,
            )
            if enable_patch_graph
            else nn.Identity()
        )
        self.image_adapter = nn.ModuleDict(
            {
                "layer_adapters": layer_adapters,
                "seg_proj": seg_proj,
                "det_proj": det_proj,
                "patch_graph": patch_graph,
            }
        )
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
            + [SimpleProj(768, 768, relu=True)]
        )
        self._init_weights_()

    def _init_weights_(self):
        for p in self.image_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def image_trainable_parameters(self):
        yield from self.image_adapter.parameters()

    def image_adapter_parameters_without_input_denoiser(self):
        # Compatibility alias for checkpoints/scripts created before the
        # diffusion-free noise-aware graph pipeline.
        yield from self.image_adapter.parameters()

    def text_trainable_parameters(self):
        yield from self.text_adapter.parameters()

    def forward_original(self, x, modality="visual"):
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")

    def _encode_pre_graph(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        tokens = []
        for i in range(24):
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if i < self.image_adapt_until:
                adapt_out = self.image_adapter["layer_adapters"][i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                )
                x = self.i_w * adapt_out + (1 - self.i_w) * x
            if i + 1 in self.levels:
                tokens.append(x[1:, :, :])

        x = x.permute(1, 0, 2)
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]
        seg_tokens = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
        ]
        det_token = self.image_adapter["det_proj"](tokens[-1])
        return seg_tokens, det_token

    @staticmethod
    def _patch_uncertainty(primary_features, reference_features):
        primary = F.normalize(primary_features, dim=-1)
        reference = F.normalize(reference_features, dim=-1)
        # Cosine distance is in [0, 2]; scale it to a bounded uncertainty.
        return ((1.0 - (primary * reference).sum(dim=-1)) * 0.5).clamp(0.0, 1.0)

    @staticmethod
    def _anomaly_probability(patch_features, text_embeddings):
        if text_embeddings is None:
            return None
        features = F.normalize(patch_features, dim=-1)
        if text_embeddings.ndim == 2:
            text = F.normalize(text_embeddings, dim=0)
            logits = features @ text
        elif text_embeddings.ndim == 3:
            if text_embeddings.shape[0] != patch_features.shape[0]:
                raise ValueError("batched text embeddings must match image batch size")
            text = F.normalize(text_embeddings, dim=1)
            logits = torch.matmul(features, text)
        else:
            raise ValueError("text embeddings must have shape [D, 2] or [B, D, 2]")
        if logits.shape[-1] != 2:
            raise ValueError("normal/abnormal text embeddings must contain two anchors")
        return torch.softmax(logits * 10.0, dim=-1)[..., 1]

    def _refine_patch_features(self, features, uncertainty, anomaly_prob):
        if not self.enable_patch_graph:
            return F.normalize(features, dim=-1)
        return self.image_adapter["patch_graph"](
            features,
            uncertainty=uncertainty,
            anomaly_prob=anomaly_prob,
        )

    def forward(
        self,
        x,
        reference_image=None,
        text_embeddings=None,
        return_aux=False,
    ):
        primary_seg, primary_det = self._encode_pre_graph(x)

        if reference_image is None:
            reference_seg = None
            graph_seg = primary_seg
            graph_det = primary_det
            seg_uncertainty = [None for _ in primary_seg]
            det_uncertainty = None
        else:
            # The perturbed branch is a stable teacher view. Gradients flow
            # through the primary branch and graph parameters only, reducing
            # memory while still enforcing noise consistency.
            with torch.no_grad():
                reference_seg, reference_det = self._encode_pre_graph(reference_image)
            graph_seg = [
                (primary + reference) * 0.5
                for primary, reference in zip(primary_seg, reference_seg)
            ]
            graph_det = (primary_det + reference_det) * 0.5
            seg_uncertainty = [
                self._patch_uncertainty(primary, reference)
                for primary, reference in zip(primary_seg, reference_seg)
            ]
            det_uncertainty = self._patch_uncertainty(primary_det, reference_det)

        seg_anomaly_prob = [
            self._anomaly_probability(features, text_embeddings)
            for features in graph_seg
        ]
        det_anomaly_prob = self._anomaly_probability(graph_det, text_embeddings)
        refined_seg = [
            self._refine_patch_features(features, uncertainty, anomaly_prob)
            for features, uncertainty, anomaly_prob in zip(
                graph_seg, seg_uncertainty, seg_anomaly_prob
            )
        ]
        refined_det = self._refine_patch_features(
            graph_det, det_uncertainty, det_anomaly_prob
        )
        det_token = F.normalize(refined_det, dim=-1).mean(1)

        if not return_aux:
            return refined_seg, det_token
        auxiliary = {
            "primary_features": primary_seg,
            "reference_features": reference_seg,
            "graph_input_features": graph_seg,
            "uncertainty": seg_uncertainty,
            "anomaly_probability": seg_anomaly_prob,
        }
        return refined_seg, det_token, auxiliary

    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(
            cast_dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(12):
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        return x
