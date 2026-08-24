import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import (
    PatchGraphBlock,
    ResidualBottleneckHead,
    SimpleAdapter,
    SimpleProj,
)


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
        patch_graph_residual_weight: float = 0.7,
        patch_graph_use_spatial: bool = True,
        det_hidden_dim: int = 128,
        det_dropout: float = 0.1,
        det_residual_scale: float = 1.0,
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
        # Keep CLIP's frozen visual projection as the image-level baseline and
        # learn only a compact residual in its 768-D joint embedding space.
        det_head = ResidualBottleneckHead(
            dim=768,
            hidden_dim=det_hidden_dim,
            dropout=det_dropout,
            residual_scale=det_residual_scale,
        )
        patch_graph = (
            PatchGraphBlock(
                dim=768,
                k=patch_graph_k,
                alpha=patch_graph_alpha,
                residual_weight=patch_graph_residual_weight,
                use_spatial=patch_graph_use_spatial,
                # all seg levels join the same big graph
                num_levels=len(levels),
            )
            if enable_patch_graph
            else nn.Identity()
        )
        self.image_adapter = nn.ModuleDict(
            {
                "layer_adapters": layer_adapters,
                "seg_proj": seg_proj,
                "det_head": det_head,
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
        # The generic Xavier pass above would overwrite the zero-initialized
        # residual output projection, so restore the identity initialization.
        self.image_adapter["det_head"].reset_parameters()

    def image_trainable_parameters(self):
        yield from self.image_adapter.parameters()

    def localization_trainable_parameters(self):
        yield from self.image_adapter["layer_adapters"].parameters()
        yield from self.image_adapter["seg_proj"].parameters()
        yield from self.image_adapter["patch_graph"].parameters()

    def classification_trainable_parameters(self):
        yield from self.image_adapter["det_head"].parameters()

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

    def forward(self, x):
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
                    / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.i_w * adapt_out + (1 - self.i_w) * x
            if i + 1 in self.levels:
                tokens.append(x[1:, :, :])

        x = x.permute(1, 0, 2)
        # Stop the image-level objective before the shared visual backbone.
        # Classification starts from CLIP's frozen visual projection and
        # updates only the compact residual head.
        det_source = self.image_encoder.ln_post(x[:, 0, :]).detach()
        visual_proj = getattr(self.image_encoder, "proj", None)
        if visual_proj is None:
            raise RuntimeError("the CLIP visual encoder has no projection matrix")
        det_base = det_source @ visual_proj.detach()
        det_token = self.image_adapter["det_head"](det_base)
        det_token = F.normalize(det_token, dim=-1)

        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]
        seg_tokens = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
        ]
        # cross-level fusion: the 4 levels enter one big graph, propagate once,
        # then are averaged back into a single fused patch feature map.
        fused_tokens = torch.cat(seg_tokens, dim=1)  # (B, L * num_levels, 768)
        fused_tokens = self.image_adapter["patch_graph"](fused_tokens)
        num_levels = len(seg_tokens)
        seg_tokens = fused_tokens.view(
            fused_tokens.shape[0], num_levels, fused_tokens.shape[1] // num_levels, -1
        ).mean(dim=1)  # (B, L, 768)
        seg_tokens = F.normalize(seg_tokens, dim=-1)

        return seg_tokens, det_token

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
                    / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        return x
