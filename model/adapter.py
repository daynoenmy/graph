import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import CPTextAdapter, PatchGraphBlock, SimpleAdapter, SimpleProj


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
        text_adapter_type: str = "simple",
        cp_rank: int = 32,
        cp_generator_layers: int = 3,
        cp_beta_std: float = 0.01,
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
        self.text_adapter_type = text_adapter_type

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
        if self.text_adapter_type == "cp_clip":
            self.text_adapter = CPTextAdapter(
                dim=768,
                rank=cp_rank,
                text_layers=text_adapt_until,
                generator_layers=cp_generator_layers,
                beta_std=cp_beta_std,
            )
        else:
            self.text_adapter = nn.ModuleList(
                [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
                + [SimpleProj(768, 768, relu=True)]
            )
        self._init_weights_()

    def _init_weights_(self):
        for p in self.image_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.text_adapter_type != "cp_clip":
            for p in self.text_adapter.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def image_trainable_parameters(self):
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
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]
        seg_tokens = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
        ]
        seg_tokens = [self.image_adapter["patch_graph"](t) for t in seg_tokens]
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]

        det_token = self.image_adapter["det_proj"](tokens[-1])
        det_token = self.image_adapter["patch_graph"](det_token)
        det_token = F.normalize(det_token, dim=-1).mean(1)
        return seg_tokens, det_token

    def _cp_attention(self, block, x, delta_q, delta_v):
        q_x = block.ln_1(x)
        embed_dim = q_x.shape[-1]
        attn_mask = self.clipmodel.attn_mask
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        in_proj_weight = block.attn.in_proj_weight
        adapted_weight = torch.cat(
            [
                in_proj_weight[:embed_dim] + delta_q.to(in_proj_weight.dtype),
                in_proj_weight[embed_dim : 2 * embed_dim],
                in_proj_weight[2 * embed_dim :] + delta_v.to(in_proj_weight.dtype),
            ],
            dim=0,
        )
        attn_out, _ = F.multi_head_attention_forward(
            query=q_x,
            key=q_x,
            value=q_x,
            embed_dim_to_check=embed_dim,
            num_heads=block.attn.num_heads,
            in_proj_weight=adapted_weight,
            in_proj_bias=block.attn.in_proj_bias,
            bias_k=block.attn.bias_k,
            bias_v=block.attn.bias_v,
            add_zero_attn=block.attn.add_zero_attn,
            dropout_p=block.attn.dropout,
            out_proj_weight=block.attn.out_proj.weight,
            out_proj_bias=block.attn.out_proj.bias,
            training=block.attn.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = x + block.ls_1(attn_out)
        x = x + block.ls_2(block.mlp(block.ln_2(x)))
        return x

    def encode_text(self, text, adapt_text=True, visual_context=None):
        if not adapt_text:
            return self.clipmodel.encode_text(text)
        if self.text_adapter_type == "cp_clip":
            return self.encode_text_cp(text, visual_context)
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

    def encode_text_cp(self, text, visual_context=None):
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        if visual_context is None:
            visual_context = torch.zeros(
                1, 1, 768, device=text.device, dtype=cast_dtype
            )
        visual_context = visual_context.to(device=text.device, dtype=cast_dtype)
        deltas = self.text_adapter(visual_context)[0]
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        for i, block in enumerate(self.clipmodel.transformer.resblocks):
            if i < self.text_adapt_until:
                x = self._cp_attention(block, x, deltas[i, 0], deltas[i, 1])
            else:
                x, _ = block(x, attn_mask=self.clipmodel.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clipmodel.text_projection
        return x
