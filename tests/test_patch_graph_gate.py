import unittest

import torch
import torch.nn.functional as F
from torch import nn

from model.adapter import AdaptedCLIP
from model.adapter_modules import (
    PatchGraphBlock,
    _build_knn_patch_graph,
    _normalize_adj,
)


def legacy_graph_v2_forward(block, patch_features):
    semantic_adj, _ = _build_knn_patch_graph(
        patch_features,
        k=block.k,
        temperature=block.temperature,
    )
    adj = _normalize_adj(semantic_adj)
    patch_features = F.normalize(patch_features, dim=-1)
    graph_features = adj @ patch_features
    graph_features = F.normalize(
        block.norm(block.proj(graph_features)), dim=-1
    )
    disagreement = torch.abs(patch_features - graph_features)
    gate = torch.sigmoid(
        block.gate_up(
            block.gate_activation(block.gate_down(disagreement))
        )
    )
    out = patch_features + gate * (graph_features - patch_features)
    return F.normalize(out, dim=-1)


def graph_config(gate_source):
    model = AdaptedCLIP.__new__(AdaptedCLIP)
    nn.Module.__init__(model)
    model.enable_patch_graph = True
    model.levels = [6, 12, 18, 24]
    model.image_adapter = nn.ModuleDict(
        {
            "patch_graph": PatchGraphBlock(
                dim=8,
                k=4,
                alpha=0.5,
                residual_weight=0.2,
                use_spatial=False,
                num_levels=4,
                temperature=0.1,
                gate_hidden_dim=4,
                gate_source=gate_source,
            )
        }
    )
    return model.patch_graph_config()


class PatchGraphGateTest(unittest.TestCase):
    def test_post_projection_is_exact_graph_v2_forward(self):
        torch.manual_seed(7)
        block = PatchGraphBlock(
            dim=8,
            k=3,
            use_spatial=False,
            gate_hidden_dim=4,
            gate_source="post_projection",
        )
        with torch.no_grad():
            block.proj.weight.normal_()
            block.gate_up.weight.normal_()
        features = torch.randn(2, 9, 8)

        expected = legacy_graph_v2_forward(block, features)
        actual = block(features)

        self.assertTrue(torch.equal(actual, expected))

    def test_gate_source_changes_only_forward_policy(self):
        torch.manual_seed(11)
        pre = PatchGraphBlock(
            dim=8,
            k=3,
            use_spatial=False,
            gate_hidden_dim=4,
            gate_source="pre_projection",
        )
        post = PatchGraphBlock(
            dim=8,
            k=3,
            use_spatial=False,
            gate_hidden_dim=4,
            gate_source="post_projection",
        )
        with torch.no_grad():
            pre.proj.weight.normal_()
            pre.gate_up.weight.normal_()
        post.load_state_dict(pre.state_dict(), strict=True)
        features = torch.randn(2, 9, 8)

        self.assertEqual(set(pre.state_dict()), set(post.state_dict()))
        self.assertFalse(torch.equal(pre(features), post(features)))

    def test_pre_projection_backward_is_finite(self):
        torch.manual_seed(13)
        block = PatchGraphBlock(
            dim=8,
            k=3,
            use_spatial=False,
            gate_hidden_dim=4,
            gate_source="pre_projection",
        )
        features = torch.randn(2, 9, 8, requires_grad=True)
        output_weights = torch.randn_like(features)

        (block(features) * output_weights).sum().backward()

        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0.0)
        for parameter in block.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_checkpoint_config_keeps_v2_compatibility(self):
        post_config = graph_config("post_projection")
        pre_config = graph_config("pre_projection")

        self.assertEqual(post_config["version"], "weighted_adaptive_v2")
        self.assertNotIn("gate_source", post_config)
        self.assertEqual(pre_config["version"], "reliability_gated_v3")
        self.assertEqual(pre_config["gate_source"], "pre_projection")

    def test_invalid_gate_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "gate_source"):
            PatchGraphBlock(dim=8, gate_source="unknown")


if __name__ == "__main__":
    unittest.main()
