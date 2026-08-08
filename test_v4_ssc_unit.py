"""Lightweight invariance tests for V4.2-SSC spectral fusion."""

import unittest

import torch

from model.modality_graph_spectral import ModalityConditionedSpectralFusion


class SemanticSpectralCouplingTest(unittest.TestCase):
    def test_zero_initialization_preserves_uniform_margin_baseline(self):
        torch.manual_seed(7)
        batch_size = 3
        num_layers = 4
        embedding_dim = 8
        fusion = ModalityConditionedSpectralFusion(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_orders=3,
            num_aspects=3,
        )
        spectral_bases = torch.randn(batch_size, num_layers, 3, 1, 5, 5)
        modality_embeddings = torch.randn(batch_size, embedding_dim)
        aspect_compatibilities = torch.rand(batch_size, 3)

        output, layer_weights, coefficients, coupling_logits = fusion(
            spectral_bases,
            modality_embeddings,
            aspect_compatibilities,
        )
        expected = spectral_bases[:, :, 0].mean(dim=1)

        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(
            layer_weights,
            torch.full_like(layer_weights, 1.0 / num_layers),
        )
        torch.testing.assert_close(coefficients, torch.zeros_like(coefficients))
        torch.testing.assert_close(coupling_logits, torch.zeros_like(coupling_logits))

    def test_coupling_receives_gradient_at_zero_initialization(self):
        torch.manual_seed(11)
        fusion = ModalityConditionedSpectralFusion(
            embedding_dim=8,
            num_layers=4,
            num_orders=3,
            num_aspects=3,
        )
        spectral_bases = torch.randn(2, 4, 3, 1, 4, 4)
        modality_embeddings = torch.randn(2, 8)
        aspect_compatibilities = torch.tensor(
            [[0.9, 0.2, 0.7], [0.1, 0.8, 0.3]],
            dtype=torch.float32,
        )

        output, _, _, _ = fusion(
            spectral_bases,
            modality_embeddings,
            aspect_compatibilities,
        )
        output.square().mean().backward()

        self.assertIsNotNone(fusion.aspect_coupling.grad)
        self.assertGreater(float(fusion.aspect_coupling.grad.abs().sum()), 0.0)

    def test_centering_and_zero_strength_have_exact_neutral_behavior(self):
        fusion = ModalityConditionedSpectralFusion(
            embedding_dim=8,
            num_layers=4,
            num_orders=3,
            num_aspects=3,
        )
        with torch.no_grad():
            fusion.aspect_coupling.copy_(
                torch.tensor([[1.0, -0.5], [-0.25, 0.75], [0.5, 0.25]])
            )
        modality_embeddings = torch.randn(2, 8)
        _, neutral_coefficients, _ = fusion.conditioning_parameters(
            modality_embeddings,
            batch_size=2,
            aspect_compatibilities=torch.full((2, 3), 0.5),
        )
        torch.testing.assert_close(
            neutral_coefficients,
            torch.zeros_like(neutral_coefficients),
        )

        fusion.aspect_coupling_strength = 0.0
        _, disabled_coefficients, raw_coupling = fusion.conditioning_parameters(
            modality_embeddings,
            batch_size=2,
            aspect_compatibilities=torch.tensor(
                [[0.9, 0.2, 0.7], [0.1, 0.8, 0.3]],
                dtype=torch.float32,
            ),
        )
        self.assertGreater(float(raw_coupling.abs().sum()), 0.0)
        torch.testing.assert_close(
            disabled_coefficients,
            torch.zeros_like(disabled_coefficients),
        )

    def test_default_vit_l_trainable_parameter_budget(self):
        fusion = ModalityConditionedSpectralFusion(
            embedding_dim=768,
            num_layers=4,
            num_orders=3,
            num_aspects=3,
        )
        self.assertEqual(sum(parameter.numel() for parameter in fusion.parameters()), 9234)


if __name__ == "__main__":
    unittest.main()
