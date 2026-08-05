"""Utilities for the independent V4 graph-spectral training path."""

import re

from v3_utils import (
    binary_focal_dice_loss,
    deterministic_test_noise,
    frozen_text_embedding_dict,
    v3_metrics,
)


medical_metrics = v3_metrics


def checkpoint_sort_key(path):
    match = re.search(r"v4_head_epoch_(\d+)\.pth$", str(path))
    if match:
        return (0, int(match.group(1)))
    return (1, str(path))


def validate_v4_checkpoint(checkpoint, expected_config):
    if checkpoint.get("method") != "modality_graph_spectral_v4":
        raise ValueError(
            "checkpoint is not a modality_graph_spectral_v4 checkpoint"
        )
    if checkpoint.get("encoder_frozen") is not True:
        raise ValueError("V4 checkpoint does not declare a frozen encoder")
    if checkpoint.get("version") != 4:
        raise ValueError("checkpoint is not compatible with V4")
    actual_config = checkpoint.get("architecture")
    if not isinstance(actual_config, dict):
        raise ValueError("V4 checkpoint has no architecture metadata")
    for name, expected in expected_config.items():
        actual = actual_config.get(name)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-8:
                raise ValueError(
                    f"V4 architecture mismatch for {name}: "
                    f"checkpoint={actual}, argument={expected}"
                )
        elif actual != expected:
            raise ValueError(
                f"V4 architecture mismatch for {name}: "
                f"checkpoint={actual}, argument={expected}"
            )


__all__ = (
    "binary_focal_dice_loss",
    "checkpoint_sort_key",
    "deterministic_test_noise",
    "frozen_text_embedding_dict",
    "medical_metrics",
    "validate_v4_checkpoint",
)
