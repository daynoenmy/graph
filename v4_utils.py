"""Utilities for the independent V4.1 graph-spectral training path."""

import hashlib
import json
import re

import torch
import torch.nn.functional as F

from model.tokenizer import tokenize
from v3_utils import (
    binary_focal_dice_loss,
    deterministic_test_noise,
    frozen_text_embedding_dict,
    v3_metrics,
)


medical_metrics = v3_metrics
V4_MODALITY_TEMPLATES = {
    "Brain": "a brain MRI scan",
    "Liver": "a liver CT scan",
    "Retina": "a retinal OCT scan",
    "Chest": "a chest X-ray",
    "Retina_OCT2017": "a retinal OCT scan",
    "Histopathology": "a histopathology microscopy image",
}


def modality_template_sha256():
    serialized = json.dumps(
        V4_MODALITY_TEMPLATES,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@torch.inference_mode()
def frozen_modality_embedding(clip_model, dataset_name, device):
    try:
        modality_template = V4_MODALITY_TEMPLATES[dataset_name]
    except KeyError as error:
        raise KeyError(
            f"no fixed V4 modality template registered for {dataset_name!r}"
        ) from error
    tokens = tokenize([f"{modality_template}."]).to(device)
    embedding = clip_model.encode_text(tokens).float()[0]
    return F.normalize(embedding, dim=0)


def checkpoint_sort_key(path):
    match = re.search(r"v4_1_head_epoch_(\d+)\.pth$", str(path))
    if match:
        return (0, int(match.group(1)))
    return (1, str(path))


def validate_v4_checkpoint(checkpoint, expected_config):
    if checkpoint.get("method") != "modality_graph_spectral_v4_1":
        raise ValueError("checkpoint is not a modality_graph_spectral_v4_1 checkpoint")
    if checkpoint.get("encoder_frozen") is not True:
        raise ValueError("V4.1 checkpoint does not declare a frozen encoder")
    if checkpoint.get("version") != 4:
        raise ValueError("checkpoint is not compatible with V4.1")
    if checkpoint.get("revision") != 1:
        raise ValueError("checkpoint is not a V4.1 revision")
    if checkpoint.get("modality_template_sha256") != modality_template_sha256():
        raise ValueError("V4.1 fixed modality templates differ from training")
    actual_config = checkpoint.get("architecture")
    if not isinstance(actual_config, dict):
        raise ValueError("V4.1 checkpoint has no architecture metadata")
    for name, expected in expected_config.items():
        actual = actual_config.get(name)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-8:
                raise ValueError(
                    f"V4.1 architecture mismatch for {name}: "
                    f"checkpoint={actual}, argument={expected}"
                )
        elif actual != expected:
            raise ValueError(
                f"V4.1 architecture mismatch for {name}: "
                f"checkpoint={actual}, argument={expected}"
            )


__all__ = (
    "binary_focal_dice_loss",
    "checkpoint_sort_key",
    "deterministic_test_noise",
    "frozen_modality_embedding",
    "frozen_text_embedding_dict",
    "medical_metrics",
    "modality_template_sha256",
    "V4_MODALITY_TEMPLATES",
    "validate_v4_checkpoint",
)
