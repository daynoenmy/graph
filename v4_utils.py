"""Utilities for V4.2-SSC semantic-spectral coupling."""

import hashlib
import json
import re

import torch
import torch.nn.functional as F

from dataset.constants import CLASS_NAMES, PROMPTS
from model.tokenizer import tokenize
from prompt_utils import get_llm_state_prompts, resolve_prompt_source
from v3_utils import (
    binary_focal_dice_loss,
    deterministic_test_noise,
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
V4_ASPECT_NAMES = ("focal", "diffuse", "structural")
V4_ASPECT_BASE_NAMES = {
    "Brain": "brain MRI scan",
    "Liver": "liver CT scan",
    "Retina": "retinal OCT scan",
    "Chest": "chest X-ray",
    "Retina_OCT2017": "retinal OCT scan",
    "Histopathology": "histopathology microscopy image",
}
V4_ASPECT_TEMPLATES = {
    "focal": "a {base_name} with a focal lesion.",
    "diffuse": "a {base_name} with a diffuse abnormality.",
    "structural": "a {base_name} with structural distortion.",
}


def modality_template_sha256():
    serialized = json.dumps(
        V4_MODALITY_TEMPLATES,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def aspect_template_sha256():
    serialized = json.dumps(
        {
            "names": V4_ASPECT_NAMES,
            "base_names": V4_ASPECT_BASE_NAMES,
            "templates": V4_ASPECT_TEMPLATES,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def anchor_template_sha256():
    serialized = json.dumps(
        {
            "base_names": V4_ASPECT_BASE_NAMES,
            "prompts": PROMPTS,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _v4_state_prompt_sentences(
    dataset_name,
    class_name,
    prompt_source,
    llm_prompt_path,
):
    if class_name not in CLASS_NAMES[dataset_name]:
        raise ValueError(f"unknown class {class_name!r} for dataset {dataset_name!r}")
    base_name = V4_ASPECT_BASE_NAMES[dataset_name]
    resolved_source = resolve_prompt_source(
        prompt_source,
        dataset_name,
        llm_prompt_path,
    )
    if resolved_source == "llm":
        return get_llm_state_prompts(
            dataset_name,
            class_name,
            base_name,
            llm_prompt_path,
        )

    prompted_states = []
    for prompts in (PROMPTS["prompt_normal"], PROMPTS["prompt_abnormal"]):
        sentences = []
        for prompt in prompts:
            state = prompt.format(base_name)
            for template in PROMPTS["prompt_templates"]:
                sentences.append(template.format(state))
        prompted_states.append(sentences)
    return prompted_states


@torch.inference_mode()
def frozen_text_embedding(
    clip_model,
    dataset_name,
    class_name,
    device,
    prompt_source,
    llm_prompt_path,
):
    state_sentences = _v4_state_prompt_sentences(
        dataset_name,
        class_name,
        prompt_source,
        llm_prompt_path,
    )
    anchors = []
    clip_model.eval()
    for sentences in state_sentences:
        tokens = tokenize(sentences).to(device)
        embeddings = clip_model.encode_text(tokens)
        embeddings = F.normalize(embeddings.float(), dim=-1)
        anchors.append(F.normalize(embeddings.mean(dim=0), dim=0))
    return torch.stack(anchors, dim=1)


@torch.inference_mode()
def frozen_text_embedding_dict(
    clip_model,
    dataset_name,
    device,
    prompt_source,
    llm_prompt_path,
):
    return {
        class_name: frozen_text_embedding(
            clip_model,
            dataset_name,
            class_name,
            device,
            prompt_source,
            llm_prompt_path,
        )
        for class_name in CLASS_NAMES[dataset_name]
    }


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


@torch.inference_mode()
def frozen_aspect_embeddings(clip_model, dataset_name, device):
    try:
        base_name = V4_ASPECT_BASE_NAMES[dataset_name]
    except KeyError as error:
        raise KeyError(
            f"no fixed V4.2 aspect base name registered for {dataset_name!r}"
        ) from error
    sentences = [
        V4_ASPECT_TEMPLATES[name].format(base_name=base_name)
        for name in V4_ASPECT_NAMES
    ]
    tokens = tokenize(sentences).to(device)
    embeddings = clip_model.encode_text(tokens).float()
    embeddings = F.normalize(embeddings, dim=-1)
    return embeddings.transpose(0, 1).contiguous()


@torch.inference_mode()
def aspect_geometry_diagnostics(text_embeddings, aspect_embeddings):
    """Measure raw and normal-relative aspect geometry in CLIP text space."""
    if text_embeddings.ndim != 2 or text_embeddings.shape[1] != 2:
        raise ValueError("text_embeddings must have shape [D, 2]")
    if aspect_embeddings.ndim != 2 or aspect_embeddings.shape[1] != len(
        V4_ASPECT_NAMES
    ):
        raise ValueError(
            f"aspect_embeddings must have shape [D, {len(V4_ASPECT_NAMES)}]"
        )
    if text_embeddings.shape[0] != aspect_embeddings.shape[0]:
        raise ValueError("text and aspect embedding dimensions must match")

    normalized_aspects = F.normalize(aspect_embeddings.float(), dim=0)
    normal_anchor = F.normalize(text_embeddings[:, 0].float(), dim=0)
    contrast_directions = normalized_aspects - normal_anchor.unsqueeze(1)
    contrast_norms = torch.linalg.vector_norm(contrast_directions, dim=0)
    normalized_contrasts = F.normalize(contrast_directions, dim=0)
    raw_cosine = normalized_aspects.transpose(0, 1) @ normalized_aspects
    contrast_cosine = normalized_contrasts.transpose(0, 1) @ normalized_contrasts
    return {
        "aspect_names": list(V4_ASPECT_NAMES),
        "raw_cosine": raw_cosine.cpu().tolist(),
        "contrast_cosine": contrast_cosine.cpu().tolist(),
        "contrast_norms": contrast_norms.cpu().tolist(),
    }


def checkpoint_sort_key(path):
    match = re.search(r"v4_2_ssc_head_epoch_(\d+)\.pth$", str(path))
    if match:
        return (0, int(match.group(1)))
    return (1, str(path))


def validate_v4_checkpoint(checkpoint, expected_config):
    if checkpoint.get("method") != "modality_graph_spectral_v4_2_ssc":
        raise ValueError(
            "checkpoint is not a modality_graph_spectral_v4_2_ssc checkpoint"
        )
    if checkpoint.get("encoder_frozen") is not True:
        raise ValueError("V4.2-SSC checkpoint does not declare a frozen encoder")
    if checkpoint.get("version") != 4:
        raise ValueError("checkpoint is not compatible with V4.2-SSC")
    if checkpoint.get("revision") != 2:
        raise ValueError("checkpoint is not a V4.2-SSC revision")
    if checkpoint.get("modality_template_sha256") != modality_template_sha256():
        raise ValueError("V4.2-SSC fixed modality templates differ from training")
    if checkpoint.get("aspect_template_sha256") != aspect_template_sha256():
        raise ValueError("V4.2-SSC aspect templates differ from training")
    if checkpoint.get("anchor_template_sha256") != anchor_template_sha256():
        raise ValueError("V4.2-SSC normal/abnormal templates differ from training")
    actual_config = checkpoint.get("architecture")
    if not isinstance(actual_config, dict):
        raise ValueError("V4.2-SSC checkpoint has no architecture metadata")
    for name, expected in expected_config.items():
        actual = actual_config.get(name)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-8:
                raise ValueError(
                    f"V4.2-SSC architecture mismatch for {name}: "
                    f"checkpoint={actual}, argument={expected}"
                )
        elif actual != expected:
            raise ValueError(
                f"V4.2-SSC architecture mismatch for {name}: "
                f"checkpoint={actual}, argument={expected}"
            )


__all__ = (
    "anchor_template_sha256",
    "aspect_geometry_diagnostics",
    "aspect_template_sha256",
    "binary_focal_dice_loss",
    "checkpoint_sort_key",
    "deterministic_test_noise",
    "frozen_aspect_embeddings",
    "frozen_modality_embedding",
    "frozen_text_embedding_dict",
    "medical_metrics",
    "modality_template_sha256",
    "V4_ASPECT_BASE_NAMES",
    "V4_ASPECT_NAMES",
    "V4_ASPECT_TEMPLATES",
    "V4_MODALITY_TEMPLATES",
    "validate_v4_checkpoint",
)
