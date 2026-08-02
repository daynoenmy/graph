"""Utilities shared by the frozen-encoder V3 training and evaluation paths."""

import hashlib
import math
import re

import numpy as np
import torch
import torch.nn.functional as F

from dataset.constants import CLASS_NAMES, PROMPTS, REAL_NAMES
from model.tokenizer import tokenize
from prompt_utils import get_llm_state_prompts, resolve_prompt_source
from utils import make_medical_noise_view


def state_prompt_sentences(
    dataset_name,
    class_name,
    prompt_source,
    llm_prompt_path,
):
    if class_name not in CLASS_NAMES[dataset_name]:
        raise ValueError(f"unknown class {class_name!r} for dataset {dataset_name!r}")
    real_name = REAL_NAMES[dataset_name][class_name]
    resolved_source = resolve_prompt_source(
        prompt_source,
        dataset_name,
        llm_prompt_path,
    )
    if resolved_source == "llm":
        return get_llm_state_prompts(
            dataset_name,
            class_name,
            real_name,
            llm_prompt_path,
        )

    prompted_states = []
    for prompts in (PROMPTS["prompt_normal"], PROMPTS["prompt_abnormal"]):
        sentences = []
        for prompt in prompts:
            state = prompt.format(real_name)
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
    state_sentences = state_prompt_sentences(
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


def binary_focal_dice_loss(logits, target, gamma=2.0, eps=1e-6):
    if logits.shape != target.shape:
        raise ValueError(
            f"logits and target must have equal shape, got {logits.shape} and "
            f"{target.shape}"
        )
    target = target.to(dtype=logits.dtype)
    probability = torch.sigmoid(logits)
    binary_ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    target_probability = probability * target + (1.0 - probability) * (1.0 - target)
    focal = ((1.0 - target_probability).pow(gamma) * binary_ce).mean()

    reduce_dims = tuple(range(1, logits.ndim))
    lesion_intersection = (probability * target).sum(dim=reduce_dims)
    lesion_dice = (2.0 * lesion_intersection + eps) / (
        probability.sum(dim=reduce_dims) + target.sum(dim=reduce_dims) + eps
    )
    normal_probability = 1.0 - probability
    normal_target = 1.0 - target
    normal_intersection = (normal_probability * normal_target).sum(dim=reduce_dims)
    normal_dice = (2.0 * normal_intersection + eps) / (
        normal_probability.sum(dim=reduce_dims)
        + normal_target.sum(dim=reduce_dims)
        + eps
    )
    dice = 1.0 - 0.5 * (lesion_dice + normal_dice).mean()
    return focal + dice


def smooth_max_pool_logits(patch_logits, temperature=10.0):
    if temperature <= 0:
        raise ValueError("pooling temperature must be positive")
    flat = patch_logits.flatten(start_dim=1)
    return (
        torch.logsumexp(flat * temperature, dim=1) - math.log(flat.shape[1])
    ) / temperature


def normal_band_consistency_loss(base_logits, perturbed_logits, mask, eps=1e-6):
    if base_logits.shape != perturbed_logits.shape:
        raise ValueError("base and perturbed logits must have equal shape")
    patch_mask = F.adaptive_max_pool2d(mask.float(), base_logits.shape[-2:])
    normal_weight = 1.0 - patch_mask
    difference = (torch.sigmoid(base_logits) - torch.sigmoid(perturbed_logits)).square()
    return (difference * normal_weight).sum() / normal_weight.sum().clamp_min(eps)


def safe_binary_metric(metric, labels, scores):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(metric(labels, scores))


def v3_metrics(masks, labels, score_maps, image_scores, class_name):
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "test_v3.py requires scikit-learn for AUC/AP metrics"
        ) from error
    return {
        "class name": class_name,
        "pixel AUC": 100.0
        * safe_binary_metric(roc_auc_score, np.asarray(masks), np.asarray(score_maps)),
        "pixel AP": 100.0
        * safe_binary_metric(
            average_precision_score,
            np.asarray(masks),
            np.asarray(score_maps),
        ),
        "image AUC": 100.0
        * safe_binary_metric(
            roc_auc_score, np.asarray(labels), np.asarray(image_scores)
        ),
        "image AP": 100.0
        * safe_binary_metric(
            average_precision_score,
            np.asarray(labels),
            np.asarray(image_scores),
        ),
    }


def checkpoint_sort_key(path):
    match = re.search(r"v3_head_epoch_(\d+)\.pth$", str(path))
    if match:
        return (0, int(match.group(1)))
    return (1, str(path))


def stable_noise_seed(seed, file_name):
    value = f"{seed}:v3-test-noise:{file_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**31)


def deterministic_test_noise(image, file_names, dataset, severity, seed):
    if severity == 0:
        return image
    views = []
    cuda_devices = []
    if image.is_cuda:
        cuda_devices = [
            image.device.index
            if image.device.index is not None
            else torch.cuda.current_device()
        ]
    for sample, file_name in zip(image.split(1, dim=0), file_names):
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(stable_noise_seed(seed, file_name))
            views.append(make_medical_noise_view(sample, dataset, severity=severity))
    return torch.cat(views, dim=0)


def validate_v3_checkpoint(checkpoint, expected_config):
    if checkpoint.get("method") != "frozen_sfgraph_v3":
        raise ValueError("checkpoint is not a frozen_sfgraph_v3 checkpoint")
    if checkpoint.get("encoder_frozen") is not True:
        raise ValueError("V3 checkpoint does not declare a frozen encoder")
    actual_config = checkpoint.get("architecture")
    if not isinstance(actual_config, dict):
        raise ValueError("V3 checkpoint has no architecture metadata")
    for name, expected in expected_config.items():
        actual = actual_config.get(name)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-8:
                raise ValueError(
                    f"V3 architecture mismatch for {name}: "
                    f"checkpoint={actual}, argument={expected}"
                )
        elif actual != expected:
            raise ValueError(
                f"V3 architecture mismatch for {name}: "
                f"checkpoint={actual}, argument={expected}"
            )
