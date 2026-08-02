"""Visualize the four post-hoc patch states used to explain V1.

This script runs one real medical case through the current V1 checkpoint and
visualizes the continuous quantities that actually control graph propagation:

* feature uncertainty U from the clean/noisy feature cosine distance;
* anomaly probability A from the fused feature and text anchors;
* prediction disagreement D between the clean and noisy branches;
* source reliability max(1 - U, 0.05);
* the learned graph update gate g;
* the exact graph aggregation source weights for representative patches.

The four colored states are an explicitly post-hoc explanation of these
continuous quantities. They are not an additional classifier and do not alter
V1 inference.

Windows example::

    python visualize_patch_states.py ^
      --dataset Liver ^
      --save_path ./ckpt/noise_graph_cls_llm ^
      --image_checkpoint image_adapter_1.pth ^
      --label anomaly ^
      --case_index 0 ^
      --noise_severity 0.06 ^
      --output_dir ./patch_state_examples/Liver
"""

import argparse
import csv
import hashlib
import math
import re
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from dataset import DOMAINS, get_dataset
from dataset.constants import CLASS_NAMES, PROMPTS, REAL_NAMES
from model.adapter import AdaptedCLIP
from model.adapter_modules import _build_knn_patch_graph, _build_spatial_patch_graph
from model.tokenizer import tokenize
from prompt_utils import (
    DEFAULT_LLM_PROMPT_PATH,
    PROMPT_SOURCES,
    get_llm_state_prompts,
    prompt_checkpoint_metadata,
    resolve_prompt_source,
    validate_checkpoint_prompt_metadata,
)
from utils import (
    CLIP_MEAN,
    CLIP_STD,
    make_medical_noise_view,
    noise_model_for_dataset,
    setup_seed,
)


STATE_NAMES = (
    "stable_normal",
    "noise_sensitive_normal",
    "lesion_candidate_preserved",
    "pseudo_anomaly_risk",
)
STATE_TITLES = (
    "Stable normal",
    "Noise-sensitive normal",
    "Lesion candidate",
    "Pseudo-anomaly risk",
)
STATE_COLORS = np.asarray(
    [
        (41, 166, 72),
        (245, 190, 36),
        (220, 50, 47),
        (143, 73, 201),
    ],
    dtype=np.uint8,
)


def resolve_checkpoint(root, checkpoint_name, description):
    path = Path(checkpoint_name)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_v1_image_checkpoint(
    model,
    path,
    device,
    prompt_source,
    dataset_name,
    llm_prompt_path,
):
    checkpoint = torch.load(path, map_location=device)
    validate_checkpoint_prompt_metadata(
        checkpoint,
        prompt_source,
        dataset_name,
        llm_prompt_path,
        f"image adapter checkpoint {path}",
    )
    if "image_adapter" not in checkpoint:
        raise KeyError(f"checkpoint does not contain 'image_adapter': {path}")
    state = checkpoint["image_adapter"]
    if not any(name.startswith("patch_graph.") for name in state):
        raise ValueError(f"checkpoint has no V1 patch graph parameters: {path}")
    checkpoint_global_weight = checkpoint.get("clip_global_weight")
    if (
        checkpoint_global_weight is not None
        and abs(float(checkpoint_global_weight) - model.clip_global_weight) > 1e-8
    ):
        raise ValueError(
            f"clip_global_weight mismatch for {path}: "
            f"checkpoint={checkpoint_global_weight}, "
            f"model={model.clip_global_weight}"
        )
    model.image_adapter.load_state_dict(state, strict=True)
    return int(checkpoint.get("epoch", -1))


def load_text_checkpoint(
    model,
    root,
    checkpoint_name,
    device,
    prompt_source,
    dataset_name,
    llm_prompt_path,
):
    if checkpoint_name.lower() == "none":
        return False, None
    path = resolve_checkpoint(root, checkpoint_name, "text adapter checkpoint")
    checkpoint = torch.load(path, map_location=device)
    validate_checkpoint_prompt_metadata(
        checkpoint,
        prompt_source,
        dataset_name,
        llm_prompt_path,
        f"text adapter checkpoint {path}",
    )
    if "text_adapter" not in checkpoint:
        raise KeyError(f"checkpoint does not contain 'text_adapter': {path}")
    model.text_adapter.load_state_dict(checkpoint["text_adapter"], strict=True)
    return True, path


def adapted_text_embedding(
    model,
    dataset_name,
    class_name,
    device,
    adapt_text,
    prompt_source,
    llm_prompt_path,
):
    """Local dependency-light equivalent of get_adapted_text_embedding()."""
    if class_name not in CLASS_NAMES[dataset_name]:
        raise ValueError(f"class {class_name!r} is not registered for {dataset_name}")
    real_name = REAL_NAMES[dataset_name][class_name]
    resolved_source = resolve_prompt_source(
        prompt_source,
        dataset_name,
        llm_prompt_path,
    )
    if resolved_source == "llm":
        prompted_states = get_llm_state_prompts(
            dataset_name,
            class_name,
            real_name,
            llm_prompt_path,
        )
    else:
        prompted_states = []
        state_prompts = [PROMPTS["prompt_normal"], PROMPTS["prompt_abnormal"]]
        for prompts in state_prompts:
            sentences = []
            for prompt in prompts:
                state = prompt.format(real_name)
                for template in PROMPTS["prompt_templates"]:
                    sentences.append(template.format(state))
            prompted_states.append(sentences)

    text_features = []
    for prompted_sentences in prompted_states:
        tokens = tokenize(prompted_sentences).to(device)
        embeddings = model.encode_text(tokens, adapt_text=adapt_text)
        embeddings = F.normalize(embeddings, dim=-1)
        embedding = F.normalize(embeddings.mean(dim=0), dim=0)
        text_features.append(embedding)
    return torch.stack(text_features, dim=1)


def stable_noise_seed(seed, file_name):
    value = f"{seed}:v1-patch-state:{file_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**31)


def deterministic_noise_view(image, dataset_name, severity, seed, file_name):
    cuda_devices = []
    if image.is_cuda:
        cuda_devices = [
            image.device.index
            if image.device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(stable_noise_seed(seed, file_name))
        return make_medical_noise_view(image, dataset_name, severity=severity)


def clip_tensor_to_rgb(image):
    mean = image.new_tensor(CLIP_MEAN).view(3, 1, 1)
    std = image.new_tensor(CLIP_STD).view(3, 1, 1)
    raw = (image * std + mean).clamp(0, 1)
    return (raw.permute(1, 2, 0).detach().cpu().numpy() * 255).round().astype(np.uint8)


def select_case(image_dataset, label_name, case_index, file_name):
    if file_name:
        matches = [
            index
            for index, metadata in enumerate(image_dataset.meta)
            if metadata["image_path"] == file_name
        ]
        if not matches:
            raise ValueError(f"file_name not found in metadata: {file_name}")
        return matches[0]

    label_value = {"all": None, "normal": 0, "anomaly": 1}[label_name]
    matches = [
        index
        for index, metadata in enumerate(image_dataset.meta)
        if label_value is None or int(metadata["label"]) == label_value
    ]
    if not matches:
        raise RuntimeError(f"no {label_name} cases are available")
    if case_index >= len(matches):
        raise IndexError(
            f"case_index={case_index} exceeds the {len(matches)} matching cases"
        )
    return matches[case_index]


def exact_graph_transition(block, patch_features, uncertainty, anomaly_probability):
    """Reproduce PatchGraphBlock's transition matrix without changing output."""
    batch_size, num_nodes, _ = patch_features.shape
    semantic_adj = _build_knn_patch_graph(patch_features, k=block.k)
    if block.use_spatial:
        grid_size = int(math.sqrt(num_nodes))
        if grid_size * grid_size == num_nodes:
            spatial_adj = _build_spatial_patch_graph(
                batch_size,
                grid_size,
                patch_features.device,
                semantic_adj.dtype,
            )
            adjacency = block.alpha * semantic_adj + (1.0 - block.alpha) * spatial_adj
        else:
            adjacency = semantic_adj
    else:
        adjacency = semantic_adj

    normalized = F.normalize(patch_features, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    feature_affinity = torch.exp(
        (similarity - 1.0) / max(block.feature_temperature, 1e-4)
    )
    anomaly_difference = (
        anomaly_probability.unsqueeze(-1) - anomaly_probability.unsqueeze(1)
    ).abs()
    boundary_affinity = torch.exp(
        -anomaly_difference / max(block.anomaly_temperature, 1e-4)
    )
    source_reliability = (1.0 - uncertainty).unsqueeze(1).clamp_min(0.05)
    weighted_adj = adjacency * feature_affinity * boundary_affinity * source_reliability
    eye = torch.eye(
        num_nodes,
        device=patch_features.device,
        dtype=patch_features.dtype,
    ).unsqueeze(0)
    weighted_adj = weighted_adj + eye
    return weighted_adj / weighted_adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def graph_gate(block, uncertainty, anomaly_probability):
    noise_scale = F.softplus(block.noise_gate_scale)
    anomaly_scale = F.softplus(block.anomaly_gate_scale)
    gate = torch.sigmoid(
        noise_scale * uncertainty
        - anomaly_scale * anomaly_probability
        + block.gate_bias
    )
    return gate, noise_scale, anomaly_scale


def normalize_values(values):
    minimum = float(values.min())
    maximum = float(values.max())
    return (values - minimum) / max(maximum - minimum, 1e-12)


def state_thresholds(uncertainty, anomaly, disagreement, args):
    if args.threshold_mode == "adaptive":
        return {
            "uncertainty": float(np.quantile(uncertainty, args.uncertainty_quantile)),
            "anomaly": float(np.quantile(anomaly, args.anomaly_quantile)),
            "disagreement": float(np.quantile(disagreement, args.instability_quantile)),
        }
    return {
        "uncertainty": args.uncertainty_threshold,
        "anomaly": args.anomaly_threshold,
        "disagreement": args.instability_threshold,
    }


def assign_posthoc_states(uncertainty, anomaly, disagreement, thresholds):
    high_uncertainty = uncertainty >= thresholds["uncertainty"]
    lesion_candidate = anomaly >= thresholds["anomaly"]
    unstable = disagreement >= thresholds["disagreement"]

    states = np.zeros(uncertainty.shape, dtype=np.int64)
    states[high_uncertainty & ~lesion_candidate] = 1
    states[lesion_candidate] = 2
    states[high_uncertainty & unstable & ~lesion_candidate] = 3
    return states


def representative_patches(
    states,
    uncertainty,
    anomaly,
    disagreement,
    primary_anomaly,
    reference_anomaly,
):
    u = normalize_values(uncertainty)
    d = normalize_values(disagreement)
    noisy_increase = np.maximum(reference_anomaly - primary_anomaly, 0.0)
    scores = (
        (1.0 - u) * (1.0 - anomaly),
        u * (1.0 - anomaly) * (1.0 - d),
        anomaly,
        u * (d + noisy_increase) * (1.0 - 0.5 * anomaly),
    )
    representatives = []
    for state_id, score in enumerate(scores):
        candidates = np.flatnonzero(states == state_id)
        matched = candidates.size > 0
        if not matched:
            candidates = np.arange(states.size)
        best = int(candidates[np.argmax(score[candidates])])
        representatives.append(
            {"state_id": state_id, "patch_index": best, "rule_matched": matched}
        )
    return representatives


def patch_grid_map(values, grid_size, output_size, nearest=False):
    grid = values.reshape(grid_size, grid_size).astype(np.float32)
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = Image.fromarray(grid, mode="F").resize(
        (output_size, output_size), resampling
    )
    return np.asarray(image, dtype=np.float32)


def colorize(values):
    values = np.clip(values, 0.0, 1.0)
    positions = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.asarray(
        [
            (31, 24, 82),
            (28, 104, 166),
            (42, 176, 137),
            (242, 211, 54),
            (184, 28, 28),
        ],
        dtype=np.float32,
    )
    output = np.empty(values.shape + (3,), dtype=np.float32)
    for channel in range(3):
        output[..., channel] = np.interp(values, positions, colors[:, channel])
    return output.round().astype(np.uint8)


def heatmap_overlay(image, values, alpha=0.5):
    colored = colorize(values)
    return np.clip(
        (1.0 - alpha) * image.astype(np.float32) + alpha * colored.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)


def mask_overlay(image, mask):
    output = image.astype(np.float32).copy()
    lesion = mask.astype(bool)
    if lesion.any():
        red = np.zeros_like(output)
        red[..., 0] = 255
        output[lesion] = 0.55 * output[lesion] + 0.45 * red[lesion]
    return np.clip(output, 0, 255).astype(np.uint8)


def state_overlay(image, states, grid_size, representatives):
    state_grid = states.reshape(grid_size, grid_size).astype(np.uint8)
    state_image = np.asarray(
        Image.fromarray(state_grid).resize(
            (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
        )
    )
    colors = STATE_COLORS[state_image]
    output = np.clip(
        0.48 * image.astype(np.float32) + 0.52 * colors.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return mark_representatives(output, grid_size, representatives)


def patch_bounds(index, grid_size, width, height):
    row, col = divmod(index, grid_size)
    x0 = round(col * width / grid_size)
    x1 = round((col + 1) * width / grid_size) - 1
    y0 = round(row * height / grid_size)
    y1 = round((row + 1) * height / grid_size) - 1
    return row, col, x0, y0, x1, y1


def mark_representatives(image, grid_size, representatives):
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for number, representative in enumerate(representatives, start=1):
        state_id = representative["state_id"]
        _, _, x0, y0, x1, y1 = patch_bounds(
            representative["patch_index"], grid_size, canvas.width, canvas.height
        )
        color = tuple(int(value) for value in STATE_COLORS[state_id])
        line_width = max(2, canvas.width // 180)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
        label_size = max(14, canvas.width // 24)
        label_box = (x0, max(0, y0 - label_size), x0 + label_size, y0)
        draw.rectangle(label_box, fill=color)
        draw.text((x0 + 3, max(0, y0 - label_size + 1)), str(number), fill="white")
    return np.asarray(canvas)


def add_panel(image, title, subtitle, panel_size):
    body = Image.fromarray(image).resize(
        (panel_size, panel_size), Image.Resampling.BILINEAR
    )
    header_height = 56
    canvas = Image.new("RGB", (panel_size, panel_size + header_height), "black")
    canvas.paste(body, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="white")
    if subtitle:
        draw.text((8, 30), subtitle[:58], fill=(205, 205, 205))
    return canvas


def legend_body(panel_size, thresholds, counts, threshold_mode):
    canvas = Image.new("RGB", (panel_size, panel_size), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)
    y = 12
    for state_id, title in enumerate(STATE_TITLES):
        color = tuple(int(value) for value in STATE_COLORS[state_id])
        draw.rectangle((10, y, 28, y + 18), fill=color)
        draw.text((36, y + 2), f"{state_id + 1}. {title}", fill="white")
        draw.text(
            (36, y + 18),
            f"patches={counts[state_id]}",
            fill=(190, 190, 190),
        )
        y += 46
    lines = [
        f"mode={threshold_mode}",
        f"U >= {thresholds['uncertainty']:.4f}: noise-sensitive",
        f"A >= {thresholds['anomaly']:.4f}: lesion candidate",
        f"D >= {thresholds['disagreement']:.4f}: unstable",
        "Colors are post-hoc only.",
    ]
    y += 4
    for line in lines:
        for wrapped in textwrap.wrap(line, width=42):
            draw.text((10, y), wrapped, fill=(220, 220, 220))
            y += 16
    return np.asarray(canvas)


def compose_grid(panels, columns, gap=8):
    rows = math.ceil(len(panels) / columns)
    panel_width = panels[0].width
    panel_height = panels[0].height
    canvas = Image.new(
        "RGB",
        (
            columns * panel_width + (columns - 1) * gap,
            rows * panel_height + (rows - 1) * gap,
        ),
        (35, 35, 35),
    )
    for index, panel in enumerate(panels):
        row, col = divmod(index, columns)
        canvas.paste(panel, (col * (panel_width + gap), row * (panel_height + gap)))
    return canvas


def crop_patch_context(image, patch_index, grid_size, radius, output_size, color):
    row, col = divmod(patch_index, grid_size)
    row0 = max(0, row - radius)
    row1 = min(grid_size, row + radius + 1)
    col0 = max(0, col - radius)
    col1 = min(grid_size, col + radius + 1)
    height, width = image.shape[:2]
    x0 = round(col0 * width / grid_size)
    x1 = round(col1 * width / grid_size)
    y0 = round(row0 * height / grid_size)
    y1 = round(row1 * height / grid_size)
    crop = Image.fromarray(image[y0:y1, x0:x1]).resize(
        (output_size, output_size), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(crop)
    cell_width = output_size / (col1 - col0)
    cell_height = output_size / (row1 - row0)
    center_x0 = round((col - col0) * cell_width)
    center_x1 = round((col - col0 + 1) * cell_width) - 1
    center_y0 = round((row - row0) * cell_height)
    center_y1 = round((row - row0 + 1) * cell_height) - 1
    draw.rectangle((center_x0, center_y0, center_x1, center_y1), outline=color, width=3)
    return crop


def create_example_card(
    number,
    representative,
    clean,
    noisy,
    transition,
    metrics,
    grid_size,
    crop_radius,
):
    state_id = representative["state_id"]
    color = tuple(int(value) for value in STATE_COLORS[state_id])
    card_width = 420
    tile_size = 124
    card_height = 300
    card = Image.new("RGB", (card_width, card_height), (20, 20, 20))
    draw = ImageDraw.Draw(card)
    matched = "rule matched" if representative["rule_matched"] else "closest fallback"
    draw.rectangle((0, 0, card_width - 1, card_height - 1), outline=color, width=3)
    draw.text(
        (10, 9),
        f"{number}. {STATE_TITLES[state_id]} ({matched})",
        fill="white",
    )

    clean_crop = crop_patch_context(
        clean,
        representative["patch_index"],
        grid_size,
        crop_radius,
        tile_size,
        color,
    )
    noisy_crop = crop_patch_context(
        noisy,
        representative["patch_index"],
        grid_size,
        crop_radius,
        tile_size,
        color,
    )
    source_map = normalize_values(transition[representative["patch_index"]]).reshape(
        grid_size, grid_size
    )
    source_rgb = Image.fromarray(colorize(source_map)).resize(
        (tile_size, tile_size), Image.Resampling.NEAREST
    )
    positions = (8, 148, 288)
    for x, tile, title in zip(
        positions,
        (clean_crop, noisy_crop, source_rgb),
        ("Clean crop", "Noise crop", "Source weights"),
    ):
        card.paste(tile, (x, 42))
        draw.text((x, 171), title, fill=(210, 210, 210))

    row = metrics[representative["patch_index"]]
    lines = [
        f"grid=({int(row['row'])},{int(row['col'])})  U={row['uncertainty']:.4f}  A={row['anomaly_probability']:.4f}",
        f"D={row['prediction_disagreement']:.4f}  gate={row['update_gate']:.4f}  reliability={row['source_reliability']:.4f}",
        f"clean_A={row['primary_anomaly_probability']:.4f}  noise_A={row['reference_anomaly_probability']:.4f}",
        f"self_T={row['transition_self_weight']:.4f}  nonself_T={row['transition_nonself_weight']:.4f}  GT={row['gt_lesion_fraction']:.3f}",
    ]
    y = 202
    for line in lines:
        draw.text((10, y), line, fill=(225, 225, 225))
        y += 21
    return card


def write_patch_csv(output_path, metrics, states, representatives):
    representative_lookup = {
        representative["patch_index"]: number
        for number, representative in enumerate(representatives, start=1)
    }
    fieldnames = [
        "patch_index",
        "row",
        "col",
        "posthoc_state",
        "representative_number",
        "uncertainty",
        "anomaly_probability",
        "primary_anomaly_probability",
        "reference_anomaly_probability",
        "prediction_disagreement",
        "source_reliability",
        "update_gate",
        "gt_lesion_fraction",
        "transition_self_weight",
        "transition_nonself_weight",
        "top_source_patch",
        "top_source_weight",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for patch_index, row in enumerate(metrics):
            record = dict(row)
            record["posthoc_state"] = STATE_NAMES[int(states[patch_index])]
            record["representative_number"] = representative_lookup.get(patch_index, "")
            writer.writerow({key: record[key] for key in fieldnames})


def safe_stem(file_name):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(Path(file_name).with_suffix("")))
    digest = hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:8]
    return f"{stem[-80:] or 'case'}_{digest}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize real V1 patch uncertainty, graph gates, and states"
    )
    parser.add_argument("--dataset", type=str, default="Liver")
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument(
        "--label", choices=["all", "normal", "anomaly"], default="anomaly"
    )
    parser.add_argument(
        "--case_index",
        type=int,
        default=0,
        help="zero-based index among cases matching --label",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default=None,
        help="exact metadata image_path; overrides --label and --case_index",
    )
    parser.add_argument("--save_path", type=str, default="ckpt/noise_graph_cls_llm")
    parser.add_argument("--image_checkpoint", type=str, default="image_adapter_1.pth")
    parser.add_argument("--text_checkpoint", type=str, default="text_adapter.pth")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--prompt_source",
        choices=PROMPT_SOURCES,
        default="llm",
        help="prompt source used to train the V1 checkpoint",
    )
    parser.add_argument(
        "--llm_prompt_path",
        type=str,
        default=str(DEFAULT_LLM_PROMPT_PATH),
    )

    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--feature_level", type=int, default=-1)
    parser.add_argument("--noise_severity", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--panel_size", type=int, default=300)
    parser.add_argument("--crop_radius", type=int, default=2)
    parser.add_argument("--difference_gain", type=float, default=4.0)

    parser.add_argument(
        "--threshold_mode",
        choices=["adaptive", "fixed"],
        default="adaptive",
        help="adaptive quantiles are convenient for one-case explanation",
    )
    parser.add_argument("--uncertainty_quantile", type=float, default=0.75)
    parser.add_argument("--anomaly_quantile", type=float, default=0.85)
    parser.add_argument("--instability_quantile", type=float, default=0.75)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.10)
    parser.add_argument("--anomaly_threshold", type=float, default=0.50)
    parser.add_argument("--instability_threshold", type=float, default=0.10)

    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    parser.add_argument("--patch_graph_k", type=int, default=8)
    parser.add_argument("--patch_graph_alpha", type=float, default=0.7)
    parser.add_argument("--patch_graph_residual_weight", type=float, default=0.2)
    parser.add_argument("--disable_patch_graph_spatial", action="store_true")
    parser.add_argument("--patch_graph_feature_temperature", type=float, default=0.2)
    parser.add_argument("--patch_graph_anomaly_temperature", type=float, default=0.2)
    parser.add_argument(
        "--clip_global_weight",
        type=float,
        default=0.2,
        help="weight of the final CLIP CLS feature in image-level fusion",
    )
    args = parser.parse_args()

    if args.dataset not in DOMAINS:
        parser.error(f"unknown dataset: {args.dataset}")
    if DOMAINS[args.dataset] != "Medical":
        parser.error("this explanation script supports medical datasets only")
    if args.class_name is not None and args.class_name not in CLASS_NAMES[args.dataset]:
        parser.error(f"unknown class_name for {args.dataset}: {args.class_name}")
    if args.case_index < 0:
        parser.error("case_index must be non-negative")
    if args.img_size < 1 or args.panel_size < 160:
        parser.error("img_size must be positive and panel_size at least 160")
    if args.noise_severity < 0:
        parser.error("noise_severity must be non-negative")
    if args.crop_radius < 0 or args.difference_gain <= 0:
        parser.error("crop_radius must be non-negative and difference_gain positive")
    for name in (
        "uncertainty_quantile",
        "anomaly_quantile",
        "instability_quantile",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"{name} must be in [0, 1]")
    for name in (
        "uncertainty_threshold",
        "anomaly_threshold",
        "instability_threshold",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"{name} must be in [0, 1]")
    if not 0.0 <= args.clip_global_weight <= 1.0:
        parser.error("clip_global_weight must be in [0, 1]")
    return args


def main():
    args = parse_args()
    # Delay the heavy CLIP import so ``--help`` and argument validation do not
    # require every optional dependency used by the original model package.
    from model.clip import create_model

    setup_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    image_checkpoint = resolve_checkpoint(
        args.save_path, args.image_checkpoint, "V1 image adapter checkpoint"
    )
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
        enable_patch_graph=True,
        patch_graph_k=args.patch_graph_k,
        patch_graph_alpha=args.patch_graph_alpha,
        patch_graph_residual_weight=args.patch_graph_residual_weight,
        patch_graph_use_spatial=not args.disable_patch_graph_spatial,
        patch_graph_feature_temperature=args.patch_graph_feature_temperature,
        patch_graph_anomaly_temperature=args.patch_graph_anomaly_temperature,
        clip_global_weight=args.clip_global_weight,
    ).to(device)
    prompt_metadata = prompt_checkpoint_metadata(
        args.prompt_source,
        args.dataset,
        args.llm_prompt_path,
    )
    epoch = load_v1_image_checkpoint(
        model,
        image_checkpoint,
        device,
        args.prompt_source,
        args.dataset,
        args.llm_prompt_path,
    )
    adapt_text, text_checkpoint = load_text_checkpoint(
        model,
        args.save_path,
        args.text_checkpoint,
        device,
        args.prompt_source,
        args.dataset,
        args.llm_prompt_path,
    )
    model.eval()

    image_datasets = get_dataset(
        args.dataset,
        args.img_size,
        None,
        -1,
        "test",
        logger=None,
    )
    class_name = args.class_name or CLASS_NAMES[args.dataset][0]
    image_dataset = image_datasets[class_name]
    dataset_index = select_case(
        image_dataset, args.label, args.case_index, args.file_name
    )
    item = image_dataset[dataset_index]
    file_name = item["file_name"]
    image = item["image"].unsqueeze(0).to(device)
    mask = item["mask"].unsqueeze(0).to(device)
    reference_image = deterministic_noise_view(
        image,
        args.dataset,
        args.noise_severity,
        args.seed,
        file_name,
    )

    with torch.inference_mode():
        text_embeddings = adapted_text_embedding(
            model,
            args.dataset,
            class_name,
            device,
            adapt_text,
            args.prompt_source,
            args.llm_prompt_path,
        )
        refined_features, _, auxiliary = model(
            image,
            reference_image=reference_image,
            text_embeddings=text_embeddings,
            return_aux=True,
        )
        level_count = len(auxiliary["uncertainty"])
        level_index = args.feature_level % level_count
        uncertainty_tensor = auxiliary["uncertainty"][level_index][0]
        anomaly_tensor = auxiliary["anomaly_probability"][level_index][0]
        primary = auxiliary["primary_features"][level_index]
        reference = auxiliary["reference_features"][level_index]
        graph_input = auxiliary["graph_input_features"][level_index]
        primary_anomaly = model._anomaly_probability(primary, text_embeddings)[0]
        reference_anomaly = model._anomaly_probability(reference, text_embeddings)[0]
        disagreement_tensor = (primary_anomaly - reference_anomaly).abs()
        block = model.image_adapter["patch_graph"]
        gate_tensor, noise_scale, anomaly_scale = graph_gate(
            block, uncertainty_tensor, anomaly_tensor
        )
        reliability_tensor = (1.0 - uncertainty_tensor).clamp_min(0.05)
        transition_tensor = exact_graph_transition(
            block,
            graph_input,
            auxiliary["uncertainty"][level_index],
            auxiliary["anomaly_probability"][level_index],
        )[0]
        refined_anomaly_tensor = model._anomaly_probability(
            refined_features[level_index], text_embeddings
        )[0]

    num_patches = uncertainty_tensor.numel()
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise RuntimeError(f"patch count {num_patches} is not a square grid")

    gt_fraction = F.adaptive_avg_pool2d(mask.float(), (grid_size, grid_size))[0, 0]
    uncertainty = uncertainty_tensor.float().cpu().numpy()
    anomaly = anomaly_tensor.float().cpu().numpy()
    primary_anomaly_values = primary_anomaly.float().cpu().numpy()
    reference_anomaly_values = reference_anomaly.float().cpu().numpy()
    disagreement = disagreement_tensor.float().cpu().numpy()
    gate = gate_tensor.float().cpu().numpy()
    reliability = reliability_tensor.float().cpu().numpy()
    transition = transition_tensor.float().cpu().numpy()
    refined_anomaly = refined_anomaly_tensor.float().cpu().numpy()
    gt_values = gt_fraction.float().cpu().numpy().reshape(-1)

    thresholds = state_thresholds(uncertainty, anomaly, disagreement, args)
    states = assign_posthoc_states(uncertainty, anomaly, disagreement, thresholds)
    representatives = representative_patches(
        states,
        uncertainty,
        anomaly,
        disagreement,
        primary_anomaly_values,
        reference_anomaly_values,
    )

    top_sources = transition.argmax(axis=1)
    metrics = []
    for patch_index in range(num_patches):
        top_source = int(top_sources[patch_index])
        metrics.append(
            {
                "patch_index": patch_index,
                "row": patch_index // grid_size,
                "col": patch_index % grid_size,
                "uncertainty": float(uncertainty[patch_index]),
                "anomaly_probability": float(anomaly[patch_index]),
                "primary_anomaly_probability": float(
                    primary_anomaly_values[patch_index]
                ),
                "reference_anomaly_probability": float(
                    reference_anomaly_values[patch_index]
                ),
                "prediction_disagreement": float(disagreement[patch_index]),
                "source_reliability": float(reliability[patch_index]),
                "update_gate": float(gate[patch_index]),
                "gt_lesion_fraction": float(gt_values[patch_index]),
                "transition_self_weight": float(transition[patch_index, patch_index]),
                "transition_nonself_weight": float(
                    1.0 - transition[patch_index, patch_index]
                ),
                "top_source_patch": top_source,
                "top_source_weight": float(transition[patch_index, top_source]),
            }
        )

    clean_rgb = clip_tensor_to_rgb(image[0])
    noisy_rgb = clip_tensor_to_rgb(reference_image[0])
    absolute_rgb_difference = np.abs(
        clean_rgb.astype(np.float32) / 255.0 - noisy_rgb.astype(np.float32) / 255.0
    )
    difference = absolute_rgb_difference.mean(axis=-1)
    pixel_mad = float(absolute_rgb_difference.mean())
    pixel_max_difference = float(absolute_rgb_difference.max())
    changed_pixel_fraction = float(
        (absolute_rgb_difference.max(axis=-1) > (1.0 / 255.0)).mean()
    )
    difference_display = colorize(np.clip(difference * args.difference_gain, 0.0, 1.0))
    mask_array = mask[0, 0].detach().cpu().numpy() > 0

    u_map = patch_grid_map(uncertainty, grid_size, args.img_size)
    a_map = patch_grid_map(anomaly, grid_size, args.img_size)
    d_map = patch_grid_map(disagreement, grid_size, args.img_size)
    gate_map = patch_grid_map(gate, grid_size, args.img_size)
    reliability_map = patch_grid_map(reliability, grid_size, args.img_size)
    refined_map = patch_grid_map(refined_anomaly, grid_size, args.img_size)
    counts = np.bincount(states, minlength=len(STATE_NAMES))

    marked_clean = mark_representatives(clean_rgb, grid_size, representatives)
    panels = [
        add_panel(
            marked_clean,
            "Clean primary + examples",
            f"{file_name}  label={int(item['label'])}",
            args.panel_size,
        ),
        add_panel(
            noisy_rgb,
            "V1 auxiliary noise view",
            f"{noise_model_for_dataset(args.dataset)}, severity={args.noise_severity:g}",
            args.panel_size,
        ),
        add_panel(
            difference_display,
            "Clean/noise difference",
            f"absolute RGB difference, display gain={args.difference_gain:g}x",
            args.panel_size,
        ),
        add_panel(
            mask_overlay(clean_rgb, mask_array),
            "Ground truth (diagnostic only)",
            "GT never enters V1 inference",
            args.panel_size,
        ),
        add_panel(
            heatmap_overlay(clean_rgb, u_map),
            "Feature uncertainty U",
            f"mean={uncertainty.mean():.4f}, threshold={thresholds['uncertainty']:.4f}",
            args.panel_size,
        ),
        add_panel(
            heatmap_overlay(clean_rgb, a_map),
            "Fused anomaly probability A",
            f"mean={anomaly.mean():.4f}, threshold={thresholds['anomaly']:.4f}",
            args.panel_size,
        ),
        add_panel(
            heatmap_overlay(clean_rgb, d_map),
            "Prediction disagreement D",
            f"mean={disagreement.mean():.4f}, threshold={thresholds['disagreement']:.4f}",
            args.panel_size,
        ),
        add_panel(
            heatmap_overlay(clean_rgb, gate_map),
            "Learned update gate g",
            f"mean={gate.mean():.4f}; larger = more graph update",
            args.panel_size,
        ),
        add_panel(
            heatmap_overlay(clean_rgb, reliability_map),
            "Source reliability max(1-U,.05)",
            f"mean={reliability.mean():.4f}; larger = stronger source",
            args.panel_size,
        ),
        add_panel(
            state_overlay(clean_rgb, states, grid_size, representatives),
            "Four post-hoc patch states",
            "continuous U/A/D converted to explanatory colors",
            args.panel_size,
        ),
        add_panel(
            heatmap_overlay(clean_rgb, refined_map),
            "Refined anomaly probability",
            f"selected feature level={level_index}",
            args.panel_size,
        ),
        add_panel(
            legend_body(args.panel_size, thresholds, counts, args.threshold_mode),
            "State legend and thresholds",
            "not an extra V1 classifier",
            args.panel_size,
        ),
    ]

    if args.output_dir is None:
        output_dir = Path("patch_state_examples") / args.dataset / safe_stem(file_name)
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the two inputs at their actual resolution. The overview uses
    # smaller panels, whose bilinear downsampling can visually suppress
    # high-frequency medical noise.
    clean_path = output_dir / "clean_primary.png"
    noisy_path = output_dir / "auxiliary_noise_view.png"
    difference_path = output_dir / "absolute_difference_amplified.png"
    Image.fromarray(clean_rgb).save(clean_path)
    Image.fromarray(noisy_rgb).save(noisy_path)
    Image.fromarray(difference_display).save(difference_path)

    full_resolution_panels = [
        add_panel(
            clean_rgb,
            "Clean primary",
            f"model input, {args.img_size}x{args.img_size}",
            args.img_size,
        ),
        add_panel(
            noisy_rgb,
            "V1 auxiliary noise view",
            f"severity={args.noise_severity:g}, MAD={pixel_mad:.5f}",
            args.img_size,
        ),
        add_panel(
            difference_display,
            "Absolute difference",
            f"display gain={args.difference_gain:g}x, max={pixel_max_difference:.5f}",
            args.img_size,
        ),
    ]
    full_resolution_path = output_dir / "noise_comparison_full_resolution.png"
    compose_grid(full_resolution_panels, columns=3).save(full_resolution_path)

    blink_path = output_dir / "clean_noise_blink.gif"
    clean_frame = Image.fromarray(clean_rgb)
    noisy_frame = Image.fromarray(noisy_rgb)
    clean_frame.save(
        blink_path,
        save_all=True,
        append_images=[noisy_frame],
        duration=650,
        loop=0,
    )

    overview_path = output_dir / "patch_state_overview.png"
    compose_grid(panels, columns=4).save(overview_path)

    cards = [
        create_example_card(
            number,
            representative,
            clean_rgb,
            noisy_rgb,
            transition,
            metrics,
            grid_size,
            args.crop_radius,
        )
        for number, representative in enumerate(representatives, start=1)
    ]
    examples_path = output_dir / "representative_patch_examples.png"
    compose_grid(cards, columns=2, gap=10).save(examples_path)
    csv_path = output_dir / "patch_metrics.csv"
    write_patch_csv(csv_path, metrics, states, representatives)

    info_lines = [
        "V1 patch-state visualization",
        f"dataset: {args.dataset}",
        f"class: {class_name}",
        f"file: {file_name}",
        f"dataset index: {dataset_index}",
        f"image label: {int(item['label'])}",
        f"image checkpoint: {image_checkpoint}",
        f"checkpoint epoch: {epoch}",
        f"CLIP global feature weight: {args.clip_global_weight}",
        f"text checkpoint: {text_checkpoint or 'none (original CLIP text)'}",
        f"prompt source: {prompt_metadata['prompt_source']}",
        f"LLM prompt bank: {args.llm_prompt_path}",
        f"LLM prompt bank SHA256: {prompt_metadata.get('llm_prompt_bank_sha256', 'n/a')}",
        f"feature level: {level_index} of {level_count}",
        f"patch grid: {grid_size} x {grid_size}",
        f"noise model: {noise_model_for_dataset(args.dataset)}",
        f"noise severity: {args.noise_severity}",
        f"noise seed: {stable_noise_seed(args.seed, file_name)}",
        f"pixel mean absolute difference: {pixel_mad}",
        f"pixel maximum absolute difference: {pixel_max_difference}",
        f"fraction of pixels changed by more than 1/255: {changed_pixel_fraction}",
        f"threshold mode: {args.threshold_mode}",
        f"uncertainty threshold: {thresholds['uncertainty']}",
        f"anomaly threshold: {thresholds['anomaly']}",
        f"disagreement threshold: {thresholds['disagreement']}",
        f"learned noise gate scale (softplus): {float(noise_scale):.6f}",
        f"learned anomaly gate scale (softplus): {float(anomaly_scale):.6f}",
        f"learned gate bias: {float(block.gate_bias):.6f}",
        f"patch graph k: {block.k}",
        f"patch graph alpha: {block.alpha}",
        f"feature temperature: {block.feature_temperature}",
        f"anomaly temperature: {block.anomaly_temperature}",
        "",
        "State rules used only for this visualization:",
        "stable_normal: U below threshold and A below threshold.",
        "noise_sensitive_normal: U high, A low, and D below threshold.",
        "lesion_candidate_preserved: A above threshold (regardless of U).",
        "pseudo_anomaly_risk: U high, A below threshold, and D high.",
        "",
        "V1 itself uses continuous U and A; it does not predict these four hard labels.",
        "Adaptive thresholds are per-case quantiles and must not be presented as learned thresholds.",
        "Ground truth is shown and exported only for post-hoc diagnosis; it never enters inference.",
        "Source-weight maps are exact rows of the V1 graph transition matrix before projection.",
        "Update gate g controls how strongly each receiver uses the projected graph feature.",
    ]
    info_path = output_dir / "analysis_info.txt"
    info_path.write_text("\n".join(info_lines) + "\n", encoding="utf-8")

    print(f"case: {file_name}")
    print(f"checkpoint epoch: {epoch}")
    print(f"patch grid: {grid_size} x {grid_size}")
    print(
        "clean/noise pixel difference: "
        f"MAD={pixel_mad:.6f}, max={pixel_max_difference:.6f}, "
        f"changed>{1 / 255:.6f}={changed_pixel_fraction:.2%}"
    )
    for state_id, name in enumerate(STATE_NAMES):
        print(f"{name}: {counts[state_id]} patch(es)")
    print(f"overview: {overview_path.resolve()}")
    print(f"full-resolution noise comparison: {full_resolution_path.resolve()}")
    print(f"clean/noise blink: {blink_path.resolve()}")
    print(f"representatives: {examples_path.resolve()}")
    print(f"patch metrics: {csv_path.resolve()}")
    print(f"analysis notes: {info_path.resolve()}")


if __name__ == "__main__":
    main()
