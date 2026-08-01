"""Paired medical case analysis for the original AA-CLIP and V1.

AA-CLIP keeps its original medical image score based on the pixel-map maximum.
V1 additionally fuses the normalized CLS-graph global score using the same
weight as ``test.py``. The script evaluates both methods on exactly the same
input tensor, writes per-case statistics, selects representative (median rather
than best) cases, and creates paper-ready panels.

Example on Windows::

    python case_analysis.py ^
      --dataset Liver ^
      --aa_save_path ./ckpt/baseline ^
      --aa_image_checkpoint image_adapter_1.pth ^
      --v1_save_path ./ckpt/noise_graph_cls ^
      --v1_image_checkpoint image_adapter_1.pth ^
      --output_dir ./case_results/Liver

Use the checkpoints selected by a validation protocol. Do not choose either
checkpoint from the target test cases produced by this script.
"""

import argparse
import hashlib
import logging
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DOMAINS, get_dataset
from dataset.constants import DATA_PATH
from forward_utils import calculate_similarity_map, get_adapted_text_embedding
from model.adapter import AdaptedCLIP
from model.clip import create_model
from utils import CLIP_MEAN, CLIP_STD, make_medical_noise_view, setup_seed


def resolve_checkpoint(root, checkpoint_name, description):
    path = Path(checkpoint_name)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def checkpoint_state(checkpoint, key, path):
    if key not in checkpoint:
        raise KeyError(f"checkpoint {path} does not contain {key!r}")
    return checkpoint[key]


def load_image_checkpoint(model, path, device, expect_graph):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint_state(checkpoint, "image_adapter", path)
    has_graph_parameters = any(name.startswith("patch_graph.") for name in state)
    if expect_graph and not has_graph_parameters:
        raise ValueError(
            f"V1 checkpoint {path} has no patch_graph parameters; "
            "select a checkpoint trained by the V1 graph model"
        )
    if not expect_graph and has_graph_parameters:
        raise ValueError(
            f"AA-CLIP checkpoint {path} contains patch_graph parameters; "
            "select the original AA-CLIP baseline checkpoint"
        )
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


def load_text_checkpoint(model, root, checkpoint_name, device, description):
    if checkpoint_name.lower() == "none":
        return False, None
    path = resolve_checkpoint(root, checkpoint_name, description)
    checkpoint = torch.load(path, map_location=device)
    model.text_adapter.load_state_dict(
        checkpoint_state(checkpoint, "text_adapter", path), strict=True
    )
    return True, path


def stable_noise_seed(seed, file_name, stream):
    value = f"{seed}:{stream}:{file_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**31)


def deterministic_noise_view(image, file_names, dataset, severity, seed, stream):
    """Generate per-file noise whose realization is independent of batching."""
    if severity == 0:
        return image.detach().clone()
    views = []
    cuda_devices = []
    if image.is_cuda:
        cuda_devices = [
            image.device.index
            if image.device.index is not None
            else torch.cuda.current_device()
        ]
    for sample, file_name in zip(image.split(1, dim=0), file_names):
        sample_seed = stable_noise_seed(seed, file_name, stream)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(sample_seed)
            views.append(make_medical_noise_view(sample, dataset, severity=severity))
    return torch.cat(views, dim=0)


def patch_anomaly_map(patch_features, text_embeddings, img_size, dataset):
    maps = [
        calculate_similarity_map(
            features,
            text_embeddings,
            img_size,
            test=True,
            domain=DOMAINS[dataset],
        )
        for features in patch_features
    ]
    return torch.cat(maps, dim=1).sum(dim=1)


def uncertainty_map(uncertainty_levels, img_size, batch_size, device):
    maps = []
    for uncertainty in uncertainty_levels:
        if uncertainty is None:
            continue
        num_patches = uncertainty.shape[1]
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            continue
        grid = uncertainty.view(batch_size, 1, grid_size, grid_size)
        maps.append(
            F.interpolate(
                grid,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=True,
            )
        )
    if not maps:
        return torch.zeros(batch_size, img_size, img_size, device=device)
    return torch.stack(maps, dim=0).mean(dim=0)[:, 0]


@torch.inference_mode()
def predict_pair(
    aa_model,
    v1_model,
    image,
    file_names,
    aa_text_embeddings,
    v1_text_embeddings,
    args,
):
    evaluated_image = deterministic_noise_view(
        image,
        file_names,
        args.dataset,
        args.test_noise_severity,
        args.seed,
        "primary",
    )

    aa_patch_features, aa_det_feature = aa_model(
        evaluated_image,
        text_embeddings=aa_text_embeddings,
    )
    aa_map = patch_anomaly_map(
        aa_patch_features, aa_text_embeddings, args.img_size, args.dataset
    )
    aa_det_logits = aa_det_feature @ aa_text_embeddings
    aa_det_score = (aa_det_logits[:, 1] + 1.0) / 2.0
    del aa_patch_features, aa_det_feature, aa_det_logits

    reference_image = deterministic_noise_view(
        evaluated_image,
        file_names,
        args.dataset,
        args.noise_severity,
        args.seed,
        "v1_probe",
    )
    v1_patch_features, v1_det_feature, auxiliary = v1_model(
        evaluated_image,
        reference_image=reference_image,
        text_embeddings=v1_text_embeddings,
        return_aux=True,
    )
    v1_map = patch_anomaly_map(
        v1_patch_features, v1_text_embeddings, args.img_size, args.dataset
    )
    v1_det_logits = v1_det_feature @ v1_text_embeddings
    v1_det_score = (v1_det_logits[:, 1] + 1.0) / 2.0
    v1_uncertainty = uncertainty_map(
        auxiliary["uncertainty"],
        args.img_size,
        evaluated_image.shape[0],
        evaluated_image.device,
    )
    return {
        "evaluated_image": evaluated_image,
        "aa_map": aa_map,
        "v1_map": v1_map,
        "aa_det_score": aa_det_score,
        "v1_det_score": v1_det_score,
        "v1_uncertainty": v1_uncertainty,
    }


def update_bounds(bounds, name, values):
    bounds[name][0] = min(bounds[name][0], float(values.min()))
    bounds[name][1] = max(bounds[name][1], float(values.max()))


def region_statistics(score_map, mask):
    lesion = mask.astype(bool)
    background = ~lesion
    lesion_mean = float(score_map[lesion].mean()) if lesion.any() else np.nan
    background_mean = (
        float(score_map[background].mean()) if background.any() else np.nan
    )
    return lesion_mean, background_mean


def collect_case_records(
    aa_model,
    v1_model,
    image_datasets,
    aa_text_embeddings,
    v1_text_embeddings,
    device,
    args,
):
    records = []
    bounds = {"aa": [float("inf"), float("-inf")], "v1": [float("inf"), float("-inf")]}
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    for class_name, image_dataset in image_datasets.items():
        loader = DataLoader(image_dataset, **dataloader_kwargs)
        dataset_index = 0
        progress = tqdm(loader, desc=f"case analysis: {class_name}")
        for input_data in progress:
            image = input_data["image"].to(device, non_blocking=True)
            masks = input_data["mask"].cpu().numpy()[:, 0]
            labels = input_data["label"].cpu().numpy().astype(int)
            file_names = list(input_data["file_name"])
            outputs = predict_pair(
                aa_model,
                v1_model,
                image,
                file_names,
                aa_text_embeddings[class_name],
                v1_text_embeddings[class_name],
                args,
            )
            aa_maps = outputs["aa_map"].cpu().numpy()
            v1_maps = outputs["v1_map"].cpu().numpy()
            aa_det_scores = outputs["aa_det_score"].cpu().numpy()
            v1_det_scores = outputs["v1_det_score"].cpu().numpy()
            update_bounds(bounds, "aa", aa_maps)
            update_bounds(bounds, "v1", v1_maps)

            for batch_index, file_name in enumerate(file_names):
                mask = masks[batch_index]
                aa_lesion, aa_background = region_statistics(aa_maps[batch_index], mask)
                v1_lesion, v1_background = region_statistics(v1_maps[batch_index], mask)
                records.append(
                    {
                        "record_id": f"{class_name}:{dataset_index + batch_index}",
                        "class_name": class_name,
                        "dataset_index": dataset_index + batch_index,
                        "file_name": file_name,
                        "label": int(labels[batch_index]),
                        "lesion_fraction": float((mask != 0).mean()),
                        "aa_raw_image_score": float(aa_maps[batch_index].max()),
                        "v1_raw_image_score": float(v1_maps[batch_index].max()),
                        "aa_raw_lesion_mean": aa_lesion,
                        "v1_raw_lesion_mean": v1_lesion,
                        "aa_raw_background_mean": aa_background,
                        "v1_raw_background_mean": v1_background,
                        "aa_det_score_diagnostic": float(aa_det_scores[batch_index]),
                        "v1_det_score_diagnostic": float(v1_det_scores[batch_index]),
                    }
                )
            dataset_index += len(file_names)
    return pd.DataFrame(records), bounds


def normalize_column(values, minimum, maximum):
    scale = max(maximum - minimum, 1e-12)
    return (values - minimum) / scale


def finalize_case_records(records, bounds, args):
    for name in ("aa", "v1"):
        minimum, maximum = bounds[name]
        records[f"{name}_image_score"] = normalize_column(
            records[f"{name}_raw_image_score"], minimum, maximum
        )
        records[f"{name}_lesion_mean"] = normalize_column(
            records[f"{name}_raw_lesion_mean"], minimum, maximum
        )
        records[f"{name}_background_mean"] = normalize_column(
            records[f"{name}_raw_background_mean"], minimum, maximum
        )
        records[f"{name}_lesion_contrast"] = (
            records[f"{name}_lesion_mean"] - records[f"{name}_background_mean"]
        )
        records[f"{name}_image_percentile"] = records.groupby("class_name")[
            f"{name}_image_score"
        ].rank(method="average", pct=True)

    for name in ("aa", "v1"):
        raw_global = records[f"{name}_det_score_diagnostic"]
        records[f"{name}_global_image_score"] = normalize_column(
            raw_global,
            float(raw_global.min()),
            float(raw_global.max()),
        )
    records["v1_pixelmax_image_score"] = records["v1_image_score"]
    records["v1_image_score"] = (
        1.0 - args.medical_image_score_global_weight
    ) * records[
        "v1_pixelmax_image_score"
    ] + args.medical_image_score_global_weight * records["v1_global_image_score"]
    records["v1_image_percentile"] = records.groupby("class_name")[
        "v1_image_score"
    ].rank(method="average", pct=True)

    records["v1_minus_aa_image_score"] = (
        records["v1_image_score"] - records["aa_image_score"]
    )
    records["v1_minus_aa_image_percentile"] = (
        records["v1_image_percentile"] - records["aa_image_percentile"]
    )
    records["correct_direction_rank_gain"] = np.where(
        records["label"] == 1,
        records["v1_minus_aa_image_percentile"],
        -records["v1_minus_aa_image_percentile"],
    )
    records["v1_minus_aa_lesion_contrast"] = (
        records["v1_lesion_contrast"] - records["aa_lesion_contrast"]
    )
    return records


def representative_rows(candidates, score_column, count, used_ids):
    candidates = candidates[~candidates["record_id"].isin(used_ids)].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(score_column).reset_index(drop=True)
    take = min(count, len(candidates))
    positions = np.linspace(0, len(candidates) - 1, take + 2)[1:-1]
    positions = np.unique(np.rint(positions).astype(int))
    return candidates.iloc[positions]


def select_representative_cases(records, args, logger):
    selected = []
    used_ids = set()
    abnormal = records[records["label"] == 1]
    normal = records[records["label"] == 0]

    definitions = []
    definitions.append(
        (
            "abnormal_rank_improvement",
            abnormal[
                (abnormal["v1_minus_aa_image_percentile"] > 0)
                & (abnormal["v1_minus_aa_lesion_contrast"] >= -args.contrast_tolerance)
            ],
            "v1_minus_aa_image_percentile",
            "median abnormal ranking improvement with retained lesion contrast",
        )
    )
    definitions.append(
        (
            "normal_false_positive_suppression",
            normal[normal["v1_minus_aa_image_percentile"] < 0],
            "correct_direction_rank_gain",
            "median decrease in normal-image anomaly ranking",
        )
    )
    if not abnormal.empty:
        small_lesion_limit = abnormal["lesion_fraction"].quantile(0.25)
        small_lesion = abnormal[
            (abnormal["lesion_fraction"] <= small_lesion_limit)
            & (abnormal["v1_minus_aa_image_percentile"] > 0)
            & (abnormal["v1_minus_aa_lesion_contrast"] >= -args.contrast_tolerance)
        ]
        definitions.append(
            (
                "small_lesion_retention",
                small_lesion,
                "v1_minus_aa_image_percentile",
                "lower-quartile lesion area with ranking gain and retained contrast",
            )
        )
    definitions.append(
        (
            "representative_failure",
            records[records["correct_direction_rank_gain"] < 0],
            "correct_direction_rank_gain",
            "median case whose image ranking moves in the wrong direction",
        )
    )

    for case_type, candidates, score_column, rule in definitions:
        chosen = representative_rows(
            candidates, score_column, args.cases_per_type, used_ids
        )
        if chosen.empty:
            logger.warning("no candidate found for %s", case_type)
            continue
        chosen = chosen.copy()
        chosen["case_type"] = case_type
        chosen["selection_rule"] = rule
        selected.append(chosen)
        used_ids.update(chosen["record_id"])

    if not selected:
        raise RuntimeError("no representative case satisfies the selection rules")
    return pd.concat(selected, ignore_index=True)


def normalize_map(score_map, bounds):
    minimum, maximum = bounds
    return np.clip((score_map - minimum) / max(maximum - minimum, 1e-12), 0, 1)


def selected_pixel_metrics(mask, score_map, threshold):
    mask = mask.astype(bool)
    if not mask.any():
        return {
            "pixel_auc": np.nan,
            "pixel_ap": np.nan,
            "dice_at_threshold": np.nan,
            "iou_at_threshold": np.nan,
            "false_positive_fraction_at_threshold": float(
                (score_map >= threshold).mean()
            ),
        }
    if mask.all():
        return {
            "pixel_auc": np.nan,
            "pixel_ap": np.nan,
            "dice_at_threshold": np.nan,
            "iou_at_threshold": np.nan,
            "false_positive_fraction_at_threshold": np.nan,
        }
    flat_mask = mask.reshape(-1).astype(np.uint8)
    flat_score = score_map.reshape(-1)
    pixel_auc = roc_auc_score(flat_mask, flat_score)
    pixel_ap = average_precision_score(flat_mask, flat_score)
    prediction = score_map >= threshold
    intersection = np.logical_and(prediction, mask).sum()
    union = np.logical_or(prediction, mask).sum()
    dice = (2.0 * intersection) / max(prediction.sum() + mask.sum(), 1)
    iou = intersection / max(union, 1)
    return {
        "pixel_auc": float(pixel_auc),
        "pixel_ap": float(pixel_ap),
        "dice_at_threshold": float(dice),
        "iou_at_threshold": float(iou),
        "false_positive_fraction_at_threshold": np.nan,
    }


def tensor_to_rgb(image):
    mean = image.new_tensor(CLIP_MEAN).view(3, 1, 1)
    std = image.new_tensor(CLIP_STD).view(3, 1, 1)
    raw = (image * std + mean).clamp(0, 1)
    return (raw.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def read_original_image(dataset, file_name, size):
    path = Path(DATA_PATH[dataset]) / file_name
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(image)


def mask_overlay(image, mask):
    overlay = image.astype(np.float32).copy()
    lesion = mask.astype(bool)
    if lesion.any():
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        overlay[lesion] = 0.55 * overlay[lesion] + 0.45 * red[lesion]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def heatmap_overlay(image, score_map, alpha=0.45):
    color = cv2.applyColorMap(
        np.round(np.clip(score_map, 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * image + alpha * color, 0, 255).astype(np.uint8)


def add_panel_title(image, title, subtitle=""):
    title_height = 58
    canvas = np.zeros(
        (image.shape[0] + title_height, image.shape[1], 3), dtype=np.uint8
    )
    canvas[title_height:] = image
    cv2.putText(
        canvas,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            canvas,
            subtitle,
            (8, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return canvas


def add_shared_colorbar(canvas):
    height = 42
    bar = np.zeros((height, canvas.shape[1], 3), dtype=np.uint8)
    gradient = np.linspace(0, 255, canvas.shape[1], dtype=np.uint8)[None, :]
    gradient = np.repeat(gradient, 14, axis=0)
    gradient = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB)
    bar[4:18] = gradient
    cv2.putText(
        bar,
        "0  AA/V1 heatmaps: shared raw-score scale",
        (4, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        "1",
        (canvas.shape[1] - 18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([canvas, bar])


def safe_file_stem(value):
    stem = Path(value).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:80]


def create_case_figure(
    row,
    original,
    evaluated,
    mask,
    aa_map,
    v1_map,
    uncertainty,
    output_path,
    panel_size,
):
    resize = lambda image: cv2.resize(  # noqa: E731
        image, (panel_size, panel_size), interpolation=cv2.INTER_AREA
    )
    original = resize(original)
    evaluated = resize(evaluated)
    mask = cv2.resize(
        mask.astype(np.uint8),
        (panel_size, panel_size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    aa_map = cv2.resize(
        aa_map, (panel_size, panel_size), interpolation=cv2.INTER_LINEAR
    )
    v1_map = cv2.resize(
        v1_map, (panel_size, panel_size), interpolation=cv2.INTER_LINEAR
    )
    uncertainty = cv2.resize(
        uncertainty,
        (panel_size, panel_size),
        interpolation=cv2.INTER_LINEAR,
    )
    panels = [
        add_panel_title(original, "Original"),
        add_panel_title(
            evaluated,
            "Evaluated input",
            f"primary noise={row['test_noise_severity']:.3f}",
        ),
        add_panel_title(mask_overlay(original, mask), "Ground truth"),
        add_panel_title(
            heatmap_overlay(evaluated, aa_map),
            "AA-CLIP heatmap",
            f"score={row['aa_image_score']:.3f}, rank={row['aa_image_percentile']:.3f}",
        ),
        add_panel_title(
            heatmap_overlay(evaluated, v1_map),
            "V1 heatmap",
            f"score={row['v1_image_score']:.3f}, rank={row['v1_image_percentile']:.3f}",
        ),
        add_panel_title(
            heatmap_overlay(evaluated, uncertainty),
            "V1 uncertainty",
            f"mean={uncertainty.mean():.3f}",
        ),
    ]
    canvas = add_shared_colorbar(np.hstack(panels))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def render_selected_cases(
    selected,
    image_datasets,
    aa_model,
    v1_model,
    aa_text_embeddings,
    v1_text_embeddings,
    bounds,
    device,
    args,
):
    metric_rows = []
    cases_dir = Path(args.output_dir) / "cases"
    for case_number, row in selected.iterrows():
        item = image_datasets[row["class_name"]][int(row["dataset_index"])]
        image = item["image"].unsqueeze(0).to(device)
        file_names = [item["file_name"]]
        outputs = predict_pair(
            aa_model,
            v1_model,
            image,
            file_names,
            aa_text_embeddings[row["class_name"]],
            v1_text_embeddings[row["class_name"]],
            args,
        )
        aa_raw_map = outputs["aa_map"][0].cpu().numpy()
        v1_raw_map = outputs["v1_map"][0].cpu().numpy()
        aa_metric_map = normalize_map(aa_raw_map, bounds["aa"])
        v1_metric_map = normalize_map(v1_raw_map, bounds["v1"])
        shared_bounds = (
            min(bounds["aa"][0], bounds["v1"][0]),
            max(bounds["aa"][1], bounds["v1"][1]),
        )
        aa_display_map = normalize_map(aa_raw_map, shared_bounds)
        v1_display_map = normalize_map(v1_raw_map, shared_bounds)
        uncertainty = np.clip(outputs["v1_uncertainty"][0].cpu().numpy(), 0, 1)
        mask = item["mask"][0].cpu().numpy() != 0
        aa_metrics = selected_pixel_metrics(mask, aa_metric_map, args.pixel_threshold)
        v1_metrics = selected_pixel_metrics(mask, v1_metric_map, args.pixel_threshold)
        metric_row = {"record_id": row["record_id"]}
        metric_row.update({f"aa_{key}": value for key, value in aa_metrics.items()})
        metric_row.update({f"v1_{key}": value for key, value in v1_metrics.items()})
        metric_rows.append(metric_row)

        original = read_original_image(args.dataset, item["file_name"], args.img_size)
        evaluated = tensor_to_rgb(outputs["evaluated_image"][0])
        figure_row = row.copy()
        figure_row["test_noise_severity"] = args.test_noise_severity
        figure_name = (
            f"{case_number + 1:02d}_{row['case_type']}_"
            f"{safe_file_stem(item['file_name'])}.png"
        )
        create_case_figure(
            figure_row,
            original,
            evaluated,
            mask,
            aa_display_map,
            v1_display_map,
            uncertainty,
            cases_dir / figure_name,
            args.panel_size,
        )
    return selected.merge(pd.DataFrame(metric_rows), on="record_id", how="left")


def dataset_image_summary(records, aa_epoch, v1_epoch, args):
    labels = records["label"].to_numpy()
    if np.unique(labels).size < 2:
        aa_auc = aa_ap = v1_auc = v1_ap = np.nan
    else:
        aa_auc = roc_auc_score(labels, records["aa_image_score"])
        aa_ap = average_precision_score(labels, records["aa_image_score"])
        v1_auc = roc_auc_score(labels, records["v1_image_score"])
        v1_ap = average_precision_score(labels, records["v1_image_score"])
    return pd.DataFrame(
        [
            {
                "method": "AA-CLIP",
                "epoch": aa_epoch,
                "image AUC": aa_auc * 100,
                "image AP": aa_ap * 100,
                "dataset": args.dataset,
                "test noise severity": args.test_noise_severity,
                "CLIP global feature weight": 0.0,
                "medical global score weight": 0.0,
            },
            {
                "method": "V1 + CLS Score",
                "epoch": v1_epoch,
                "image AUC": v1_auc * 100,
                "image AP": v1_ap * 100,
                "dataset": args.dataset,
                "test noise severity": args.test_noise_severity,
                "CLIP global feature weight": args.clip_global_weight,
                "medical global score weight": args.medical_image_score_global_weight,
            },
        ]
    )


def write_analysis_info(
    args,
    aa_image_path,
    v1_image_path,
    aa_text_path,
    v1_text_path,
    bounds,
):
    lines = [
        "AA-CLIP vs V1 paired medical case analysis",
        f"dataset: {args.dataset}",
        f"AA image checkpoint: {aa_image_path}",
        f"V1 image checkpoint: {v1_image_path}",
        f"AA text checkpoint: {aa_text_path or 'none (original CLIP text)'}",
        f"V1 text checkpoint: {v1_text_path or 'none (original CLIP text)'}",
        f"V1 auxiliary noise severity: {args.noise_severity}",
        f"primary test noise severity: {args.test_noise_severity}",
        f"V1 CLIP global feature weight: {args.clip_global_weight}",
        f"V1 medical global score weight: {args.medical_image_score_global_weight}",
        f"seed: {args.seed}",
        f"AA heatmap global bounds: {bounds['aa']}",
        f"V1 heatmap global bounds: {bounds['v1']}",
        "Figure AA/V1 heatmaps use their combined raw-score bounds for a shared visual scale.",
        "",
        "AA image scores use dataset-normalized pixel-map maxima; V1 scores additionally fuse the normalized global branch.",
        "Cases are chosen by median percentile-rank change, not by maximum improvement.",
        "V1 auxiliary noise is deterministic per file, so results do not depend on batch size.",
        "Pixel Dice/IoU use the configured threshold after each method's dataset-level min-max normalization.",
        "These cases explain the aggregate result; they do not replace dataset-level statistics.",
        "Checkpoints must be selected without using the target test set.",
    ]
    output_path = Path(args.output_dir) / "analysis_info.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired AA-CLIP/V1 medical case analysis"
    )
    parser.add_argument("--dataset", type=str, default="Liver")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--aa_save_path", type=str, default="ckpt/baseline")
    parser.add_argument(
        "--aa_image_checkpoint", type=str, default="image_adapter_1.pth"
    )
    parser.add_argument("--aa_text_checkpoint", type=str, default="text_adapter.pth")
    parser.add_argument("--v1_save_path", type=str, default="ckpt/noise_graph_cls")
    parser.add_argument(
        "--v1_image_checkpoint", type=str, default="image_adapter_1.pth"
    )
    parser.add_argument("--v1_text_checkpoint", type=str, default="text_adapter.pth")

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
        help="weight of the final CLIP CLS feature in V1 image fusion",
    )
    parser.add_argument(
        "--medical_image_score_global_weight",
        type=float,
        default=0.2,
        help="weight of the V1 global score in medical image scoring",
    )
    parser.add_argument(
        "--noise_severity",
        type=float,
        default=0.06,
        help="V1 auxiliary-view noise severity",
    )
    parser.add_argument(
        "--test_noise_severity",
        type=float,
        default=0.0,
        help="optional noise applied equally to both primary inputs",
    )
    parser.add_argument("--cases_per_type", type=int, default=1)
    parser.add_argument("--contrast_tolerance", type=float, default=0.02)
    parser.add_argument("--pixel_threshold", type=float, default=0.5)
    parser.add_argument("--panel_size", type=int, default=320)
    args = parser.parse_args()

    if args.dataset not in DOMAINS:
        parser.error(f"unknown dataset: {args.dataset}")
    if DOMAINS[args.dataset] != "Medical":
        parser.error("case_analysis.py currently supports medical datasets only")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch_size must be positive and num_workers non-negative")
    if args.noise_severity < 0 or args.test_noise_severity < 0:
        parser.error("noise severities must be non-negative")
    if args.cases_per_type < 1:
        parser.error("cases_per_type must be at least 1")
    if not 0 <= args.pixel_threshold <= 1:
        parser.error("pixel_threshold must be in [0, 1]")
    if not 0 <= args.clip_global_weight <= 1:
        parser.error("clip_global_weight must be in [0, 1]")
    if not 0 <= args.medical_image_score_global_weight <= 1:
        parser.error("medical_image_score_global_weight must be in [0, 1]")
    if args.panel_size < 128:
        parser.error("panel_size must be at least 128")
    if args.output_dir is None:
        args.output_dir = str(Path("case_results") / args.dataset)
    return args


def main():
    args = parse_args()
    setup_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=output_dir / "case_analysis.log",
        encoding="utf-8",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info("args: %s", vars(args))

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    aa_image_path = resolve_checkpoint(
        args.aa_save_path, args.aa_image_checkpoint, "AA-CLIP image checkpoint"
    )
    v1_image_path = resolve_checkpoint(
        args.v1_save_path, args.v1_image_checkpoint, "V1 image checkpoint"
    )

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    aa_model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
        enable_patch_graph=False,
        clip_global_weight=0.0,
    ).to(device)
    v1_model = AdaptedCLIP(
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
    aa_epoch = load_image_checkpoint(aa_model, aa_image_path, device, False)
    v1_epoch = load_image_checkpoint(v1_model, v1_image_path, device, True)
    aa_adapt_text, aa_text_path = load_text_checkpoint(
        aa_model,
        args.aa_save_path,
        args.aa_text_checkpoint,
        device,
        "AA-CLIP text checkpoint",
    )
    v1_adapt_text, v1_text_path = load_text_checkpoint(
        v1_model,
        args.v1_save_path,
        args.v1_text_checkpoint,
        device,
        "V1 text checkpoint",
    )
    aa_model.eval()
    v1_model.eval()

    image_datasets = get_dataset(
        args.dataset,
        args.img_size,
        None,
        -1,
        "test",
        logger=logger,
    )
    with torch.inference_mode():
        aa_text_embeddings = get_adapted_text_embedding(
            aa_model, args.dataset, device, adapt_text=aa_adapt_text
        )
        v1_text_embeddings = get_adapted_text_embedding(
            v1_model, args.dataset, device, adapt_text=v1_adapt_text
        )

    records, bounds = collect_case_records(
        aa_model,
        v1_model,
        image_datasets,
        aa_text_embeddings,
        v1_text_embeddings,
        device,
        args,
    )
    records = finalize_case_records(records, bounds, args)
    records["test_noise_severity"] = args.test_noise_severity
    records["v1_probe_noise_severity"] = args.noise_severity
    records.to_csv(output_dir / "case_predictions.csv", index=False)

    summary = dataset_image_summary(records, aa_epoch, v1_epoch, args)
    summary.to_csv(output_dir / "image_summary.csv", index=False)
    logger.info("image summary:\n%s", summary.to_string(index=False))

    selected = select_representative_cases(records, args, logger)
    selected = render_selected_cases(
        selected,
        image_datasets,
        aa_model,
        v1_model,
        aa_text_embeddings,
        v1_text_embeddings,
        bounds,
        device,
        args,
    )
    selected.to_csv(output_dir / "selected_cases.csv", index=False)
    write_analysis_info(
        args,
        aa_image_path,
        v1_image_path,
        aa_text_path,
        v1_text_path,
        bounds,
    )
    print(summary.to_string(index=False))
    print(f"case analysis saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
