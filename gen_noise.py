"""Generate visual examples with the exact auxiliary noise used by V1.

The script reads the test view produced by the repository dataset loader,
calls ``utils.make_medical_noise_view`` directly, and saves clean, noisy, and
difference images. Consequently, resizing, CLIP normalization, modality
routing, and noise equations stay consistent with the current V1 pipeline.

Windows example::

    python generate_noisy_images.py ^
      --dataset Liver ^
      --severity 0.06 ^
      --max_images 20 ^
      --output_dir ./noise_examples/Liver
"""

import argparse
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import DOMAINS, get_dataset
from utils import (
    CLIP_MEAN,
    CLIP_STD,
    make_medical_noise_view,
    noise_model_for_dataset,
    setup_seed,
)


def severity_tag(severity):
    return f"{severity:g}".replace(".", "p")


def stable_file_seed(seed, file_name):
    value = f"{seed}:v1-noise:{file_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**31)


def deterministic_v1_noise(image, dataset_name, severity, seed, file_name):
    """Use V1 noise equations with a reproducible realization per file."""
    cuda_devices = []
    if image.is_cuda:
        cuda_devices = [
            image.device.index
            if image.device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(stable_file_seed(seed, file_name))
        return make_medical_noise_view(
            image,
            dataset_name,
            severity=severity,
        )


def clip_tensor_to_rgb(image):
    """Convert one CLIP-normalized tensor to an RGB uint8 array."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("expected one image with shape [3, H, W]")
    mean = image.new_tensor(CLIP_MEAN).view(3, 1, 1)
    std = image.new_tensor(CLIP_STD).view(3, 1, 1)
    raw = (image * std + mean).clamp(0, 1)
    return (raw.permute(1, 2, 0).detach().cpu().numpy() * 255).round().astype(np.uint8)


def safe_output_name(file_name):
    path = Path(file_name)
    readable = "__".join(path.with_suffix("").parts)
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", readable).strip("_")
    readable = readable[-120:] or "image"
    digest = hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}.png"


def add_title(image, title, subtitle=""):
    image = Image.fromarray(image)
    header_height = 52
    canvas = Image.new("RGB", (image.width, image.height + header_height), "black")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill="white")
    if subtitle:
        draw.text((8, 27), subtitle, fill=(205, 205, 205))
    return canvas


def make_comparison(clean, noisy, difference, dataset_name, severity, difference_gain):
    panels = [
        add_title(clean, "Clean model input"),
        add_title(
            noisy,
            "V1 auxiliary noise view",
            f"{noise_model_for_dataset(dataset_name)}, severity={severity:g}",
        ),
        add_title(
            difference,
            "Absolute difference",
            f"display gain={difference_gain:g}x",
        ),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width
    return canvas


def label_matches(label, requested_label):
    if requested_label == "all":
        return True
    if requested_label == "normal":
        return int(label) == 0
    return int(label) == 1


def select_device(device_name):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def write_manifest(output_dir, rows):
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "dataset",
        "class_name",
        "label",
        "source_file",
        "noise_model",
        "severity",
        "seed",
        "per_file_seed",
        "clean_file",
        "noisy_file",
        "difference_file",
        "comparison_file",
        "mean_absolute_difference",
        "max_absolute_difference",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with the current V1 modality-aware noise"
    )
    parser.add_argument("--dataset", type=str, default="Liver")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--severity", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--max_images",
        type=int,
        default=20,
        help="maximum number to save; use 0 to process all matching images",
    )
    parser.add_argument(
        "--label",
        choices=["all", "normal", "anomaly"],
        default="all",
        help="optionally restrict examples by image label",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--difference_gain", type=float, default=4.0)
    parser.add_argument(
        "--no_comparison",
        action="store_true",
        help="do not save the clean/noisy/difference horizontal panel",
    )
    args = parser.parse_args()

    if args.dataset not in DOMAINS:
        parser.error(f"unknown dataset: {args.dataset}")
    if args.img_size < 1:
        parser.error("img_size must be positive")
    if args.severity < 0:
        parser.error("severity must be non-negative")
    if args.max_images < 0:
        parser.error("max_images must be non-negative")
    if args.difference_gain <= 0:
        parser.error("difference_gain must be positive")
    if args.output_dir is None:
        args.output_dir = str(
            Path("noise_examples")
            / f"{args.dataset}_severity_{severity_tag(args.severity)}"
        )
    return args


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    clean_dir = output_dir / "clean"
    noisy_dir = output_dir / "noisy"
    difference_dir = output_dir / "difference"
    comparison_dir = output_dir / "comparison"
    for directory in (clean_dir, noisy_dir, difference_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if not args.no_comparison:
        comparison_dir.mkdir(parents=True, exist_ok=True)

    image_datasets = get_dataset(
        args.dataset,
        args.img_size,
        None,
        -1,
        "test",
        logger=None,
    )
    manifest_rows = []
    saved_count = 0
    stop = False
    for class_name, image_dataset in image_datasets.items():
        for index in range(len(image_dataset)):
            item = image_dataset[index]
            if not label_matches(item["label"], args.label):
                continue
            file_name = item["file_name"]
            image = item["image"].unsqueeze(0).to(device)
            noisy_image = deterministic_v1_noise(
                image,
                args.dataset,
                args.severity,
                args.seed,
                file_name,
            )
            clean_rgb = clip_tensor_to_rgb(image[0])
            noisy_rgb = clip_tensor_to_rgb(noisy_image[0])
            absolute_difference = np.abs(
                clean_rgb.astype(np.float32) / 255.0
                - noisy_rgb.astype(np.float32) / 255.0
            )
            difference_rgb = (
                (np.clip(absolute_difference * args.difference_gain, 0, 1) * 255)
                .round()
                .astype(np.uint8)
            )

            output_name = safe_output_name(file_name)
            clean_path = clean_dir / output_name
            noisy_path = noisy_dir / output_name
            difference_path = difference_dir / output_name
            Image.fromarray(clean_rgb).save(clean_path)
            Image.fromarray(noisy_rgb).save(noisy_path)
            Image.fromarray(difference_rgb).save(difference_path)

            comparison_path = ""
            if not args.no_comparison:
                comparison_file = comparison_dir / output_name
                comparison = make_comparison(
                    clean_rgb,
                    noisy_rgb,
                    difference_rgb,
                    args.dataset,
                    args.severity,
                    args.difference_gain,
                )
                comparison.save(comparison_file)
                comparison_path = str(comparison_file)

            manifest_rows.append(
                {
                    "dataset": args.dataset,
                    "class_name": class_name,
                    "label": int(item["label"]),
                    "source_file": file_name,
                    "noise_model": noise_model_for_dataset(args.dataset),
                    "severity": args.severity,
                    "seed": args.seed,
                    "per_file_seed": stable_file_seed(args.seed, file_name),
                    "clean_file": str(clean_path),
                    "noisy_file": str(noisy_path),
                    "difference_file": str(difference_path),
                    "comparison_file": comparison_path,
                    "mean_absolute_difference": float(absolute_difference.mean()),
                    "max_absolute_difference": float(absolute_difference.max()),
                }
            )
            saved_count += 1
            print(f"[{saved_count}] {file_name} -> {noisy_path}")
            if args.max_images and saved_count >= args.max_images:
                stop = True
                break
        if stop:
            break

    if not manifest_rows:
        raise RuntimeError(
            f"no images matched dataset={args.dataset!r}, label={args.label!r}"
        )
    manifest_path = write_manifest(output_dir, manifest_rows)
    print(f"saved {saved_count} V1 noise example(s) to {output_dir.resolve()}")
    print(f"manifest saved to {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
