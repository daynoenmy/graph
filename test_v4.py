"""Evaluate V4.1 bounded graph-spectral residual checkpoints."""

import argparse
import csv
import logging
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset
from model.clip import create_model
from model.modality_graph_spectral import FrozenModalityGraphSpectralModel
from prompt_utils import (
    DEFAULT_LLM_PROMPT_PATH,
    PROMPT_SOURCES,
    validate_checkpoint_prompt_metadata,
)
from utils import setup_seed
from v4_utils import (
    checkpoint_sort_key,
    deterministic_test_noise,
    frozen_modality_embedding,
    frozen_text_embedding_dict,
    medical_metrics,
    validate_v4_checkpoint,
)


RESULT_FIELDS = (
    "checkpoint",
    "epoch",
    "dataset",
    "class name",
    "pixel AUC",
    "pixel AP",
    "image AUC",
    "image AP",
    "masked anomaly coverage",
    "test noise severity",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Test V4.1 bounded modality-conditioned graph spectra"
    )
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--dataset", type=str, default="Liver")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/v4_1_graph_spectral")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="v4_1_head_epoch_*.pth",
        help="checkpoint file name or glob pattern relative to save_path",
    )
    parser.add_argument("--results_file", type=str, default="v4_1_results.csv")
    parser.add_argument("--test_noise_severity", type=float, default=0.0)

    parser.add_argument(
        "--feature_layers",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
    )
    parser.add_argument("--text_temperature", type=float, default=10.0)
    parser.add_argument("--laplacian_temperature", type=float, default=0.2)
    parser.add_argument("--spectral_uniform_mass", type=float, default=0.2)
    parser.add_argument("--max_spectral_coefficient", type=float, default=1.0)
    parser.add_argument("--readout_temperature", type=float, default=1.0)
    parser.add_argument(
        "--prompt_source",
        choices=PROMPT_SOURCES,
        default="template",
    )
    parser.add_argument(
        "--llm_prompt_path",
        type=str,
        default=str(DEFAULT_LLM_PROMPT_PATH),
    )
    return parser


def validate_args(parser, args):
    if args.img_size < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error(
            "img_size/batch_size must be positive and num_workers non-negative"
        )
    if (
        not args.feature_layers
        or min(args.feature_layers) < 1
        or args.feature_layers != sorted(set(args.feature_layers))
    ):
        parser.error("feature_layers must be unique, positive, and strictly increasing")
    for name in (
        "text_temperature",
        "laplacian_temperature",
        "max_spectral_coefficient",
        "readout_temperature",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not 0 <= args.spectral_uniform_mass < 1:
        parser.error("spectral_uniform_mass must lie in [0, 1)")
    if args.test_noise_severity < 0:
        parser.error("test_noise_severity must be non-negative")


def configure_logger(save_path):
    logger = logging.getLogger("test_v4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(
        Path(save_path) / "test_v4.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


@torch.inference_mode()
def predict_class(
    model,
    text_embeddings,
    modality_embedding,
    dataloader,
    device,
    args,
):
    masks = []
    labels = []
    score_maps = []
    image_scores = []
    mask_validity = []
    anomaly_mask_availability = []
    for input_data in tqdm(dataloader, desc="V4.1 inference", leave=False):
        image = input_data["image"].to(device, non_blocking=True)
        image = deterministic_test_noise(
            image,
            list(input_data["file_name"]),
            args.dataset,
            args.test_noise_severity,
            args.seed,
        )
        patch_logits, image_logits = model(
            image,
            text_embeddings,
            modality_embedding,
            return_image_logits=True,
        )
        pixel_logits = F.interpolate(
            patch_logits,
            size=input_data["mask"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        score_maps.append(torch.sigmoid(pixel_logits).cpu().numpy())
        image_scores.append(torch.sigmoid(image_logits).cpu().numpy())
        masks.append(input_data["mask"].cpu().numpy())
        labels.append(input_data["label"].cpu().numpy().reshape(-1))
        mask_validity.append(input_data["mask_valid"].cpu().numpy().reshape(-1))
        anomaly_mask_availability.append(
            input_data["has_anomaly_mask"].cpu().numpy().reshape(-1)
        )
    return (
        np.concatenate(masks, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(score_maps, axis=0),
        np.concatenate(image_scores, axis=0),
        np.concatenate(mask_validity, axis=0),
        np.concatenate(anomaly_mask_availability, axis=0),
    )


def average_result(class_results):
    row = {"class name": "Average"}
    for name in (
        "pixel AUC",
        "pixel AP",
        "image AUC",
        "image AP",
        "masked anomaly coverage",
    ):
        values = np.asarray([result[name] for result in class_results], dtype=float)
        row[name] = float(np.nanmean(values)) if not np.isnan(values).all() else np.nan
    return row


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    setup_seed(args.seed)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(save_path)
    logger.info("args: %s", vars(args))
    checkpoint_paths = sorted(
        [Path(path) for path in glob(str(save_path / args.checkpoint))],
        key=checkpoint_sort_key,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"no V4.1 checkpoint matches {save_path / args.checkpoint}"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = FrozenModalityGraphSpectralModel(
        clip_model=clip_model,
        feature_layers=args.feature_layers,
        text_temperature=args.text_temperature,
        laplacian_temperature=args.laplacian_temperature,
        spectral_uniform_mass=args.spectral_uniform_mass,
        max_spectral_coefficient=args.max_spectral_coefficient,
        readout_temperature=args.readout_temperature,
    ).to(device)
    model.eval()
    expected_architecture = model.architecture_config()
    text_embedding_dict = frozen_text_embedding_dict(
        model.clip_model,
        args.dataset,
        device,
        args.prompt_source,
        args.llm_prompt_path,
    )
    modality_embedding = frozen_modality_embedding(
        model.clip_model,
        args.dataset,
        device,
    )
    image_datasets = get_dataset(
        args.dataset,
        args.img_size,
        None,
        -1,
        "test",
        logger=logger,
    )
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers if use_cuda else 0,
        "pin_memory": use_cuda,
    }

    all_rows = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        validate_v4_checkpoint(checkpoint, expected_architecture)
        validate_checkpoint_prompt_metadata(
            checkpoint,
            args.prompt_source,
            args.dataset,
            args.llm_prompt_path,
            f"V4.1 checkpoint {checkpoint_path}",
        )
        if checkpoint.get("model_name") != args.model_name:
            raise ValueError(
                f"model_name mismatch for {checkpoint_path}: "
                f"checkpoint={checkpoint.get('model_name')}, argument={args.model_name}"
            )
        if int(checkpoint.get("img_size", -1)) != args.img_size:
            raise ValueError(
                f"img_size mismatch for {checkpoint_path}: "
                f"checkpoint={checkpoint.get('img_size')}, argument={args.img_size}"
            )
        checkpoint_lodo_target = checkpoint.get("lodo_target")
        if checkpoint_lodo_target is not None and len(checkpoint_paths) != 1:
            raise ValueError(
                "LODO evaluation requires exactly one fixed checkpoint; "
                "do not select an epoch using the held-out target dataset"
            )
        if (
            checkpoint_lodo_target is not None
            and checkpoint_lodo_target != args.dataset
        ):
            raise ValueError(
                f"LODO target mismatch for {checkpoint_path}: "
                f"checkpoint={checkpoint_lodo_target}, test={args.dataset}"
            )

        model.load_head_state_dict(checkpoint["head"], strict=True)
        model.eval()
        epoch = int(checkpoint["epoch"])
        class_results = []
        for class_name, image_dataset in image_datasets.items():
            anchors = text_embedding_dict[class_name]
            conditioning = model.conditioning_state(modality_embedding)
            logger.info(
                "checkpoint %s conditioning | %s | layers %s | weights %s | "
                "L/L2 coefficients %s",
                checkpoint_path,
                class_name,
                args.feature_layers,
                [round(value, 4) for value in conditioning["layer_weights"]],
                [
                    [round(value, 4) for value in row]
                    for row in conditioning["residual_coefficients"]
                ],
            )
            dataloader = DataLoader(image_dataset, **dataloader_kwargs)
            (
                masks,
                labels,
                score_maps,
                image_scores,
                mask_valid,
                has_anomaly_mask,
            ) = predict_class(
                model,
                anchors,
                modality_embedding,
                dataloader,
                device,
                args,
            )
            class_results.append(
                medical_metrics(
                    masks,
                    labels,
                    score_maps,
                    image_scores,
                    class_name,
                    mask_valid=mask_valid,
                    has_anomaly_mask=has_anomaly_mask,
                )
            )
        class_results.append(average_result(class_results))

        logger.info("checkpoint %s (epoch %d)", checkpoint_path, epoch)
        for result in class_results:
            logger.info(
                "%s | pixel AUC %.4f | pixel AP %.4f | image AUC %.4f | "
                "image AP %.4f | masked anomaly coverage %.2f%%",
                result["class name"],
                result["pixel AUC"],
                result["pixel AP"],
                result["image AUC"],
                result["image AP"],
                result["masked anomaly coverage"],
            )
            all_rows.append(
                {
                    "checkpoint": checkpoint_path.name,
                    "epoch": epoch,
                    "dataset": args.dataset,
                    **result,
                    "test noise severity": args.test_noise_severity,
                }
            )

    results_path = Path(args.results_file)
    if not results_path.is_absolute():
        results_path = save_path / results_path
    write_results(results_path, all_rows)
    print(f"V4.1 results saved to {results_path.resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
