"""Evaluate every selected V3.1 lesion-preserving graph checkpoint."""

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
from model.frozen_sfgraph import FrozenSFGraphModel
from prompt_utils import (
    DEFAULT_LLM_PROMPT_PATH,
    PROMPT_SOURCES,
    validate_checkpoint_prompt_metadata,
)
from utils import setup_seed
from v3_utils import (
    checkpoint_sort_key,
    deterministic_test_noise,
    frozen_text_embedding_dict,
    smooth_max_pool_logits,
    v3_metrics,
    validate_v3_checkpoint,
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
    "test noise severity",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Test the frozen-encoder V3.1 lesion-preserving graph head"
    )
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--dataset", type=str, default="Liver")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/v3_lesion_sfgraph")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="v3_head_epoch_*.pth",
        help="checkpoint file name or glob pattern relative to save_path",
    )
    parser.add_argument("--results_file", type=str, default="v3_results.csv")
    parser.add_argument("--test_noise_severity", type=float, default=0.0)

    parser.add_argument("--feature_layer", type=int, default=18)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--text_temperature", type=float, default=10.0)
    parser.add_argument("--low_frequency_temperature", type=float, default=0.2)
    parser.add_argument("--high_frequency_temperature", type=float, default=1.0)
    parser.add_argument("--semantic_graph_temperature", type=float, default=0.1)
    parser.add_argument("--max_correction", type=float, default=4.0)
    parser.add_argument("--image_pool_temperature", type=float, default=10.0)
    parser.add_argument(
        "--prompt_source",
        choices=PROMPT_SOURCES,
        default="llm",
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
    if args.feature_layer < 1 or args.hidden_dim < 1:
        parser.error("feature_layer and hidden_dim must be positive")
    for name in (
        "text_temperature",
        "low_frequency_temperature",
        "high_frequency_temperature",
        "semantic_graph_temperature",
        "max_correction",
        "image_pool_temperature",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.test_noise_severity < 0:
        parser.error("test_noise_severity must be non-negative")


def configure_logger(save_path):
    logger = logging.getLogger("test_v3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(
        Path(save_path) / "test_v3.log",
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
    dataloader,
    device,
    args,
):
    masks = []
    labels = []
    score_maps = []
    image_scores = []
    for input_data in tqdm(dataloader, desc="V3 inference", leave=False):
        image = input_data["image"].to(device, non_blocking=True)
        file_names = list(input_data["file_name"])
        image = deterministic_test_noise(
            image,
            file_names,
            args.dataset,
            args.test_noise_severity,
            args.seed,
        )
        patch_logits = model(image, text_embeddings)
        pixel_logits = F.interpolate(
            patch_logits,
            size=input_data["mask"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        score_maps.append(torch.sigmoid(pixel_logits).cpu().numpy())
        image_logits = smooth_max_pool_logits(
            patch_logits,
            temperature=args.image_pool_temperature,
        )
        image_scores.append(torch.sigmoid(image_logits).cpu().numpy())
        masks.append(input_data["mask"].cpu().numpy())
        labels.append(input_data["label"].cpu().numpy().reshape(-1))
    return (
        np.concatenate(masks, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(score_maps, axis=0),
        np.concatenate(image_scores, axis=0),
    )


def average_result(class_results):
    row = {"class name": "Average"}
    for name in ("pixel AUC", "pixel AP", "image AUC", "image AP"):
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
            f"no V3 checkpoint matches {save_path / args.checkpoint}"
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
    model = FrozenSFGraphModel(
        clip_model=clip_model,
        feature_layer=args.feature_layer,
        hidden_dim=args.hidden_dim,
        text_temperature=args.text_temperature,
        low_frequency_temperature=args.low_frequency_temperature,
        high_frequency_temperature=args.high_frequency_temperature,
        semantic_graph_temperature=args.semantic_graph_temperature,
        max_correction=args.max_correction,
    ).to(device)
    model.eval()
    expected_architecture = model.architecture_config()
    expected_architecture["image_pool_temperature"] = float(args.image_pool_temperature)
    text_embedding_dict = frozen_text_embedding_dict(
        model.clip_model,
        args.dataset,
        device,
        args.prompt_source,
        args.llm_prompt_path,
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
        validate_v3_checkpoint(checkpoint, expected_architecture)
        validate_checkpoint_prompt_metadata(
            checkpoint,
            args.prompt_source,
            args.dataset,
            args.llm_prompt_path,
            f"V3 checkpoint {checkpoint_path}",
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
        model.load_head_state_dict(checkpoint["head"], strict=True)
        model.eval()
        epoch = int(checkpoint["epoch"])
        class_results = []
        for class_name, image_dataset in image_datasets.items():
            dataloader = DataLoader(image_dataset, **dataloader_kwargs)
            masks, labels, score_maps, image_scores = predict_class(
                model,
                text_embedding_dict[class_name],
                dataloader,
                device,
                args,
            )
            result = v3_metrics(
                masks,
                labels,
                score_maps,
                image_scores,
                class_name,
            )
            class_results.append(result)
        class_results.append(average_result(class_results))

        logger.info("checkpoint %s (epoch %d)", checkpoint_path, epoch)
        for result in class_results:
            logger.info(
                "%s | pixel AUC %.4f | pixel AP %.4f | image AUC %.4f | image AP %.4f",
                result["class name"],
                result["pixel AUC"],
                result["pixel AP"],
                result["image AUC"],
                result["image AP"],
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
    print(f"V3 results saved to {results_path.resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
