"""Train V3.1: a frozen CLIP encoder with a lesion-preserving SF graph."""

import argparse
import logging
import os
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
    prompt_checkpoint_metadata,
    validate_checkpoint_prompt_metadata,
)
from utils import setup_seed
from v3_utils import (
    binary_focal_dice_loss,
    frozen_text_embedding_dict,
    lesion_band_preservation_loss,
    normal_band_consistency_loss,
    smooth_max_pool_logits,
    validate_v3_checkpoint,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the frozen-encoder V3.1 lesion-preserving graph head"
    )
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--dataset", type=str, default="Brain")
    parser.add_argument(
        "--training_mode",
        choices=["few_shot", "full_shot"],
        default="full_shot",
    )
    parser.add_argument("--shot", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/v3_lesion_sfgraph")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="none",
        help="checkpoint file name relative to save_path, or none",
    )

    parser.add_argument("--feature_layer", type=int, default=18)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--text_temperature", type=float, default=10.0)
    parser.add_argument("--low_frequency_temperature", type=float, default=0.2)
    parser.add_argument("--high_frequency_temperature", type=float, default=1.0)
    parser.add_argument("--semantic_graph_temperature", type=float, default=0.1)
    parser.add_argument("--max_correction", type=float, default=4.0)
    parser.add_argument("--image_pool_temperature", type=float, default=10.0)
    parser.add_argument("--image_loss_weight", type=float, default=1.0)
    parser.add_argument("--band_consistency_weight", type=float, default=0.05)
    parser.add_argument("--lesion_preservation_weight", type=float, default=0.05)
    parser.add_argument("--band_scale_min", type=float, default=0.5)
    parser.add_argument("--band_scale_max", type=float, default=1.5)

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
    if args.epochs < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("epochs and learning_rate must be positive")
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
    if (
        min(
            args.image_loss_weight,
            args.band_consistency_weight,
            args.lesion_preservation_weight,
        )
        < 0
    ):
        parser.error("loss weights must be non-negative")
    if not 0 <= args.band_scale_min < 1 < args.band_scale_max <= 2:
        parser.error("band scale range must satisfy 0 <= min < 1 < max <= 2")
    if args.training_mode == "few_shot" and args.shot < 1:
        parser.error("shot must be positive in few_shot mode")


def configure_logger(save_path):
    logger = logging.getLogger("train_v3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(
        Path(save_path) / "train_v3.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_resume_checkpoint(save_path, checkpoint_name):
    if checkpoint_name.lower() == "none":
        return None
    path = Path(checkpoint_name)
    if not path.is_absolute():
        path = Path(save_path) / path
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def checkpoint_payload(
    model,
    optimizer,
    epoch,
    args,
    prompt_metadata,
):
    architecture = model.architecture_config()
    architecture["image_pool_temperature"] = float(args.image_pool_temperature)
    return {
        "method": "frozen_sfgraph_v3",
        "version": 2,
        "epoch": epoch,
        "encoder_frozen": True,
        "model_name": args.model_name,
        "img_size": args.img_size,
        "architecture": architecture,
        "head": model.head_state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_dataset": args.dataset,
        "training_args": vars(args),
        **prompt_metadata,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    setup_seed(args.seed)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(save_path)
    logger.info("args: %s", vars(args))

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
    if any(parameter.requires_grad for parameter in model.clip_model.parameters()):
        raise RuntimeError("V3 invariant violated: CLIP encoder is not fully frozen")

    trainable_parameters = list(model.trainable_parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "trainable parameters: %d / %d (%.6f%%)",
        trainable_count,
        total_count,
        100.0 * trainable_count / total_count,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    expected_architecture = model.architecture_config()
    expected_architecture["image_pool_temperature"] = float(args.image_pool_temperature)

    prompt_metadata = prompt_checkpoint_metadata(
        args.prompt_source,
        args.dataset,
        args.llm_prompt_path,
    )
    text_embeddings = frozen_text_embedding_dict(
        model.clip_model,
        args.dataset,
        device,
        args.prompt_source,
        args.llm_prompt_path,
    )

    start_epoch = 0
    resume_path = resolve_resume_checkpoint(args.save_path, args.resume_checkpoint)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        validate_v3_checkpoint(checkpoint, expected_architecture)
        validate_checkpoint_prompt_metadata(
            checkpoint,
            args.prompt_source,
            args.dataset,
            args.llm_prompt_path,
            f"V3 checkpoint {resume_path}",
        )
        if checkpoint.get("model_name") != args.model_name:
            raise ValueError("resume checkpoint model_name does not match arguments")
        if int(checkpoint.get("img_size", -1)) != args.img_size:
            raise ValueError("resume checkpoint img_size does not match arguments")
        model.load_head_state_dict(checkpoint["head"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        logger.info("resumed from %s at epoch %d", resume_path, start_epoch)

    if args.training_mode == "full_shot":
        args.shot = -1
    # The first dataset view omits color jitter. This keeps medical intensity
    # statistics intact while retaining mask-aligned spatial augmentation.
    train_dataset, _ = get_dataset(
        args.dataset,
        args.img_size,
        args.training_mode,
        args.shot,
        "train",
        logger,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if use_cuda else 0,
        pin_memory=use_cuda,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        pixel_losses = []
        image_losses = []
        consistency_losses = []
        lesion_preservation_losses = []
        progress = tqdm(dataloader, desc=f"V3.1 epoch {epoch + 1}/{args.epochs}")
        for input_data in progress:
            image = input_data["image"].to(device, non_blocking=True)
            mask = input_data["mask"].to(device, non_blocking=True).float()
            label = input_data["label"].to(device, non_blocking=True).float()
            class_names = input_data["class_name"]
            batch_text = torch.stack(
                [text_embeddings[class_name] for class_name in class_names],
                dim=0,
            )

            optimizer.zero_grad(set_to_none=True)
            base_output, perturbed_logits, _ = model(
                image,
                batch_text,
                return_band_perturbation=True,
                band_scale_range=(
                    args.band_scale_min,
                    args.band_scale_max,
                ),
            )
            base_logits = base_output
            pixel_logits = F.interpolate(
                base_logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pixel_loss = binary_focal_dice_loss(pixel_logits, mask)
            image_logits = smooth_max_pool_logits(
                base_logits,
                temperature=args.image_pool_temperature,
            )
            image_loss = F.binary_cross_entropy_with_logits(image_logits, label)
            consistency_loss = normal_band_consistency_loss(
                base_logits,
                perturbed_logits,
                mask,
            )
            lesion_preservation_loss = lesion_band_preservation_loss(
                base_logits,
                perturbed_logits,
                mask,
            )
            loss = pixel_loss + args.image_loss_weight * image_loss
            loss = loss + args.band_consistency_weight * consistency_loss
            loss = loss + args.lesion_preservation_weight * lesion_preservation_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
            optimizer.step()

            losses.append(float(loss.detach()))
            pixel_losses.append(float(pixel_loss.detach()))
            image_losses.append(float(image_loss.detach()))
            consistency_losses.append(float(consistency_loss.detach()))
            lesion_preservation_losses.append(float(lesion_preservation_loss.detach()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")

        logger.info(
            "epoch %d | loss %.6f | pixel %.6f | image %.6f | "
            "normal-band %.6f | lesion-preserve %.6f",
            epoch + 1,
            np.mean(losses),
            np.mean(pixel_losses),
            np.mean(image_losses),
            np.mean(consistency_losses),
            np.mean(lesion_preservation_losses),
        )
        payload = checkpoint_payload(
            model,
            optimizer,
            epoch + 1,
            args,
            prompt_metadata,
        )
        torch.save(payload, save_path / "v3_head_latest.pth")
        torch.save(payload, save_path / f"v3_head_epoch_{epoch + 1}.pth")

    logger.info("training complete")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
