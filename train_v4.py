"""Train V4: frozen CLIP with modality-conditioned graph spectra."""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import get_dataset
from dataset.constants import BMAD_DATASETS
from model.clip import create_model
from model.modality_graph_spectral import FrozenModalityGraphSpectralModel
from prompt_utils import (
    DEFAULT_LLM_PROMPT_PATH,
    PROMPT_SOURCES,
    prompt_checkpoint_metadata,
    resolve_prompt_source,
    validate_checkpoint_prompt_metadata,
)
from utils import setup_seed
from v4_utils import (
    binary_focal_dice_loss,
    frozen_text_embedding_dict,
    validate_v4_checkpoint,
)


DEFAULT_SAVE_PATH = "ckpt/v4_graph_spectral"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train V4 modality-conditioned graph spectral fusion"
    )
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--dataset", type=str, default="Brain")
    parser.add_argument(
        "--lodo_target",
        choices=("none", *BMAD_DATASETS),
        default="none",
        help="held-out BMAD dataset; the remaining five are balanced sources",
    )
    parser.add_argument(
        "--training_mode",
        choices=("few_shot", "full_shot"),
        default="full_shot",
    )
    parser.add_argument("--shot", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="none",
        help="checkpoint file name relative to save_path, or none",
    )

    parser.add_argument(
        "--feature_layers",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
    )
    parser.add_argument("--text_temperature", type=float, default=10.0)
    parser.add_argument("--laplacian_temperature", type=float, default=0.2)
    parser.add_argument("--spectral_uniform_mass", type=float, default=0.2)
    parser.add_argument("--readout_temperature", type=float, default=1.0)
    parser.add_argument("--image_loss_weight", type=float, default=1.0)
    parser.add_argument("--pixel_loss_weight", type=float, default=1.0)

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
    if (
        not args.feature_layers
        or min(args.feature_layers) < 1
        or args.feature_layers != sorted(set(args.feature_layers))
    ):
        parser.error(
            "feature_layers must be unique, positive, and strictly increasing"
        )
    for name in (
        "text_temperature",
        "laplacian_temperature",
        "readout_temperature",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not 0 <= args.spectral_uniform_mass < 1:
        parser.error("spectral_uniform_mass must lie in [0, 1)")
    if min(args.image_loss_weight, args.pixel_loss_weight) < 0:
        parser.error("loss weights must be non-negative")
    if args.training_mode == "few_shot" and args.shot < 1:
        parser.error("shot must be positive in few_shot mode")
    if args.lodo_target != "none" and args.training_mode != "full_shot":
        parser.error("BMAD leave-one-dataset-out requires full_shot")


def configure_logger(save_path):
    logger = logging.getLogger("train_v4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(
        Path(save_path) / "train_v4.log",
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


def resolve_training_protocol(args):
    if args.lodo_target == "none":
        return [args.dataset], None
    return [name for name in BMAD_DATASETS if name != args.lodo_target], args.lodo_target


def validate_protocol_prompt_source(prompt_source, dataset_names, prompt_path):
    resolved_sources = {
        resolve_prompt_source(prompt_source, name, prompt_path)
        for name in dataset_names
    }
    if len(resolved_sources) != 1:
        raise ValueError(
            "all LODO datasets must resolve to one common prompt source; "
            f"got {sorted(resolved_sources)}"
        )
    return resolved_sources.pop()


def build_training_dataloader(dataset_names, args, logger, use_cuda):
    datasets = []
    lengths = []
    for dataset_name in dataset_names:
        dataset, _ = get_dataset(
            dataset_name,
            args.img_size,
            args.training_mode,
            args.shot,
            "train",
            logger,
        )
        datasets.append(dataset)
        lengths.append(len(dataset))

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers if use_cuda else 0,
        "pin_memory": use_cuda,
    }
    if len(datasets) == 1:
        return DataLoader(datasets[0], shuffle=True, **dataloader_kwargs), lengths

    combined = ConcatDataset(datasets)
    sample_weights = torch.cat(
        [torch.full((length,), 1.0 / length, dtype=torch.double) for length in lengths]
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(combined),
        replacement=True,
        generator=generator,
    )
    logger.info(
        "balanced LODO sources: %s",
        {name: length for name, length in zip(dataset_names, lengths)},
    )
    return DataLoader(combined, sampler=sampler, **dataloader_kwargs), lengths


def checkpoint_payload(
    model,
    optimizer,
    epoch,
    args,
    prompt_metadata,
    training_datasets,
    lodo_target,
):
    return {
        "method": "modality_graph_spectral_v4",
        "version": 4,
        "epoch": epoch,
        "encoder_frozen": True,
        "model_name": args.model_name,
        "img_size": args.img_size,
        "architecture": model.architecture_config(),
        "head": model.head_state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_dataset": args.dataset if lodo_target is None else "BMAD_LODO",
        "training_datasets": list(training_datasets),
        "lodo_target": lodo_target,
        "training_args": vars(args),
        **prompt_metadata,
    }


def log_conditioning_weights(model, text_embeddings, logger, epoch):
    for dataset_name, class_embeddings in text_embeddings.items():
        for class_name, anchors in class_embeddings.items():
            matrix = model.conditioning_weights(anchors)
            logger.info(
                "epoch %d spectral weights | %s/%s | layers %s x "
                "orders 0/1/2: %s",
                epoch,
                dataset_name,
                class_name,
                list(model.feature_layers),
                [[round(value, 4) for value in row] for row in matrix],
            )


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    setup_seed(args.seed)

    training_datasets, lodo_target = resolve_training_protocol(args)
    prompt_datasets = list(training_datasets)
    if lodo_target is not None:
        prompt_datasets.append(lodo_target)
        if args.save_path == DEFAULT_SAVE_PATH:
            args.save_path = str(Path("ckpt/v4_bmad_lodo") / lodo_target)
    resolved_prompt_source = validate_protocol_prompt_source(
        args.prompt_source,
        prompt_datasets,
        args.llm_prompt_path,
    )

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(save_path)
    logger.info("args: %s", vars(args))
    logger.info(
        "training datasets: %s | held-out target: %s",
        training_datasets,
        lodo_target,
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
        readout_temperature=args.readout_temperature,
    ).to(device)
    if any(parameter.requires_grad for parameter in model.clip_model.parameters()):
        raise RuntimeError("V4 invariant violated: CLIP encoder is not fully frozen")

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

    prompt_metadata = prompt_checkpoint_metadata(
        resolved_prompt_source,
        training_datasets[0],
        args.llm_prompt_path,
    )
    text_embeddings = {
        dataset_name: frozen_text_embedding_dict(
            model.clip_model,
            dataset_name,
            device,
            resolved_prompt_source,
            args.llm_prompt_path,
        )
        for dataset_name in training_datasets
    }

    start_epoch = 0
    resume_path = resolve_resume_checkpoint(args.save_path, args.resume_checkpoint)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        validate_v4_checkpoint(checkpoint, expected_architecture)
        validate_checkpoint_prompt_metadata(
            checkpoint,
            resolved_prompt_source,
            training_datasets[0],
            args.llm_prompt_path,
            f"V4 checkpoint {resume_path}",
        )
        if checkpoint.get("model_name") != args.model_name:
            raise ValueError("resume checkpoint model_name does not match arguments")
        if int(checkpoint.get("img_size", -1)) != args.img_size:
            raise ValueError("resume checkpoint img_size does not match arguments")
        if checkpoint.get("training_datasets") != training_datasets:
            raise ValueError(
                "resume checkpoint training datasets do not match arguments"
            )
        if checkpoint.get("lodo_target") != lodo_target:
            raise ValueError("resume checkpoint LODO target does not match arguments")
        model.load_head_state_dict(checkpoint["head"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        logger.info("resumed from %s at epoch %d", resume_path, start_epoch)

    if args.training_mode == "full_shot":
        args.shot = -1
    dataloader, dataset_lengths = build_training_dataloader(
        training_datasets,
        args,
        logger,
        use_cuda,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        pixel_losses = []
        image_losses = []
        pixel_supervised_samples = 0
        total_samples = 0
        progress = tqdm(dataloader, desc=f"V4 epoch {epoch + 1}/{args.epochs}")
        for input_data in progress:
            image = input_data["image"].to(device, non_blocking=True)
            mask = input_data["mask"].to(device, non_blocking=True).float()
            label = input_data["label"].to(device, non_blocking=True).float()
            mask_valid = input_data["mask_valid"].to(device, non_blocking=True)
            class_names = input_data["class_name"]
            dataset_names = input_data["dataset_name"]
            batch_text = torch.stack(
                [
                    text_embeddings[dataset_name][class_name]
                    for dataset_name, class_name in zip(dataset_names, class_names)
                ],
                dim=0,
            )

            optimizer.zero_grad(set_to_none=True)
            patch_logits, image_logits = model(
                image,
                batch_text,
                return_image_logits=True,
            )
            pixel_logits = F.interpolate(
                patch_logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pixel_loss = binary_focal_dice_loss(
                pixel_logits,
                mask,
                sample_valid=mask_valid,
            )
            image_loss = F.binary_cross_entropy_with_logits(image_logits, label)
            loss = args.pixel_loss_weight * pixel_loss
            loss = loss + args.image_loss_weight * image_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
            optimizer.step()

            losses.append(float(loss.detach()))
            pixel_losses.append(float(pixel_loss.detach()))
            image_losses.append(float(image_loss.detach()))
            pixel_supervised_samples += int(mask_valid.sum().item())
            total_samples += int(label.shape[0])
            progress.set_postfix(loss=f"{losses[-1]:.4f}")

        logger.info(
            "epoch %d | loss %.6f | pixel %.6f | image %.6f",
            epoch + 1,
            np.mean(losses),
            np.mean(pixel_losses),
            np.mean(image_losses),
        )
        logger.info(
            "epoch %d supervision | pixel-valid %d/%d | source lengths %s",
            epoch + 1,
            pixel_supervised_samples,
            total_samples,
            dict(zip(training_datasets, dataset_lengths)),
        )
        log_conditioning_weights(
            model,
            text_embeddings,
            logger,
            epoch + 1,
        )
        payload = checkpoint_payload(
            model,
            optimizer,
            epoch + 1,
            args,
            prompt_metadata,
            training_datasets,
            lodo_target,
        )
        torch.save(payload, save_path / "v4_head_latest.pth")
        torch.save(payload, save_path / f"v4_head_epoch_{epoch + 1}.pth")

    logger.info("V4 training complete")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
