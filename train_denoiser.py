"""Self-supervised blind-spot + conditional-diffusion denoising pipeline.

The commands are intentionally separate because diffusion training and sampling
are long-running stages that should be resumable and inspectable independently.
All stages use images only; labels and masks in metadata files are ignored.
"""

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from dataset.constants import DATA_PATH
from model.denoising import (
    ConditionalDiffusionUNet,
    GaussianDiffusion,
    LightweightRawDenoiser,
    MaskedBlindSpotUNet,
    blind_spot_reconstruct,
    make_random_blind_spot_mask,
    replace_masked_with_neighbors,
)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_data_paths(args) -> tuple[str, str]:
    data_root = args.data_root
    metadata_path = args.metadata_path
    if data_root is None and args.dataset is not None:
        data_root = DATA_PATH.get(args.dataset)
    if metadata_path is None and args.dataset is not None:
        metadata_path = os.path.join(
            "dataset", "metadata", args.dataset, "full-shot.jsonl"
        )
    if data_root is None or metadata_path is None:
        raise ValueError(
            "provide --data-root and --metadata-path, or use a dataset registered "
            "in dataset/constants.py"
        )
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")
    return data_root, metadata_path


def safe_relative_output_path(image_path: str) -> Path:
    normalized = image_path.replace("\\", "/")
    parts = [
        part
        for part in Path(normalized).parts
        if part not in ("", "/", ".", "..") and not part.endswith(":")
    ]
    if not parts:
        raise ValueError(f"invalid image path in metadata: {image_path}")
    return Path(*parts).with_suffix(".png")


def read_records(metadata_path: str) -> list[dict]:
    records = []
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "image_path" not in record:
                raise KeyError(
                    f"missing image_path in {metadata_path} line {line_number}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"no image records found in {metadata_path}")
    return records


class DenoisingMetadataDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        metadata_path: str,
        image_size: int,
        image_channels: int = 1,
        paired_root: Optional[str] = None,
        augment: bool = False,
    ):
        self.data_root = data_root
        self.records = read_records(metadata_path)
        self.image_size = image_size
        self.image_channels = image_channels
        self.paired_root = paired_root
        self.augment = augment
        if image_channels not in (1, 3):
            raise ValueError("image_channels must be 1 or 3")

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> torch.Tensor:
        mode = "L" if self.image_channels == 1 else "RGB"
        with Image.open(path) as image:
            image = image.convert(mode)
            image = image.resize(
                (self.image_size, self.image_size), resample=Image.Resampling.BICUBIC
            )
            return TF.to_tensor(image)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image_path = record["image_path"]
        source_path = (
            image_path if os.path.isabs(image_path) else os.path.join(self.data_root, image_path)
        )
        image = self._load_image(source_path)
        sample = {
            "image": image,
            "relative_path": image_path,
        }
        if self.paired_root is not None:
            paired_path = os.path.join(
                self.paired_root, safe_relative_output_path(image_path)
            )
            if not os.path.isfile(paired_path):
                raise FileNotFoundError(f"paired image not found: {paired_path}")
            sample["target"] = self._load_image(paired_path)

        if self.augment:
            if random.random() < 0.5:
                sample["image"] = torch.flip(sample["image"], dims=(-1,))
                if "target" in sample:
                    sample["target"] = torch.flip(sample["target"], dims=(-1,))
            if random.random() < 0.5:
                sample["image"] = torch.flip(sample["image"], dims=(-2,))
                if "target" in sample:
                    sample["target"] = torch.flip(sample["target"], dims=(-2,))
        return sample


def make_loader(dataset: Dataset, args, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= args.batch_size,
    )


def save_image_batch(
    images: torch.Tensor,
    relative_paths: list[str],
    output_root: str,
) -> None:
    images = images.detach().cpu().clamp(0.0, 1.0)
    for image, relative_path in zip(images, relative_paths):
        output_path = Path(output_root) / safe_relative_output_path(relative_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        TF.to_pil_image(image).save(output_path)


def save_checkpoint(state: dict, output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state, os.path.join(output_dir, filename))


def load_checkpoint(path: Optional[str], device: torch.device) -> Optional[dict]:
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def train_blind_spot(args) -> None:
    setup_seed(args.seed)
    device = select_device(args.device)
    data_root, metadata_path = resolve_data_paths(args)
    dataset = DenoisingMetadataDataset(
        data_root,
        metadata_path,
        args.image_size,
        args.channels,
        augment=True,
    )
    loader = make_loader(dataset, args, shuffle=True)
    model = MaskedBlindSpotUNet(args.channels, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 0
    checkpoint = load_checkpoint(args.resume, device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]

    config = {
        "image_channels": args.channels,
        "base_channels": args.base_channels,
        "image_size": args.image_size,
        "mask_probability": args.mask_probability,
    }
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        progress = tqdm(loader, desc=f"blind-spot {epoch + 1}/{args.epochs}")
        for batch in progress:
            image = batch["image"].to(device)
            mask = make_random_blind_spot_mask(image, args.mask_probability)
            masked_image = replace_masked_with_neighbors(image, mask)
            prediction = model(masked_image)
            expanded_mask = mask.expand_as(image)
            loss = (
                (prediction - image).abs() * expanded_mask
            ).sum() / expanded_mask.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.5f}")

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "mean_loss": float(np.mean(losses)),
        }
        save_checkpoint(state, args.output_dir, "blind_spot_latest.pth")
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                state, args.output_dir, f"blind_spot_epoch_{epoch + 1}.pth"
            )


def build_blind_spot_from_checkpoint(checkpoint: dict, device: torch.device):
    config = checkpoint.get("config", {})
    model = MaskedBlindSpotUNet(
        config.get("image_channels", 1), config.get("base_channels", 32)
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


@torch.no_grad()
def cache_blind_spot_conditions(args) -> None:
    setup_seed(args.seed)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.blind_spot_checkpoint, device)
    model, config = build_blind_spot_from_checkpoint(checkpoint, device)
    channels = config.get("image_channels", args.channels)
    image_size = args.image_size or config.get("image_size", 256)
    data_root, metadata_path = resolve_data_paths(args)
    dataset = DenoisingMetadataDataset(
        data_root, metadata_path, image_size, channels, augment=False
    )
    loader = make_loader(dataset, args, shuffle=False)
    for batch in tqdm(loader, desc="cache blind-spot conditions"):
        image = batch["image"].to(device)
        condition = blind_spot_reconstruct(model, image, args.blind_spot_stride)
        save_image_batch(condition, batch["relative_path"], args.output_dir)


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(
            ema_model.parameters(), model.parameters()
        ):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def train_diffusion(args) -> None:
    setup_seed(args.seed)
    device = select_device(args.device)
    data_root, metadata_path = resolve_data_paths(args)
    dataset = DenoisingMetadataDataset(
        data_root,
        metadata_path,
        args.image_size,
        args.channels,
        paired_root=args.condition_root,
        augment=True,
    )
    loader = make_loader(dataset, args, shuffle=True)
    model = ConditionalDiffusionUNet(args.channels, args.base_channels).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = GaussianDiffusion(args.timesteps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    checkpoint = load_checkpoint(args.resume, device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]

    config = {
        "image_channels": args.channels,
        "base_channels": args.base_channels,
        "image_size": args.image_size,
        "timesteps": args.timesteps,
    }
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        progress = tqdm(loader, desc=f"diffusion {epoch + 1}/{args.epochs}")
        for batch in progress:
            noisy_observation = batch["image"].to(device) * 2.0 - 1.0
            condition = batch["target"].to(device) * 2.0 - 1.0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = diffusion.training_loss(model, noisy_observation, condition)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model, args.ema_decay)
            losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.5f}")

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "mean_loss": float(np.mean(losses)),
        }
        save_checkpoint(state, args.output_dir, "diffusion_latest.pth")
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                state, args.output_dir, f"diffusion_epoch_{epoch + 1}.pth"
            )


def build_diffusion_from_checkpoint(checkpoint: dict, device: torch.device):
    config = checkpoint.get("config", {})
    channels = config.get("image_channels", 1)
    model = ConditionalDiffusionUNet(
        channels, config.get("base_channels", 64)
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    model.eval()
    diffusion = GaussianDiffusion(config.get("timesteps", 1000)).to(device)
    return model, diffusion, config


@torch.no_grad()
def generate_pseudo_clean(args) -> None:
    setup_seed(args.seed)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.diffusion_checkpoint, device)
    model, diffusion, config = build_diffusion_from_checkpoint(checkpoint, device)
    channels = config.get("image_channels", args.channels)
    image_size = args.image_size or config.get("image_size", 256)
    data_root, metadata_path = resolve_data_paths(args)
    dataset = DenoisingMetadataDataset(
        data_root,
        metadata_path,
        image_size,
        channels,
        paired_root=args.condition_root,
        augment=False,
    )
    loader = make_loader(dataset, args, shuffle=False)
    for batch in tqdm(loader, desc="generate pseudo-clean images"):
        condition = batch["target"].to(device) * 2.0 - 1.0
        if args.single_sample:
            generated = diffusion.ddim_sample(
                model, condition, args.sampling_steps
            )
        else:
            generated = diffusion.symmetric_ddim_sample(
                model, condition, args.sampling_steps
            )
        generated = (generated + 1.0) * 0.5
        save_image_batch(generated, batch["relative_path"], args.output_dir)


def local_ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
) -> torch.Tensor:
    padding = window_size // 2
    mean_prediction = F.avg_pool2d(
        prediction, window_size, stride=1, padding=padding
    )
    mean_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    variance_prediction = F.avg_pool2d(
        prediction * prediction, window_size, stride=1, padding=padding
    ) - mean_prediction.square()
    variance_target = F.avg_pool2d(
        target * target, window_size, stride=1, padding=padding
    ) - mean_target.square()
    covariance = F.avg_pool2d(
        prediction * target, window_size, stride=1, padding=padding
    ) - mean_prediction * mean_target
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = (
        (2.0 * mean_prediction * mean_target + c1)
        * (2.0 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (variance_prediction + variance_target + c2)
        ).clamp_min(1e-8)
    )
    return 1.0 - ssim.mean()


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    prediction_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(prediction_dx, target_dx) + F.l1_loss(
        prediction_dy, target_dy
    )


def train_student(args) -> None:
    setup_seed(args.seed)
    device = select_device(args.device)
    data_root, metadata_path = resolve_data_paths(args)
    dataset = DenoisingMetadataDataset(
        data_root,
        metadata_path,
        args.image_size,
        args.channels,
        paired_root=args.pseudo_clean_root,
        augment=True,
    )
    loader = make_loader(dataset, args, shuffle=True)
    model = LightweightRawDenoiser(
        args.channels,
        args.width,
        args.depth,
        args.max_residual,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 0
    checkpoint = load_checkpoint(args.resume, device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["student"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]

    config = {
        "image_channels": args.channels,
        "width": args.width,
        "depth": args.depth,
        "max_residual": args.max_residual,
        "image_size": args.image_size,
    }
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        progress = tqdm(loader, desc=f"student {epoch + 1}/{args.epochs}")
        for batch in progress:
            noisy_observation = batch["image"].to(device)
            pseudo_clean = batch["target"].to(device)
            prediction = model(noisy_observation)
            reconstruction = F.smooth_l1_loss(prediction, pseudo_clean)
            structural = local_ssim_loss(prediction, pseudo_clean)
            edges = edge_loss(prediction, pseudo_clean)
            loss = (
                reconstruction
                + args.ssim_weight * structural
                + args.edge_weight * edges
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())
            progress.set_postfix(
                loss=f"{loss.item():.5f}", l1=f"{reconstruction.item():.5f}"
            )

        state = {
            "epoch": epoch + 1,
            "student": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "mean_loss": float(np.mean(losses)),
        }
        save_checkpoint(state, args.output_dir, "input_denoiser.pth")
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                state, args.output_dir, f"input_denoiser_epoch_{epoch + 1}.pth"
            )


def add_common_data_arguments(
    parser: argparse.ArgumentParser,
    image_size_default: Optional[int] = 256,
) -> None:
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--metadata-path", type=str)
    parser.add_argument("--image-size", type=int, default=image_size_default)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=111)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--save-every", type=int, default=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-supervised diffusion denoising for AA-CLIP"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    blind_spot = subparsers.add_parser("train-blind-spot")
    add_common_data_arguments(blind_spot)
    add_training_arguments(blind_spot)
    blind_spot.add_argument("--base-channels", type=int, default=32)
    blind_spot.add_argument("--mask-probability", type=float, default=0.05)
    blind_spot.set_defaults(function=train_blind_spot)

    cache = subparsers.add_parser("cache-conditions")
    add_common_data_arguments(cache, image_size_default=None)
    cache.add_argument("--blind-spot-checkpoint", type=str, required=True)
    cache.add_argument("--blind-spot-stride", type=int, default=4)
    cache.add_argument("--output-dir", type=str, required=True)
    cache.set_defaults(function=cache_blind_spot_conditions)

    diffusion = subparsers.add_parser("train-diffusion")
    add_common_data_arguments(diffusion)
    add_training_arguments(diffusion)
    diffusion.add_argument("--condition-root", type=str, required=True)
    diffusion.add_argument("--base-channels", type=int, default=64)
    diffusion.add_argument("--timesteps", type=int, default=1000)
    diffusion.add_argument("--ema-decay", type=float, default=0.9999)
    diffusion.add_argument("--amp", action="store_true")
    diffusion.set_defaults(function=train_diffusion)

    generate = subparsers.add_parser("generate-pseudo-clean")
    add_common_data_arguments(generate, image_size_default=None)
    generate.add_argument("--condition-root", type=str, required=True)
    generate.add_argument("--diffusion-checkpoint", type=str, required=True)
    generate.add_argument("--sampling-steps", type=int, default=50)
    generate.add_argument("--single-sample", action="store_true")
    generate.add_argument("--output-dir", type=str, required=True)
    generate.set_defaults(function=generate_pseudo_clean)

    student = subparsers.add_parser("train-student")
    add_common_data_arguments(student)
    add_training_arguments(student)
    student.add_argument("--pseudo-clean-root", type=str, required=True)
    student.add_argument("--width", type=int, default=32)
    student.add_argument("--depth", type=int, default=5)
    student.add_argument("--max-residual", type=float, default=0.2)
    student.add_argument("--ssim-weight", type=float, default=0.2)
    student.add_argument("--edge-weight", type=float, default=0.05)
    student.set_defaults(function=train_student)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
