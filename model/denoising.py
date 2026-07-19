import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        groups = _group_count(c_out)
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(groups, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(groups, c_out),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MaskedBlindSpotUNet(nn.Module):
    """U-Net trained only on pixels hidden from its input.

    Full-image inference must use :func:`blind_spot_reconstruct`, which hides
    every output pixel in one of several lattice passes. Calling ``forward``
    directly is intended for masked training inputs only.
    """

    def __init__(self, image_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.enc1 = ConvBlock(image_channels, base_channels)
        self.down1 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.mid = ConvBlock(base_channels * 4, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        h = self.mid(self.down2(e2))

        h = self.up2(h)
        if h.shape[-2:] != e2.shape[-2:]:
            h = F.interpolate(h, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        h = self.dec2(torch.cat([h, e2], dim=1))

        h = self.up1(h)
        if h.shape[-2:] != e1.shape[-2:]:
            h = F.interpolate(h, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.dec1(torch.cat([h, e1], dim=1))
        return torch.sigmoid(self.out(h))


def make_random_blind_spot_mask(
    image: torch.Tensor, mask_probability: float = 0.05
) -> torch.Tensor:
    if not 0.0 < mask_probability < 1.0:
        raise ValueError("mask_probability must be between 0 and 1")
    shape = (image.shape[0], 1, image.shape[2], image.shape[3])
    return (torch.rand(shape, device=image.device) < mask_probability).to(image.dtype)


def replace_masked_with_neighbors(
    image: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    reflected = F.pad(image, (1, 1, 1, 1), mode="reflect")
    padded_sum = F.avg_pool2d(reflected, kernel_size=3, stride=1) * 9.0
    neighbor_mean = (padded_sum - image) / 8.0
    return image * (1.0 - mask) + neighbor_mean * mask


@torch.no_grad()
def blind_spot_reconstruct(
    model: nn.Module,
    image: torch.Tensor,
    stride: int = 4,
) -> torch.Tensor:
    """J-invariant reconstruction in which every predicted pixel was hidden."""
    if stride < 2:
        raise ValueError("blind-spot reconstruction stride must be at least 2")
    output = torch.zeros_like(image)
    coverage = torch.zeros(
        image.shape[0], 1, image.shape[2], image.shape[3],
        device=image.device, dtype=image.dtype
    )
    for row_offset in range(stride):
        for col_offset in range(stride):
            mask = torch.zeros_like(coverage)
            mask[:, :, row_offset::stride, col_offset::stride] = 1.0
            masked_image = replace_masked_with_neighbors(image, mask)
            prediction = model(masked_image)
            output.add_(prediction * mask)
            coverage.add_(mask)
    return (output / coverage.clamp_min(1.0)).clamp(0.0, 1.0)


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int
) -> torch.Tensor:
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        half_dim, device=timesteps.device, dtype=torch.float32
    ) / max(half_dim - 1, 1)
    frequencies = exponent.exp()
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([arguments.sin(), arguments.cos()], dim=1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class DiffusionResBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        time_dim: int,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, c_out)
        self.norm2 = nn.GroupNorm(_group_count(c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if c_in == c_out
            else nn.Conv2d(c_in, c_out, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(F.silu(time_embedding)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalDiffusionUNet(nn.Module):
    """Compact conditional U-Net that predicts diffusion noise."""

    def __init__(self, image_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.image_channels = image_channels
        self.base_channels = base_channels
        time_dim = base_channels * 4
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_conv = nn.Conv2d(
            image_channels * 2, base_channels, kernel_size=3, padding=1
        )
        self.down_block1 = DiffusionResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.down_block2 = DiffusionResBlock(
            base_channels * 2, base_channels * 2, time_dim
        )
        self.downsample2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.mid_block1 = DiffusionResBlock(
            base_channels * 4, base_channels * 4, time_dim
        )
        self.mid_block2 = DiffusionResBlock(
            base_channels * 4, base_channels * 4, time_dim
        )
        self.upsample2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_block2 = DiffusionResBlock(
            base_channels * 4, base_channels * 2, time_dim
        )
        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_block1 = DiffusionResBlock(
            base_channels * 2, base_channels, time_dim
        )
        self.out_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.out_conv = nn.Conv2d(
            base_channels, image_channels, kernel_size=3, padding=1
        )

    def forward(
        self,
        noisy_image: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = sinusoidal_timestep_embedding(
            timesteps, self.base_channels
        ).to(noisy_image.dtype)
        time_embedding = self.time_mlp(time_embedding)

        h = self.input_conv(torch.cat([noisy_image, condition], dim=1))
        skip1 = self.down_block1(h, time_embedding)
        skip2 = self.down_block2(self.downsample1(skip1), time_embedding)
        h = self.downsample2(skip2)
        h = self.mid_block1(h, time_embedding)
        h = self.mid_block2(h, time_embedding)

        h = self.upsample2(h)
        if h.shape[-2:] != skip2.shape[-2:]:
            h = F.interpolate(h, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up_block2(torch.cat([h, skip2], dim=1), time_embedding)

        h = self.upsample1(h)
        if h.shape[-2:] != skip1.shape[-2:]:
            h = F.interpolate(h, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up_block1(torch.cat([h, skip1], dim=1), time_embedding)
        return self.out_conv(F.silu(self.out_norm(h)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        if num_timesteps < 2:
            raise ValueError("num_timesteps must be at least 2")
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.num_timesteps = num_timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape) -> torch.Tensor:
        result = values.gather(0, timesteps)
        return result.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(
        self,
        clean_image: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_image)
        return (
            self._extract(self.sqrt_alpha_bars, timesteps, clean_image.shape) * clean_image
            + self._extract(
                self.sqrt_one_minus_alpha_bars, timesteps, clean_image.shape
            )
            * noise
        )

    def training_loss(
        self,
        model: nn.Module,
        image: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = torch.randint(
            0, self.num_timesteps, (image.shape[0],), device=image.device
        )
        noise = torch.randn_like(image)
        noisy_image = self.q_sample(image, timesteps, noise)
        predicted_noise = model(noisy_image, timesteps, condition)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        sampling_steps: int = 50,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sampling_steps = min(max(2, sampling_steps), self.num_timesteps)
        if initial_noise is None:
            image = torch.randn_like(condition)
        else:
            image = initial_noise.clone()
        timesteps = torch.linspace(
            self.num_timesteps - 1,
            0,
            sampling_steps,
            device=condition.device,
        ).round().long()

        for index, timestep in enumerate(timesteps):
            batch_timesteps = torch.full(
                (condition.shape[0],),
                int(timestep.item()),
                device=condition.device,
                dtype=torch.long,
            )
            predicted_noise = model(image, batch_timesteps, condition)
            alpha_bar = self.alpha_bars[timestep]
            predicted_clean = (
                image - (1.0 - alpha_bar).sqrt() * predicted_noise
            ) / alpha_bar.sqrt().clamp_min(1e-8)
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)

            if index == len(timesteps) - 1:
                image = predicted_clean
                continue
            previous_alpha_bar = self.alpha_bars[timesteps[index + 1]]
            image = (
                previous_alpha_bar.sqrt() * predicted_clean
                + (1.0 - previous_alpha_bar).sqrt() * predicted_noise
            )
        return image.clamp(-1.0, 1.0)

    @torch.no_grad()
    def symmetric_ddim_sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        sampling_steps: int = 50,
    ) -> torch.Tensor:
        initial_noise = torch.randn_like(condition)
        positive = self.ddim_sample(
            model, condition, sampling_steps, initial_noise=initial_noise
        )
        negative = self.ddim_sample(
            model, condition, sampling_steps, initial_noise=-initial_noise
        )
        return ((positive + negative) * 0.5).clamp(-1.0, 1.0)


class StudentResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.block(x)


class LightweightRawDenoiser(nn.Module):
    """Fast deterministic student operating on raw [0, 1] images."""

    def __init__(
        self,
        image_channels: int = 1,
        width: int = 32,
        depth: int = 5,
        max_residual: float = 0.2,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.width = width
        self.depth = depth
        self.max_residual = max_residual
        self.head = nn.Conv2d(image_channels, width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[StudentResidualBlock(width) for _ in range(depth)]
        )
        self.tail = nn.Conv2d(width, image_channels, kernel_size=3, padding=1)
        self.reset_identity()

    def reset_identity(self) -> None:
        nn.init.zeros_(self.tail.weight)
        if self.tail.bias is not None:
            nn.init.zeros_(self.tail.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = F.gelu(self.head(image))
        features = self.blocks(features)
        residual = self.max_residual * torch.tanh(self.tail(features))
        return (image + residual).clamp(0.0, 1.0)


class CLIPInputDenoiser(nn.Module):
    """Wrap a raw-image student around CLIP normalization."""

    def __init__(
        self,
        image_channels: int = 1,
        width: int = 32,
        depth: int = 5,
        max_residual: float = 0.2,
    ):
        super().__init__()
        if image_channels not in (1, 3):
            raise ValueError("image_channels must be 1 or 3")
        self.image_channels = image_channels
        self.student = LightweightRawDenoiser(
            image_channels=image_channels,
            width=width,
            depth=depth,
            max_residual=max_residual,
        )
        self.register_buffer(
            "clip_mean",
            torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def reset_identity(self) -> None:
        self.student.reset_identity()

    def _to_student_channels(self, raw_rgb: torch.Tensor) -> torch.Tensor:
        if self.image_channels == 3:
            return raw_rgb
        weights = raw_rgb.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        weights = weights / weights.sum()
        return (raw_rgb * weights).sum(dim=1, keepdim=True)

    def _to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 3:
            return image
        return image.expand(-1, 3, -1, -1)

    def denoise_raw(self, raw_rgb: torch.Tensor) -> torch.Tensor:
        student_input = self._to_student_channels(raw_rgb)
        return self._to_rgb(self.student(student_input))

    def forward(self, normalized_rgb: torch.Tensor) -> torch.Tensor:
        mean = self.clip_mean.to(dtype=normalized_rgb.dtype)
        std = self.clip_std.to(dtype=normalized_rgb.dtype)
        raw_rgb = (normalized_rgb * std + mean).clamp(0.0, 1.0)
        denoised_rgb = self.denoise_raw(raw_rgb)
        return (denoised_rgb - mean) / std


def load_student_checkpoint(
    module: CLIPInputDenoiser,
    checkpoint_path: str,
    map_location="cpu",
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if "student" in checkpoint:
        state_dict = checkpoint["student"]
    elif "input_denoiser" in checkpoint:
        state_dict = checkpoint["input_denoiser"]
    else:
        state_dict = checkpoint

    # Distillation checkpoints contain the raw student. Main-model checkpoints
    # can contain the complete CLIPInputDenoiser wrapper.
    if any(key.startswith("student.") for key in state_dict):
        module.load_state_dict(state_dict, strict=True)
    else:
        module.student.load_state_dict(state_dict, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {"state_dict": checkpoint}


def read_student_checkpoint_config(
    checkpoint_path: str,
    map_location="cpu",
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        return {}
    return checkpoint.get("config", {})
