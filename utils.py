import os
import random
import numpy as np
import torch
from torch.nn import functional as F
import kornia as K
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
GENERIC_MEDICAL_NOISE_TYPES = (
    "additive",
    "magnitude",
    "signal_dependent",
    "multiplicative",
    "low_frequency",
)
NOISE_TYPE_ALIASES = {
    "gaussian": "additive",
    "rician": "magnitude",
    "ct_quantum": "signal_dependent",
    "quantum": "signal_dependent",
    "speckle": "multiplicative",
    "bias_field": "low_frequency",
    "shot_noise": "shot",
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
        ]
    )


def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a], [0, 1, b]])


def rot_img(x, scale):
    theta = scale
    dtype = torch.FloatTensor
    if x.dim() == 3:
        x = x.unsqueeze(0)
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    x = x.squeeze(0)
    return x


def translation_img(x, translation):
    a, b = translation
    dtype = torch.FloatTensor
    if x.dim() == 3:
        x = x.unsqueeze(0)
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    x = x.squeeze(0)
    return x


def hflip_img(x, **kwargs):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = K.geometry.transform.hflip(x)
    x = x.squeeze(0)
    return x


def vflip_img(x, **kwargs):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = K.geometry.transform.vflip(x)
    x = x.squeeze(0)
    return x


def add_gaussian_noise(x, scale=0.05):
    std = scale
    noise_mask = torch.randn(x.shape[-2:]) > 3
    noise = torch.randn_like(x) * std  # mean = 0
    noised_img = (x + noise) * noise_mask
    noise_img = torch.where(noised_img > 0, noised_img, x)
    return noise_img


def noise_model_for_dataset(dataset_name):
    """Return the intensity-noise family used for a dataset.

    The perturbations deliberately preserve spatial alignment with the mask.
    They are not intended to reproduce an acquisition pipeline exactly; they
    provide two views from which patch-level noise sensitivity can be measured.
    """
    name = dataset_name.lower()
    if "brain" in name:
        return "mri"
    if "ddti" in name:
        return "ultrasound"
    if "liver" in name:
        return "ct"
    if "retina" in name:
        return "retina"
    if "colon" in name or "kvasir" in name:
        return "endoscopy"
    return "generic"


def _smooth_random_field(raw):
    field = torch.randn(
        raw.shape[0], 1, 4, 4, device=raw.device, dtype=raw.dtype
    )
    field = F.interpolate(
        field, size=raw.shape[-2:], mode="bilinear", align_corners=False
    )
    return field / field.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def make_medical_noise_view(
    image,
    dataset_name,
    severity=0.06,
    noise_type=None,
):
    """Create a mask-aligned, modality-aware perturbation of a CLIP input.

    Args:
        image: CLIP-normalized RGB tensor with shape ``[B, 3, H, W]``.
        dataset_name: Name registered in ``dataset/constants.py``.
        severity: Perturbation magnitude in raw ``[0, 1]`` intensity space.
        noise_type: Optional explicit corruption mechanism. When omitted, the
            modality-specific family registered for ``dataset_name`` is used.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B, 3, H, W]")
    if severity < 0:
        raise ValueError("noise severity must be non-negative")
    if severity == 0:
        return image.detach().clone()

    mean = image.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    raw = (image * std + mean).clamp(0.0, 1.0)
    noise_model = noise_type or noise_model_for_dataset(dataset_name)
    noise_model = NOISE_TYPE_ALIASES.get(noise_model.lower(), noise_model.lower())

    if noise_model == "mri":
        # Magnitude-MR approximation: Rician corruption plus a smooth bias field.
        noise_real = torch.randn_like(raw) * severity
        noise_imag = torch.randn_like(raw) * severity
        perturbed = torch.sqrt((raw + noise_real).square() + noise_imag.square())
        perturbed = perturbed * (1.0 + severity * _smooth_random_field(raw))
    elif noise_model in {"ultrasound", "multiplicative"}:
        # Multiplicative speckle with a small electronic-noise component.
        speckle = torch.randn_like(raw) * (severity * 1.5)
        perturbed = raw * (1.0 + speckle)
        perturbed = perturbed + torch.randn_like(raw) * (severity * 0.25)
    elif noise_model in {"ct", "signal_dependent"}:
        # Signal-dependent Gaussian approximation of quantum noise.
        quantum_std = severity * torch.sqrt(raw.clamp_min(1.0 / 255.0))
        perturbed = raw + torch.randn_like(raw) * quantum_std
    elif noise_model in {"retina", "shot"}:
        shot_std = severity * torch.sqrt(raw.clamp_min(1.0 / 255.0))
        perturbed = raw + torch.randn_like(raw) * shot_std
        perturbed = perturbed + torch.randn_like(raw) * (severity * 0.2)
    elif noise_model in {"endoscopy", "illumination"}:
        illumination = torch.empty(
            raw.shape[0], 1, 1, 1, device=raw.device, dtype=raw.dtype
        ).uniform_(1.0 - severity, 1.0 + severity)
        perturbed = raw * illumination
        perturbed = perturbed + torch.randn_like(raw) * (severity * 0.35)
    elif noise_model == "magnitude":
        noise_real = torch.randn_like(raw) * severity
        noise_imag = torch.randn_like(raw) * severity
        perturbed = torch.sqrt((raw + noise_real).square() + noise_imag.square())
    elif noise_model == "low_frequency":
        perturbed = raw * (1.0 + severity * _smooth_random_field(raw))
    elif noise_model in {"additive", "generic"}:
        perturbed = raw + torch.randn_like(raw) * severity
    else:
        valid_types = sorted(
            set(GENERIC_MEDICAL_NOISE_TYPES)
            | {
                "mri",
                "ct",
                "ultrasound",
                "retina",
                "endoscopy",
                "shot",
                "illumination",
            }
            | set(NOISE_TYPE_ALIASES)
        )
        raise ValueError(
            f"unknown noise type {noise_model!r}; available types: {valid_types}"
        )

    perturbed = perturbed.clamp(0.0, 1.0)
    return (perturbed - mean) / std


@torch.no_grad()
def make_random_medical_noise_view(
    image,
    dataset_name,
    noise_types=GENERIC_MEDICAL_NOISE_TYPES,
    severity_min=0.0,
    severity_max=0.06,
    apply_probability=1.0,
    noise_weights=None,
):
    """Apply independently sampled, mask-aligned noise to each batch item."""
    if not 0.0 <= apply_probability <= 1.0:
        raise ValueError("apply_probability must be in [0, 1]")
    if severity_min < 0 or severity_max < severity_min:
        raise ValueError("expected 0 <= severity_min <= severity_max")
    noise_types = tuple(noise_types)
    if not noise_types:
        raise ValueError("noise_types must contain at least one noise mechanism")
    if noise_weights is not None:
        if len(noise_weights) != len(noise_types):
            raise ValueError("noise_weights must match noise_types")
        if any(weight < 0 for weight in noise_weights) or sum(noise_weights) <= 0:
            raise ValueError("noise_weights must be non-negative with a positive sum")

    views = []
    sampled_types = []
    sampled_severities = []
    for sample in image.split(1, dim=0):
        if random.random() >= apply_probability:
            views.append(sample.detach().clone())
            sampled_types.append("clean")
            sampled_severities.append(0.0)
            continue
        sampled_type = random.choices(
            noise_types,
            weights=noise_weights,
            k=1,
        )[0]
        sampled_severity = random.uniform(severity_min, severity_max)
        views.append(
            make_medical_noise_view(
                sample,
                dataset_name,
                severity=sampled_severity,
                noise_type=sampled_type,
            )
        )
        sampled_types.append(sampled_type)
        sampled_severities.append(sampled_severity)
    return torch.cat(views, dim=0), sampled_types, sampled_severities


@torch.no_grad()
def preserve_lesion_contrast(
    original,
    perturbed,
    mask,
    min_retention=0.7,
):
    """Blend overly strong training noise back toward the source image.

    Contrast is measured between the lesion and a local dilated background
    ring. The mask is used only for source-domain training augmentation.
    """
    if not 0.0 <= min_retention <= 1.0:
        raise ValueError("min_retention must be in [0, 1]")
    if min_retention == 0 or mask is None:
        return perturbed
    if original.shape != perturbed.shape:
        raise ValueError("original and perturbed images must have the same shape")
    if mask.ndim != 4 or mask.shape[0] != original.shape[0]:
        raise ValueError("mask must have shape [B, 1, H, W]")

    mean = original.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = original.new_tensor(CLIP_STD).view(1, 3, 1, 1)

    def grayscale(value):
        raw = (value * std + mean).clamp(0.0, 1.0)
        return raw.mean(dim=1, keepdim=True)

    lesion = mask.float().clamp(0.0, 1.0)
    dilated = F.max_pool2d(lesion, kernel_size=17, stride=1, padding=8)
    background = (dilated - lesion).clamp(0.0, 1.0)
    lesion_count = lesion.sum(dim=(-2, -1), keepdim=True)
    background_count = background.sum(dim=(-2, -1), keepdim=True)
    valid = (lesion_count > 0) & (background_count > 0)

    def local_contrast(value):
        gray = grayscale(value)
        lesion_mean = (gray * lesion).sum(dim=(-2, -1), keepdim=True)
        lesion_mean = lesion_mean / lesion_count.clamp_min(1.0)
        background_mean = (gray * background).sum(dim=(-2, -1), keepdim=True)
        background_mean = background_mean / background_count.clamp_min(1.0)
        return (lesion_mean - background_mean).abs()

    original_contrast = local_contrast(original)
    perturbed_contrast = local_contrast(perturbed)
    ratio = perturbed_contrast / original_contrast.clamp_min(1e-6)
    needs_blend = valid & (original_contrast > 1e-4) & (ratio < min_retention)
    max_alpha = (1.0 - min_retention) / (1.0 - ratio).clamp_min(1e-6)
    alpha = torch.where(needs_blend, max_alpha.clamp(0.0, 1.0), torch.ones_like(ratio))
    blended = original + alpha * (perturbed - original)

    # A sign change in lesion/background contrast can violate the linear
    # estimate above. Halving the perturbation provides a safe bounded fallback.
    for _ in range(4):
        blended_ratio = local_contrast(blended) / original_contrast.clamp_min(1e-6)
        still_low = valid & (original_contrast > 1e-4) & (
            blended_ratio < min_retention
        )
        alpha = torch.where(still_low, alpha * 0.5, alpha)
        blended = original + alpha * (perturbed - original)
    return blended


def cos_sim(a_norm, b_norm):
    if len(a_norm.shape) == 2:
        sim_mt = b_norm @ a_norm.transpose(1, 0)
    elif len(a_norm.shape) == 1:
        sim_mt = b_norm @ a_norm
    else:
        raise NotImplementedError
    return sim_mt


# 定义一个自定义的噪音类
class AddGaussianNoise(object):
    def __init__(self, std=1.0, p=0.5):
        """
        mean: 高斯噪声的均值
        std: 高斯噪声的标准差
        p: 添加噪音的概率
        """
        self.std = std
        self.p = p

    def __call__(self, x):
        """
        在数据张量上应用噪音
        """
        if random.random() < self.p:
            return x
        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        noise_mask = (torch.randn(x.shape[-2:]) > 3).int()
        noise = torch.randn_like(x) * self.std  # mean = 0
        noised_img = (1 - noise_mask) * x + noise * x * noise_mask
        noised_img = torch.clamp(noised_img, 0.0, 1.0)
        return transforms.ToPILImage()(noised_img)

    def __repr__(self):
        return self.__class__.__name__ + f"(std={self.std}, p={self.p})"
