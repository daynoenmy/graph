import os
import random
import numpy as np
import torch
from torch.nn import functional as F
import kornia as K
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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


@torch.no_grad()
def make_medical_noise_view(image, dataset_name, severity=0.06):
    """Create a mask-aligned, modality-aware perturbation of a CLIP input.

    Args:
        image: CLIP-normalized RGB tensor with shape ``[B, 3, H, W]``.
        dataset_name: Name registered in ``dataset/constants.py``.
        severity: Perturbation magnitude in raw ``[0, 1]`` intensity space.
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
    noise_model = noise_model_for_dataset(dataset_name)

    if noise_model == "mri":
        # Magnitude-MR approximation: Rician corruption plus a smooth bias field.
        noise_real = torch.randn_like(raw) * severity
        noise_imag = torch.randn_like(raw) * severity
        perturbed = torch.sqrt((raw + noise_real).square() + noise_imag.square())
        field = torch.randn(
            raw.shape[0], 1, 4, 4, device=raw.device, dtype=raw.dtype
        )
        field = F.interpolate(
            field, size=raw.shape[-2:], mode="bilinear", align_corners=False
        )
        field = field / field.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        perturbed = perturbed * (1.0 + severity * field)
    elif noise_model == "ultrasound":
        # Multiplicative speckle with a small electronic-noise component.
        speckle = torch.randn_like(raw) * (severity * 1.5)
        perturbed = raw * (1.0 + speckle)
        perturbed = perturbed + torch.randn_like(raw) * (severity * 0.25)
    elif noise_model == "ct":
        # Signal-dependent Gaussian approximation of quantum noise.
        quantum_std = severity * torch.sqrt(raw.clamp_min(1.0 / 255.0))
        perturbed = raw + torch.randn_like(raw) * quantum_std
    elif noise_model == "retina":
        shot_std = severity * torch.sqrt(raw.clamp_min(1.0 / 255.0))
        perturbed = raw + torch.randn_like(raw) * shot_std
        perturbed = perturbed + torch.randn_like(raw) * (severity * 0.2)
    elif noise_model == "endoscopy":
        illumination = torch.empty(
            raw.shape[0], 1, 1, 1, device=raw.device, dtype=raw.dtype
        ).uniform_(1.0 - severity, 1.0 + severity)
        perturbed = raw * illumination
        perturbed = perturbed + torch.randn_like(raw) * (severity * 0.35)
    else:
        perturbed = raw + torch.randn_like(raw) * severity

    perturbed = perturbed.clamp(0.0, 1.0)
    return (perturbed - mean) / std


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
