import json
import math
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import CLASS_NAMES, DATA_PATH


def _read_metadata(meta_path):
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"metadata file not found: {meta_path}; create a JSONL file before "
            "training or evaluation"
        )
    metadata = []
    with open(meta_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            for required_key in ("image_path", "label", "class_name"):
                if required_key not in item:
                    raise ValueError(
                        f"{meta_path}:{line_number} has no {required_key!r} field"
                    )
            metadata.append(item)
    if not metadata:
        raise ValueError(f"metadata file is empty: {meta_path}")
    return metadata


def _load_optional_anomaly_mask(meta, data_path, img_size, transform_mask):
    """Return a placeholder mask plus explicit supervision availability."""
    is_anomaly = bool(meta["label"])
    mask_relative_path = meta.get("mask_path")
    has_anomaly_mask = is_anomaly and bool(mask_relative_path)
    if has_anomaly_mask:
        mask_path = os.path.join(data_path, mask_relative_path)
        mask = Image.open(mask_path).convert("L")
        mask = (transform_mask(mask) != 0).float()
    else:
        mask = torch.zeros([1, img_size, img_size])

    # A normal image is valid all-zero pixel supervision even without a mask
    # file. An abnormal image without a mask has unknown pixel labels.
    mask_valid = (not is_anomaly) or has_anomaly_mask
    return mask, mask_valid, has_anomaly_mask


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        text: bool = False,
        dataset_name: str | None = None,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.text = text
        self.dataset_name = dataset_name
        self.full_shot = "full-shot" in meta_path
        self.meta = _read_metadata(meta_path)

        self.transforms_list = [
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=math.degrees(math.pi / 6))], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]

        transform_x = []
        # transform_x.append(AddGaussianNoise(std=1, p=0.7))
        if not text:
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=0.7)
            )
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=0.7)
            )
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], p=0.7)
            )
        self.transform_x = transforms.Compose(
            transform_x
            + [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        data_path = self.data_path
        img_path = os.path.join(data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")

        img = self.transform_x(img)
        mask, mask_valid, has_anomaly_mask = _load_optional_anomaly_mask(
            meta,
            data_path,
            self.img_size,
            self.transform_mask,
        )

        random_transform = transforms.Compose(self.transforms_list)
        transform_tensor = torch.cat([img, mask], dim=0)
        assert transform_tensor.shape[0] == 4
        transform_tensor = random_transform(transform_tensor)
        img = transform_tensor[0:3, :, :]
        mask = transform_tensor[3:4, :, :]

        inputs = {
            "image": img,
            "mask": mask,
            "label": torch.tensor(meta["label"]).to(torch.int64),
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
            "dataset_name": self.dataset_name or meta.get("dataset_name", "unknown"),
            "mask_valid": torch.tensor(mask_valid, dtype=torch.bool),
            "has_anomaly_mask": torch.tensor(
                has_anomaly_mask,
                dtype=torch.bool,
            ),
        }
        return inputs


class BaseSingleClassDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        class_name: str,
        dataset_name: str | None = None,
        logger=None,
    ):

        assert class_name is not None, "class_name should be provided"
        self.data_path = data_path
        self.img_size = img_size
        self.dataset_name = dataset_name
        self.meta = [
            item
            for item in _read_metadata(meta_path)
            if item["class_name"] == class_name
        ]
        if not self.meta:
            raise ValueError(
                f"metadata {meta_path} has no samples for class {class_name!r}"
            )

        # Define transforms
        self.transform_x = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(  # set image / mean metadata from pretrained_cfg if available, or use default
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

        # logging
        if logger:
            logger.info(f"Class name: {class_name}")
            logger.info(f"Sample number: {len(self.meta)}")
            logger.info("=====================================")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        img_path = os.path.join(self.data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform_x(img)
        mask, mask_valid, has_anomaly_mask = _load_optional_anomaly_mask(
            meta,
            self.data_path,
            self.img_size,
            self.transform_mask,
        )
        inputs = {
            "image": img,
            "mask": mask,
            "label": meta["label"],
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
            "dataset_name": self.dataset_name or meta.get("dataset_name", "unknown"),
            "mask_valid": torch.tensor(mask_valid, dtype=torch.bool),
            "has_anomaly_mask": torch.tensor(
                has_anomaly_mask,
                dtype=torch.bool,
            ),
        }
        return inputs


def get_dataset(
    dataset_name: str,
    img_size: int,
    training_mode: str,
    shot: int = -1,
    stage: str = "train",
    logger=None,
):
    if "Med" not in dataset_name:
        assert dataset_name in DATA_PATH, (
            f"Dataset {dataset_name} not found; available datasets: {list(DATA_PATH.keys())}"
        )

    if stage == "train":
        if training_mode == "few_shot":
            assert shot > 0, "shot should be positive"
            meta_path = os.path.join(
                "./dataset/metadata", dataset_name, f"{shot}-shot.jsonl"
            )
        else:
            meta_path = os.path.join(
                "./dataset/metadata", dataset_name, "full-shot.jsonl"
            )

        data_key = (
            dataset_name if dataset_name in DATA_PATH else dataset_name.split("-")[0]
        )
        data_path = DATA_PATH[data_key]
        text_dataset = BaseDataset(
            data_path,
            meta_path,
            img_size,
            text=True,
            dataset_name=dataset_name,
        )
        image_dataset = BaseDataset(
            data_path,
            meta_path,
            img_size,
            text=False,
            dataset_name=dataset_name,
        )
        return text_dataset, image_dataset
    elif stage == "test":
        meta_path = os.path.join("./dataset/metadata", dataset_name, "full-shot.jsonl")
        class_names = CLASS_NAMES[dataset_name]
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                dataset_name=dataset_name,
                logger=logger,
            )
            datasets[class_name] = image_dataset
        return datasets
    elif stage == "visualize":
        class_names = CLASS_NAMES[dataset_name]
        meta_path = os.path.join("./dataset/metadata", dataset_name, "full-shot.jsonl")
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                dataset_name=dataset_name,
                logger=None,
            )
            datasets[class_name] = image_dataset
        return datasets
    else:
        raise ValueError(f"stage {stage} not found; available stages: train, test")
