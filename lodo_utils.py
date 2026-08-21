import json
import math
import os

import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from dataset.constants import CLASS_NAMES, DATA_PATH, DOMAINS, REAL_NAMES


DEFAULT_LODO_DATASETS = ["Chest", "Liver", "Brain", "OCT2017", "RESC", "HIS"]


class HomogeneousDatasetBatchSampler(Sampler):
    """Build batches without mixing component datasets of a ConcatDataset.

    Every source sample is still visited once per epoch. Consequently, this
    sampler isolates dataset-specific supervision within a batch, but it does
    not by itself balance datasets with different numbers of samples.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if not isinstance(dataset, ConcatDataset):
            raise TypeError("dataset must be a torch.utils.data.ConcatDataset")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        batches = []
        start = 0
        for end in self.dataset.cumulative_sizes:
            dataset_size = end - start
            if self.shuffle:
                local_indices = torch.randperm(dataset_size).tolist()
            else:
                local_indices = list(range(dataset_size))
            source_indices = [start + index for index in local_indices]
            for offset in range(0, dataset_size, self.batch_size):
                batch = source_indices[offset : offset + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
            start = end

        if self.shuffle and batches:
            batch_order = torch.randperm(len(batches)).tolist()
            batches = [batches[index] for index in batch_order]
        yield from batches

    def __len__(self):
        if self.drop_last:
            return sum(
                len(dataset) // self.batch_size for dataset in self.dataset.datasets
            )
        return sum(
            math.ceil(len(dataset) / self.batch_size)
            for dataset in self.dataset.datasets
        )


class DatasetWithName(Dataset):
    """Attach the source dataset name without changing the existing datasets."""

    def __init__(
        self,
        dataset: Dataset,
        dataset_name: str,
        has_pixel_masks: bool = True,
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.has_pixel_masks = has_pixel_masks

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["dataset_name"] = self.dataset_name
        if not self.has_pixel_masks:
            sample["has_mask"] = torch.tensor(False, dtype=torch.bool)
        return sample


def configure_lodo_datasets(dataset_names, data_path_entries=None):
    """Register CLI paths and discover class names from full-shot metadata."""
    data_path_entries = data_path_entries or []
    overrides = {}
    for entry in data_path_entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --data_path {entry!r}; expected DATASET=/absolute/path"
            )
        dataset_name, data_path = entry.split("=", 1)
        if not dataset_name or not data_path:
            raise ValueError(
                f"Invalid --data_path {entry!r}; expected DATASET=/absolute/path"
            )
        overrides[dataset_name] = data_path

    DATA_PATH.update(overrides)
    for dataset_name in dataset_names:
        if dataset_name not in DATA_PATH:
            raise ValueError(
                f"No data path configured for {dataset_name!r}. Pass "
                f"--data_path {dataset_name}=/absolute/path"
            )

        meta_path = os.path.join(
            "dataset", "metadata", dataset_name, "full-shot.jsonl"
        )
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Missing metadata for {dataset_name!r}: {meta_path}"
            )

        class_names = []
        with open(meta_path, "r", encoding="utf-8") as metadata_file:
            for line in metadata_file:
                if not line.strip():
                    continue
                class_name = json.loads(line)["class_name"]
                if class_name not in class_names:
                    class_names.append(class_name)
        if not class_names:
            raise ValueError(f"No samples found in {meta_path}")

        configured_classes = CLASS_NAMES.get(dataset_name)
        if configured_classes is not None and set(configured_classes) != set(class_names):
            raise ValueError(
                f"CLASS_NAMES[{dataset_name!r}]={configured_classes} does not match "
                f"metadata classes {class_names}"
            )
        CLASS_NAMES[dataset_name] = class_names
        DOMAINS.setdefault(dataset_name, "Medical")
        real_names = REAL_NAMES.setdefault(dataset_name, {})
        for class_name in class_names:
            real_names.setdefault(class_name, class_name.replace("_", " "))

    return overrides
