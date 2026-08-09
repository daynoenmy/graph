import json
import os

from torch.utils.data import Dataset

from dataset.constants import CLASS_NAMES, DATA_PATH, DOMAINS, REAL_NAMES


DEFAULT_LODO_DATASETS = ["Chest", "Liver", "Brain", "OCT2017", "RESC", "HIS"]


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
            sample["has_mask"] = False
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
