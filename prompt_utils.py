"""Reproducible prompt-source utilities for template and offline LLM prompts."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path


DEFAULT_LLM_PROMPT_PATH = Path(__file__).parent / "dataset" / "llm_prompts.json"
PROMPT_SOURCES = ("auto", "llm", "template")


def _resolved_path(prompt_path):
    return Path(prompt_path or DEFAULT_LLM_PROMPT_PATH).expanduser().resolve()


@lru_cache(maxsize=8)
def _load_prompt_bank_cached(resolved_path):
    path = Path(resolved_path)
    if not path.is_file():
        raise FileNotFoundError(f"LLM prompt bank not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        bank = json.load(file)
    if not isinstance(bank, dict) or not isinstance(bank.get("datasets"), dict):
        raise ValueError(f"invalid LLM prompt bank schema: {path}")
    if bank.get("schema_version") != 1:
        raise ValueError(
            f"unsupported LLM prompt bank schema version in {path}: "
            f"{bank.get('schema_version')!r}"
        )
    return bank


def load_prompt_bank(prompt_path=None):
    return _load_prompt_bank_cached(str(_resolved_path(prompt_path)))


def prompt_bank_sha256(prompt_path=None):
    path = _resolved_path(prompt_path)
    if not path.is_file():
        raise FileNotFoundError(f"LLM prompt bank not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_prompt_entry(bank, dataset_name):
    datasets = bank["datasets"]
    if dataset_name in datasets:
        return datasets[dataset_name]
    if dataset_name.lower().startswith("colon") and "Colon" in datasets:
        return datasets["Colon"]
    base_name = dataset_name.split("-", maxsplit=1)[0]
    return datasets.get(base_name)


def has_llm_prompts(dataset_name, prompt_path=None):
    bank = load_prompt_bank(prompt_path)
    return _dataset_prompt_entry(bank, dataset_name) is not None


def resolve_prompt_source(prompt_source, dataset_name, prompt_path=None):
    if prompt_source not in PROMPT_SOURCES:
        raise ValueError(
            f"unknown prompt source {prompt_source!r}; choose from {PROMPT_SOURCES}"
        )
    if prompt_source == "auto":
        return "llm" if has_llm_prompts(dataset_name, prompt_path) else "template"
    if prompt_source == "llm" and not has_llm_prompts(dataset_name, prompt_path):
        raise KeyError(f"no LLM prompts registered for dataset {dataset_name!r}")
    return prompt_source


def get_llm_state_prompts(dataset_name, class_name, real_name, prompt_path=None):
    bank = load_prompt_bank(prompt_path)
    entry = _dataset_prompt_entry(bank, dataset_name)
    if entry is None:
        raise KeyError(f"no LLM prompts registered for dataset {dataset_name!r}")
    if not isinstance(entry, dict):
        raise ValueError(f"LLM prompt entry {dataset_name!r} must be an object")
    if isinstance(entry.get("classes"), dict):
        entry = entry["classes"].get(class_name, entry.get("default"))
        if entry is None:
            raise KeyError(
                f"no LLM prompts registered for {dataset_name!r}/{class_name!r}"
            )

    result = []
    for state in ("normal", "abnormal"):
        prompts = entry.get(state)
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(
                f"LLM prompt entry {dataset_name!r}/{state!r} must be a non-empty list"
            )
        formatted = []
        for prompt in prompts:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"LLM prompt entry {dataset_name!r}/{state!r} contains invalid text"
                )
            formatted.append(
                prompt.strip().format(
                    dataset=dataset_name,
                    class_name=class_name,
                    real_name=real_name,
                )
            )
        if len(set(formatted)) != len(formatted):
            raise ValueError(
                f"LLM prompt entry {dataset_name!r}/{state!r} contains duplicates"
            )
        result.append(formatted)
    if set(result[0]) & set(result[1]):
        raise ValueError(
            f"LLM prompt entry {dataset_name!r} reuses text across normal/abnormal"
        )
    return result


def prompt_checkpoint_metadata(prompt_source, dataset_name, prompt_path=None):
    resolved_source = resolve_prompt_source(prompt_source, dataset_name, prompt_path)
    metadata = {"prompt_source": resolved_source}
    if resolved_source == "llm":
        bank = load_prompt_bank(prompt_path)
        metadata.update(
            {
                "llm_prompt_bank_sha256": prompt_bank_sha256(prompt_path),
                "llm_prompt_bank_schema_version": bank.get("schema_version"),
            }
        )
    return metadata


def validate_checkpoint_prompt_metadata(
    checkpoint,
    prompt_source,
    dataset_name,
    prompt_path=None,
    description="text checkpoint",
):
    expected = prompt_checkpoint_metadata(prompt_source, dataset_name, prompt_path)
    # Checkpoints created before prompt metadata existed used the fixed
    # hand-written template prompts.
    actual_source = checkpoint.get("prompt_source", "template")
    if actual_source != expected["prompt_source"]:
        raise ValueError(
            f"{description} prompt source mismatch: checkpoint={actual_source}, "
            f"requested={expected['prompt_source']}"
        )
    if actual_source == "llm":
        actual_hash = checkpoint.get("llm_prompt_bank_sha256")
        if actual_hash is None:
            raise ValueError(f"{description} has no LLM prompt-bank hash")
        if actual_hash != expected["llm_prompt_bank_sha256"]:
            raise ValueError(
                f"{description} LLM prompt bank differs from training: "
                f"checkpoint={actual_hash}, "
                f"current={expected['llm_prompt_bank_sha256']}"
            )
    return expected
