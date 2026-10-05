"""Helpers to load and format SocialIQA queries.

Change notes:
- Default SocialIQA loading now uses the public archive (JSONL + labels) instead of the Hugging Face dataset script.
- Local paths (zip or extracted directory) are supported to avoid relying on repo-hosted scripts.
- This is an opinionated default to keep datasets>=4.x working without script execution.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import zipfile

import requests
from datasets import Dataset, load_dataset

from data_attribution.config import resolve_hf_datasets_cache
from data_attribution.data.socialiqa_queries import (
    build_socialiqa_queries,
    socialiqa_manifest,
    socialiqa_prompt,
)
from data_attribution.data.socialiqa_types import (
    CHOICE_FIELDS,
    SOCIALIQA_DATASET_ID,
    SOCIALIQA_ZIP_NAME,
    SOCIALIQA_ZIP_URL,
    SocialIQAFiles,
    SocialIQAManifest,
)


def load_socialiqa(
    *,
    split: str = "validation",
    dataset_path: str | Path = SOCIALIQA_DATASET_ID,
    data_files: Mapping[str, object] | None = None,
    cache_dir: Path | None = None,
) -> Dataset:
    """Load a SocialIQA split from Hugging Face or a local archive."""

    resolved_cache_dir = resolve_hf_datasets_cache(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    if data_files is not None:
        return load_dataset(
            "json",
            split=split,
            data_files=data_files,
            cache_dir=str(resolved_cache_dir),
        )

    resolved_files = _ensure_socialiqa_files(dataset_path, resolved_cache_dir)
    dataset = load_dataset(
        "json",
        split=split,
        data_files=resolved_files.data_files,
        cache_dir=str(resolved_cache_dir),
    )
    return _attach_labels(dataset, resolved_files.label_files.get(split, ""), split)


def _ensure_socialiqa_files(
    dataset_path: str | Path, cache_dir: Path | None
) -> SocialIQAFiles:
    """Download and extract SocialIQA JSONL files for datasets>=4.x."""

    cache_root = resolve_hf_datasets_cache(cache_dir)
    zip_path, extract_root, local_root = _resolve_socialiqa_paths(
        dataset_path, cache_root
    )
    if local_root is not None:
        return _validate_socialiqa_files(
            local_root,
            "SocialIQA JSONL files not found in provided dataset_path directory.",
        )

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    _download_socialiqa_zip(zip_path)
    _extract_socialiqa_zip(zip_path, extract_root)
    return _validate_socialiqa_files(
        extract_root, "SocialIQA JSONL files not found after extraction."
    )


def _resolve_socialiqa_paths(
    dataset_path: str | Path, cache_root: Path
) -> tuple[Path, Path, Path | None]:
    data_root = cache_root / "social_i_qa_raw"
    zip_path = data_root / SOCIALIQA_ZIP_NAME
    extract_root = data_root / "extracted"

    dataset_path = Path(dataset_path)
    if dataset_path.exists():
        if dataset_path.is_dir():
            return zip_path, extract_root, dataset_path
        if dataset_path.is_file() and dataset_path.suffix == ".zip":
            return dataset_path, data_root / "extracted_local", None
        raise ValueError(f"Unsupported SocialIQA dataset_path: {dataset_path}")

    return zip_path, extract_root, None


def _download_socialiqa_zip(zip_path: Path) -> None:
    if zip_path.exists():
        return

    response = requests.get(SOCIALIQA_ZIP_URL, timeout=60, stream=True)
    response.raise_for_status()
    tmp_path = zip_path.with_suffix(".download")
    with tmp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    tmp_path.replace(zip_path)


def _extract_socialiqa_zip(zip_path: Path, extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    train_path = _find_socialiqa_file(extract_root, "train.jsonl")
    dev_path = _find_socialiqa_file(extract_root, "dev.jsonl")
    if train_path and dev_path:
        return

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_root)


def _validate_socialiqa_files(root: Path, error_message: str) -> SocialIQAFiles:
    train_path = _find_socialiqa_file(root, "train.jsonl")
    dev_path = _find_socialiqa_file(root, "dev.jsonl")
    train_labels = _find_socialiqa_file(root, "train-labels.lst")
    dev_labels = _find_socialiqa_file(root, "dev-labels.lst")
    if not train_path or not dev_path:
        raise FileNotFoundError(error_message)

    return SocialIQAFiles(
        data_files={"train": str(train_path), "validation": str(dev_path)},
        label_files={
            "train": str(train_labels) if train_labels else "",
            "validation": str(dev_labels) if dev_labels else "",
        },
    )


def _find_socialiqa_file(root: Path, filename: str) -> Path | None:
    candidates = [
        path
        for path in root.rglob(filename)
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    ]
    return candidates[0] if candidates else None


def _read_labels(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def _attach_labels(dataset: Dataset, labels_path: str, split: str) -> Dataset:
    if not labels_path:
        return dataset
    labels = _read_labels(Path(labels_path))
    if len(labels) != len(dataset):
        raise ValueError(
            f"Label count mismatch for {split}: {len(labels)} vs {len(dataset)}"
        )
    return dataset.add_column("label", labels)


__all__ = [
    "CHOICE_FIELDS",
    "SOCIALIQA_DATASET_ID",
    "SocialIQAManifest",
    "build_socialiqa_queries",
    "load_socialiqa",
    "socialiqa_manifest",
    "socialiqa_prompt",
]
