"""Prompt and manifest helpers for SocialIQA."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from collections.abc import Iterable, Mapping

from data_attribution.data.socialiqa_types import (
    CHOICE_FIELDS,
    SOCIALIQA_DATASET_ID,
    SocialIQAManifest,
)

if TYPE_CHECKING:
    from datasets import Dataset


def socialiqa_prompt(example: Mapping[str, object]) -> str:
    """Construct a multiple-choice prompt string from a SocialIQA row."""

    context = str(example.get("context") or "").strip()
    question = str(example.get("question") or "").strip()
    choices = [str(example.get(field) or "").strip() for field in CHOICE_FIELDS]

    lines: list[str] = []
    if context:
        lines.append(context)
    if question:
        lines.append(f"Question: {question}")
    for idx, choice in enumerate(choices):
        label = chr(ord("A") + idx)
        lines.append(f"{label}) {choice}")
    return "\n".join(lines).strip()


def _choice_entries(example: Mapping[str, object]) -> list[dict[str, str]]:
    return [
        {
            "id": chr(ord("A") + idx),
            "text": str(example.get(field) or "").strip(),
        }
        for idx, field in enumerate(CHOICE_FIELDS)
    ]


def _label_from_example(example: Mapping[str, object]) -> str | None:
    label_value = example.get("label") or example.get("correct")
    if label_value is None:
        return None

    if isinstance(label_value, str):
        cleaned = label_value.strip()
        if cleaned.upper() in {"A", "B", "C"}:
            return cleaned.upper()
        if cleaned.isdigit():
            index = int(cleaned) - 1
        else:
            return None
    elif isinstance(label_value, int):
        index = label_value - 1
    else:
        return None

    if 0 <= index < len(CHOICE_FIELDS):
        return chr(ord("A") + index)
    return None


def _completion_from_label(
    example: Mapping[str, object], label: str | None = None
) -> str | None:
    label = label or _label_from_example(example)
    if label is None:
        return None

    index = ord(label) - ord("A")
    if not 0 <= index < len(CHOICE_FIELDS):
        return None

    choice_value = example.get(CHOICE_FIELDS[index])
    if choice_value is None:
        return None

    completion = str(choice_value).strip()
    return completion or None


def _query_id(example: Mapping[str, object], fallback_split: str, index: int) -> str:
    return str(
        example.get("id") or example.get("question_id") or f"{fallback_split}-{index}"
    )


def build_socialiqa_queries(
    dataset: Iterable[Mapping[str, object]],
    *,
    split: str,
    dataset_id: str = SOCIALIQA_DATASET_ID,
) -> list[dict[str, object]]:
    queries: list[dict[str, object]] = []
    for idx, example in enumerate(dataset):
        prompt = socialiqa_prompt(example)
        label = _label_from_example(example)
        queries.append(
            {
                "query_id": _query_id(example, split, idx),
                "prompt": prompt,
                "choices": _choice_entries(example),
                "label": label,
                "completion": _completion_from_label(example, label=label)
                if label is not None
                else None,
                "source": {"dataset": dataset_id, "split": split},
            }
        )
    return queries


def socialiqa_manifest(
    *,
    split: str = "validation",
    dataset_path: str | Path = SOCIALIQA_DATASET_ID,
    data_files: Mapping[str, object] | None = None,
    cache_dir: Path | None = None,
    sample_size: int | None = None,
    seed: int = 0,
    dataset: Dataset | None = None,
) -> SocialIQAManifest:
    """Load SocialIQA examples and emit a query manifest."""

    from datasets import Dataset

    from data_attribution.data.socialiqa import load_socialiqa

    socialiqa_ds = dataset or load_socialiqa(
        split=split,
        dataset_path=dataset_path,
        data_files=data_files,
        cache_dir=cache_dir,
    )
    if not isinstance(socialiqa_ds, Dataset):
        raise TypeError("Expected a datasets.Dataset for SocialIQA manifest.")

    if sample_size is not None:
        socialiqa_ds = socialiqa_ds.shuffle(seed=seed).select(
            range(min(sample_size, len(socialiqa_ds)))
        )

    queries = build_socialiqa_queries(
        socialiqa_ds, split=split, dataset_id=str(dataset_path)
    )

    return SocialIQAManifest(
        dataset=str(dataset_path),
        split=split,
        seed=seed,
        sample_size=sample_size,
        query_count=len(queries),
        queries=queries,
    )


__all__ = [
    "build_socialiqa_queries",
    "socialiqa_manifest",
    "socialiqa_prompt",
]
