"""Shared types and constants for SocialIQA helpers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

SOCIALIQA_DATASET_ID = "socialiqa-archive"
SOCIALIQA_ZIP_URL = (
    "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
)
SOCIALIQA_ZIP_NAME = "socialiqa-train-dev.zip"
CHOICE_FIELDS: Sequence[str] = ("answerA", "answerB", "answerC")


@dataclass(frozen=True)
class SocialIQAManifest:
    dataset: str
    split: str
    seed: int
    sample_size: int | None
    query_count: int
    queries: list[dict[str, object]]


@dataclass(frozen=True)
class SocialIQAFiles:
    data_files: Mapping[str, str]
    label_files: Mapping[str, str]
