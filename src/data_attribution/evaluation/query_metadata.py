"""Query metadata sidecar: benchmark, subject, and reasoning type labels.

Loads evaluation data from the HF dataset and produces a metadata table
joinable against attribution results by query_id.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from data_attribution.evaluation.socialiqa_classifier import (
    classify_socialiqa_reasoning_type,
)

logger = logging.getLogger(__name__)

OLMES_HF_REPO = "HCAI-Lab/olmes-eval-olmo3-7b-base"
VALID_SUBSETS = (
    "gsm8k",
    "mmlu_social_science",
    "mmlu_stem",
    "socialiqa",
    "arc_easy",
    "arc_challenge",
    "bbh_snarks",
    "bbh_causal_judgement",
    "bbh_sports_understanding",
)
_MMLU_SUBSETS = frozenset({"mmlu_social_science", "mmlu_stem"})


def _extract_subject(row: dict[str, Any], subset: str) -> str | None:
    if subset not in _MMLU_SUBSETS:
        return None
    task_name = row.get("task_name", "")
    if task_name.startswith("mmlu_"):
        return task_name[len("mmlu_") :]
    return task_name or None


def _extract_reasoning_type(row: dict[str, Any], subset: str) -> str | None:
    if subset != "socialiqa":
        return None
    query_text = row.get("query_text", "")
    question_match = re.search(r"Question:\s*(.+?)(?:\n|$)", query_text)
    full_text = question_match.group(1).strip() if question_match else query_text
    sentences = re.split(r"(?<=[.!?])\s+", full_text.strip())
    context = " ".join(sentences[:-1]).strip() if len(sentences) > 1 else None
    return classify_socialiqa_reasoning_type(
        full_text, query_id=row.get("query_id"), context=context
    )


def build_query_metadata(
    *,
    hf_repo: str = OLMES_HF_REPO,
    subsets: tuple[str, ...] = VALID_SUBSETS,
) -> list[dict[str, Any]]:
    """Load evaluation data from HF and build a metadata record per query."""
    records: list[dict[str, Any]] = []
    for subset in subsets:
        logger.info("Loading %s/%s", hf_repo, subset)
        dataset = load_dataset(hf_repo, name=subset, split="train")
        for row in dataset:
            records.append(
                {
                    "query_id": row["query_id"],
                    "benchmark": subset,
                    "subject": _extract_subject(row, subset),
                    "reasoning_type": _extract_reasoning_type(row, subset),
                    "is_correct": row["is_correct"],
                    "predicted_answer": row["predicted_answer"],
                    "correct_answer": row["correct_answer"],
                }
            )
        logger.info("Loaded %d rows from %s", len(dataset), subset)
    return records


def write_query_metadata_jsonl(
    output_path: Path,
    *,
    hf_repo: str = OLMES_HF_REPO,
    subsets: tuple[str, ...] = VALID_SUBSETS,
) -> int:
    """Write the full query metadata sidecar as JSONL."""
    records = build_query_metadata(hf_repo=hf_repo, subsets=subsets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    logger.info("Wrote %d query metadata records to %s", len(records), output_path)
    return len(records)
