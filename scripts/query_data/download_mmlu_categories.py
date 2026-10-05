"""Download MMLU subjects by superset category from cais/mmlu and write Bergson-compatible JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from datasets import load_dataset

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

SOCIAL_SCIENCE_SUBJECTS = [
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
]

# Source of truth: src/data_attribution/evaluation/task_suites.py (OlmoBaseEval STEM suite)
STEM_SUBJECTS = [
    "abstract_algebra",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
]

CATEGORIES = {
    "social_science": SOCIAL_SCIENCE_SUBJECTS,
    "stem": STEM_SUBJECTS,
}

logger = logging.getLogger(__name__)


def format_prompt(question: str, choices: list[str]) -> str:
    lines = ["Answer the multiple-choice question.", ""]
    lines.append(f"Question: {question}")
    for idx, choice in enumerate(choices):
        letter = chr(ord("A") + idx)
        lines.append(f"({letter}) {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def make_id(subject: str, question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()
    return f"mmlu_{digest}"


def convert_split(
    dataset, subject: str, category: str, split: str, output_path: Path
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for row in dataset:
            letter = ANSWER_MAP.get(row["answer"])
            if letter is None:
                continue
            entry = {
                "id": make_id(subject, row["question"]),
                "dataset": "mmlu",
                "group": category,
                "subject": subject,
                "label": None,
                "prompt": format_prompt(row["question"], row["choices"]),
                "completion": letter,
                "source_file": f"mmlu_{subject}_{split}.jsonl",
            }
            fout.write(json.dumps(entry) + "\n")
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download MMLU subjects by superset category and write Bergson query JSONL."
    )
    parser.add_argument(
        "categories",
        nargs="+",
        choices=list(CATEGORIES.keys()),
        help="Superset categories to download.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "validation", "dev"],
        help="Splits to download (default: test validation dev).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("query_data"),
        help="Root output directory (default: query_data).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    for category in args.categories:
        subjects = CATEGORIES[category]
        cat_dir = args.output_dir / f"mmlu_{category}"
        logger.info(
            "Category: %s (%d subjects) -> %s", category, len(subjects), cat_dir
        )

        for subject in subjects:
            for split in args.splits:
                logger.info("Loading cais/mmlu %s split=%s", subject, split)
                try:
                    ds = load_dataset("cais/mmlu", subject, split=split)
                except ValueError:
                    logger.warning(
                        "Split %s not found for %s, skipping.", split, subject
                    )
                    continue
                output_path = cat_dir / f"{subject}_{split}.jsonl"
                count = convert_split(ds, subject, category, split, output_path)
                logger.info("Wrote %d queries to %s", count, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
