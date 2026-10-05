"""Download MMLU sociology from HuggingFace and write Bergson-compatible JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from datasets import load_dataset

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
SUBJECT = "sociology"

logger = logging.getLogger(__name__)


def format_prompt(question: str, choices: list[str]) -> str:
    lines = ["Answer the multiple-choice question.", ""]
    lines.append(f"Question: {question}")
    for idx, choice in enumerate(choices):
        letter = chr(ord("A") + idx)
        lines.append(f"({letter}) {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def make_id(question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()
    return f"mmlu_{digest}"


def convert_split(dataset, split: str, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for row in dataset:
            letter = ANSWER_MAP.get(row["answer"])
            if letter is None:
                continue
            entry = {
                "id": make_id(row["question"]),
                "dataset": "mmlu",
                "group": SUBJECT,
                "subject": SUBJECT,
                "label": None,
                "prompt": format_prompt(row["question"], row["choices"]),
                "completion": letter,
                "source_file": f"mmlu_{SUBJECT}_{split}.jsonl",
            }
            fout.write(json.dumps(entry) + "\n")
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download MMLU sociology and write Bergson query JSONL."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Splits to download (default: test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("queries/mmlu"),
        help="Output directory for JSONL files.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    for split in args.splits:
        logger.info("Loading cais/mmlu %s split=%s", SUBJECT, split)
        ds = load_dataset("cais/mmlu", SUBJECT, split=split)
        output_path = args.output_dir / f"{SUBJECT}_{split}.jsonl"
        count = convert_split(ds, split, output_path)
        logger.info("Wrote %d queries to %s", count, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
