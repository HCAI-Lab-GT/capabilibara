"""Convert social_iqa_labelled.jsonl to Bergson query index format.

Reads the labelled SocialIQA JSONL and produces a JSONL where each row
has ``prompt``, ``completion`` (letter only), ``query_id``, and ``label``
fields expected by ``bergson build --prompt_column prompt
--completion_column completion``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

ANSWER_FIELD_TO_LETTER: dict[str, str] = {
    "answerA": "A",
    "answerB": "B",
    "answerC": "C",
}

logger = logging.getLogger(__name__)


def format_prompt(row: dict[str, str]) -> str:
    context = row.get("context", "").strip()
    question = row.get("question", "").strip()
    lines: list[str] = []
    if context:
        lines.append(context)
    if question:
        lines.append(f"Question: {question}")
    for field, letter in ANSWER_FIELD_TO_LETTER.items():
        choice = row.get(field, "").strip()
        lines.append(f"{letter}) {choice}")
    return "\n".join(lines)


def convert_row(row: dict[str, str], index: int) -> dict[str, str | None]:
    correct = row.get("correct_answer", "")
    letter = ANSWER_FIELD_TO_LETTER.get(correct)
    if letter is None:
        logger.warning(
            "Row %d: unrecognized correct_answer=%r, skipping", index, correct
        )
        return {}
    return {
        "prompt": format_prompt(row),
        "completion": letter,
        "query_id": f"socialiqa-{index}",
        "label": row.get("label"),
    }


def convert(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            converted = convert_row(row, idx)
            if not converted:
                skipped += 1
                continue
            fout.write(json.dumps(converted) + "\n")
            written += 1
    logger.info("Wrote %d queries to %s (skipped %d)", written, output_path, skipped)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert social_iqa_labelled.jsonl to Bergson query format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("queries/social_iqa_labelled.jsonl"),
        help="Path to the labelled SocialIQA JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("queries/social_iqa_bergson.jsonl"),
        help="Output path for Bergson-compatible JSONL.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    convert(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
