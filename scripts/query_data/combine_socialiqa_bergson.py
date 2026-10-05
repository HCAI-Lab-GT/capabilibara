"""Combine SocialIQA v1.4 splits into a single bergson-format JSONL file."""

import json
from pathlib import Path

SOURCE_DIR = Path("query_data/social_iqa")
OUTPUT_PATH = Path("query_data/social_iqa_bergson_combined.jsonl")

SPLITS = [
    ("trn", "train"),
    ("dev", "dev"),
    ("tst", "test"),
]

QUESTION_LABEL_MAP = [
    ("want to do next", "wants"),
    ("need to do before", "needs"),
    ("How would you describe", "descriptions"),
    ("feel", "reactions"),
    ("Why did", "motivations"),
    ("What will happen", "effects"),
]


def classify_question(question: str) -> str:
    for pattern, label in QUESTION_LABEL_MAP:
        if pattern.lower() in question.lower():
            return label
    return "other"


def format_prompt(row: dict) -> str:
    lines = []
    context = row.get("context", "").strip()
    question = row.get("question", "").strip()
    if context:
        lines.append(context)
    if question:
        lines.append(f"Question: {question}")
    for key, letter in [("answerA", "A"), ("answerB", "B"), ("answerC", "C")]:
        choice = row.get(key, "").strip()
        lines.append(f"{letter}) {choice}")
    return "\n".join(lines)


def main() -> None:
    records = []
    global_idx = 0

    for file_suffix, split_name in SPLITS:
        path = SOURCE_DIR / f"socialIQa_v1.4_{file_suffix}.jsonl"
        if not path.exists():
            print(f"skipping missing file: {path}")
            continue

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = format_prompt(row)
                question = row.get("question", "")
                label = classify_question(question)
                correct = row.get("correct", "").strip()
                completion = correct if correct in ("A", "B", "C") else None

                record = {
                    "prompt": prompt,
                    "completion": completion,
                    "query_id": f"socialiqa-{global_idx}",
                    "label": label,
                    "split": split_name,
                }
                records.append(record)
                global_idx += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} records to {OUTPUT_PATH}")

    from collections import Counter

    split_counts = Counter(r["split"] for r in records)
    label_counts = Counter(r["label"] for r in records)
    print(f"splits: {dict(split_counts)}")
    print(f"labels: {dict(label_counts)}")


if __name__ == "__main__":
    main()
