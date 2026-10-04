"""Manifest parsing for materialized unlearning train-text sets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    label: str
    condition: str
    topic_bin: str
    seed: int
    forget_texts: str
    stratum_type: str = ""
    stratum: str = ""
    format: str = ""
    target: str = ""


def read_rows(path: Path) -> list[Row]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        Row(
            label=require(row, "label"),
            condition=require(row, "condition"),
            topic_bin=require(row, "topic_bin"),
            seed=int(require(row, "seed")),
            forget_texts=row.get("forget_texts", "").strip(),
            stratum_type=row.get("stratum_type", "").strip(),
            stratum=row.get("stratum", "").strip(),
            format=row.get("format", "").strip(),
            target=row.get("target", "").strip(),
        )
        for row in rows
    ]


def require(row: dict[str, str], key: str) -> str:
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"manifest row missing {key}: {row}")
    return value


def split_values(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def filter_rows(
    rows: list[Row],
    conditions: set[str],
    labels: set[str],
    offset: int,
    limit: int | None,
) -> list[Row]:
    selected = [
        row
        for row in rows
        if (not conditions or row.condition in conditions)
        and (not labels or row.label in labels)
    ]
    if labels:
        found = {row.label for row in selected}
        missing = labels - found
        if missing:
            raise ValueError(f"labels not found in manifest: {sorted(missing)}")
    selected = selected[offset:]
    return selected[:limit] if limit is not None else selected
