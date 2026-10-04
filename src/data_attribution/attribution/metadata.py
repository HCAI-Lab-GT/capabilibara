"""Load metadata for attribution joins."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

import pyarrow.parquet as pq


def _load_metadata_parquet(
    path: Path, doc_id_field: str
) -> dict[object, Mapping[str, object]]:
    table = pq.read_table(path)
    lookup: dict[object, Mapping[str, object]] = {}
    for row in table.to_pylist():
        doc_id = row.get(doc_id_field)
        if doc_id is None:
            continue
        lookup[doc_id] = row
    return lookup


def _load_metadata_jsonl(
    path: Path, doc_id_field: str
) -> dict[object, Mapping[str, object]]:
    lookup: dict[object, Mapping[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            doc_id = record.get(doc_id_field)
            if doc_id is None:
                continue
            lookup[doc_id] = record
    return lookup


def load_metadata(
    path: Path | None, doc_id_field: str
) -> dict[object, Mapping[str, object]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Metadata path not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _load_metadata_parquet(path, doc_id_field)
    if suffix == ".jsonl":
        return _load_metadata_jsonl(path, doc_id_field)
    raise ValueError(f"Unsupported metadata format: {path.suffix}")


__all__ = ["load_metadata"]
