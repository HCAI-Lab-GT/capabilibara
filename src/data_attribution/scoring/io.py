from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from collections.abc import Mapping, Sequence


def materialize_ids(ids: list[object] | None, total: int) -> list[object]:
    if ids is None:
        return list(range(total))
    if len(ids) != total:
        raise ValueError(f"Id list length {len(ids)} does not match expected {total}")
    return ids


def load_id_field(index_path: Path, field_name: str) -> list[object] | None:
    data_path = index_path / "data.hf"
    if not data_path.exists():
        return _load_id_field_from_index_config(index_path, field_name)

    if importlib.util.find_spec("datasets") is None:
        return _load_id_field_from_index_config(index_path, field_name)

    datasets = importlib.import_module("datasets")
    try:
        dataset = datasets.load_from_disk(str(data_path))
    except Exception:
        return _load_id_field_from_index_config(index_path, field_name)
    split = next(iter(dataset.values())) if isinstance(dataset, dict) else dataset
    if field_name not in split.column_names:
        return _load_id_field_from_index_config(index_path, field_name)
    return [row[field_name] for row in split]


def load_id_field_from_jsonl(path: Path, field_name: str) -> list[object] | None:
    if not path.exists() or not path.is_file():
        return None
    if path.suffix.lower() != ".jsonl":
        return None

    ids: list[object] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if field_name not in row:
                return None
            ids.append(row[field_name])
    return ids or None


def _load_id_field_from_index_config(
    index_path: Path, field_name: str
) -> list[object] | None:
    cfg_path = index_path / "index_config.json"
    if not cfg_path.exists():
        return None

    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError:
        return None

    data_cfg = cfg.get("data")
    dataset_value = data_cfg.get("dataset") if isinstance(data_cfg, dict) else None
    if not isinstance(dataset_value, str):
        return None

    dataset_path = Path(dataset_value)
    if not dataset_path.is_absolute():
        dataset_path = cfg_path.parent / dataset_path
    return load_id_field_from_jsonl(dataset_path, field_name)


def load_text_lookup(
    index_path: Path, id_field: str, text_field: str
) -> Mapping[object, str] | None:
    data_path = index_path / "data.hf"
    if not data_path.exists():
        return None

    if importlib.util.find_spec("datasets") is None:
        return None

    datasets = importlib.import_module("datasets")
    dataset = datasets.load_from_disk(str(data_path))
    split = next(iter(dataset.values())) if isinstance(dataset, dict) else dataset
    if id_field not in split.column_names or text_field not in split.column_names:
        return None

    id_column = split[id_field]
    text_column = split[text_field]
    return {id_column[i]: text_column[i] for i in range(len(split))}


def write_json(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2)
        stream.write("\n")


def resolve_query_ids(
    query_index: Path,
    query_id_field: str,
    query_manifest: Path | None,
    total_queries: int,
) -> list[object]:
    if query_manifest is not None:
        manifest_ids = load_id_field_from_jsonl(query_manifest, query_id_field)
        if manifest_ids is None:
            raise ValueError(
                "Failed to load "
                f"'{query_id_field}' from query manifest: {query_manifest}"
            )
        return materialize_ids(manifest_ids, total_queries)

    return materialize_ids(load_id_field(query_index, query_id_field), total_queries)


def build_record(
    query_id: object,
    doc_ids: list[object],
    scores: list[float],
    gradient_keys: Sequence[str],
    text_lookup: Mapping[object, str] | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "query_id": query_id,
        "top_k_indices": doc_ids,
        "scores": scores,
        "gradient_keys": list(gradient_keys),
    }
    if text_lookup is not None:
        record["documents"] = [
            {"doc_id": doc_id, "text": text_lookup.get(doc_id)} for doc_id in doc_ids
        ]
    return record
