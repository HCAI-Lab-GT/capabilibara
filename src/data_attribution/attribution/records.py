"""Build and persist attribution records."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from data_attribution.attribution.types import AttributionRunConfig


def _metadata_payload(
    doc_id: object,
    metadata_lookup: Mapping[object, Mapping[str, object]],
    join_keys: Sequence[str],
) -> Mapping[str, object] | None:
    metadata = metadata_lookup.get(doc_id)
    if metadata is None:
        return None
    if not join_keys:
        return dict(metadata)
    return {key: metadata.get(key) for key in join_keys}


def _materialize_ids(ids: list[object] | None, total: int) -> list[object]:
    if ids is None:
        return list(range(total))
    if len(ids) != total:
        raise ValueError(f"Id list length {len(ids)} does not match expected {total}")
    return ids


def _records_from_scores(
    scores: torch.Tensor,
    *,
    config: AttributionRunConfig,
    metadata_lookup: Mapping[object, Mapping[str, object]],
    training_ids: list[object],
    query_ids: list[object],
) -> list[dict[str, object]]:
    top_k = min(config.top_k, scores.shape[0])
    timestamp = config.started_at.isoformat()
    records: list[dict[str, object]] = []

    for query_idx, query_id in enumerate(query_ids):
        column = scores[:, query_idx]
        values, indices = torch.topk(column, k=top_k)
        for rank, (score, doc_pos) in enumerate(
            zip(values.tolist(), indices.tolist()), start=1
        ):
            doc_id = training_ids[doc_pos]
            payload: dict[str, object] = {
                "run_id": config.run_id,
                "timestamp": timestamp,
                "query_id": query_id,
                "doc_id": doc_id,
                "influence_score": float(score),
                "rank": rank,
            }
            metadata = _metadata_payload(
                doc_id, metadata_lookup, config.metadata_join_keys
            )
            if metadata is not None:
                payload["metadata"] = metadata
            records.append(payload)
    return records


def _write_jsonl(records: Iterable[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            json.dump(record, stream)
            stream.write("\n")


def _write_parquet(records: Iterable[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(records))
    pq.write_table(table, path)


def persist_outputs(
    records: Iterable[Mapping[str, object]],
    config: AttributionRunConfig,
) -> Path:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output_format == "jsonl":
        output_path = output_dir / "attributions.jsonl"
        _write_jsonl(records, output_path)
    elif config.output_format == "parquet":
        output_path = output_dir / "attributions.parquet"
        _write_parquet(records, output_path)
    else:
        raise ValueError(f"Unsupported output format: {config.output_format}")

    with (output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config.to_serializable(), stream, indent=2)
        stream.write("\n")
    return output_path


__all__ = ["persist_outputs"]
