"""Worker function for parallel pool sampling."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dolma.pool_sample.core import estimate_pool_tokens, extract_manifest_row
from dolma.sample import project_record
from dolma.sharded_writer import ShardedJsonlWriter

logger = logging.getLogger(__name__)

LOG_INTERVAL = 100_000

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string()),
        pa.field("shard_path", pa.string()),
        pa.field("token_count", pa.int64()),
        pa.field("weborganizer_topic", pa.string()),
        pa.field("weborganizer_format", pa.string()),
    ]
)


@dataclass
class WorkerResult:
    worker_id: int
    doc_count: int = 0
    token_total: int = 0
    shard_paths: list[str] = field(default_factory=list)
    manifest_path: str = ""


def _iter_shard_records(shard_path: Path):
    from dolma.remove_white_spaces import iter_jsonl_zst_streaming

    try:
        for line in iter_jsonl_zst_streaming(shard_path):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    except (UnicodeDecodeError, OSError) as exc:
        logger.warning(
            "Skipping corrupt shard %s: %s: %s",
            shard_path.name,
            type(exc).__name__,
            exc,
        )


def process_shards(
    shard_paths: list[Path],
    output_dir: Path,
    token_budget: int,
    worker_id: int,
    lines_per_shard: int = 10_000_000,
    log_every: int = LOG_INTERVAL,
) -> WorkerResult:
    result = WorkerResult(worker_id=worker_id)
    manifest_rows: list[dict[str, object]] = []
    worker_dir = output_dir / f"worker_{worker_id:03d}"

    with ShardedJsonlWriter(worker_dir, lines_per_shard=lines_per_shard) as writer:
        for shard_path in shard_paths:
            if result.token_total >= token_budget:
                break

            shard_name = shard_path.name
            logger.info("Worker %d: processing %s", worker_id, shard_name)

            for record in _iter_shard_records(shard_path):
                projected = project_record(record)
                text = projected.get("text") or ""
                metadata = projected.get("metadata") or {}
                tokens = estimate_pool_tokens(text, metadata)

                writer.write(projected, tokens=tokens)
                result.doc_count += 1
                result.token_total += tokens

                manifest_rows.append(
                    extract_manifest_row(projected, tokens, shard_name)
                )

                if result.doc_count % log_every == 0:
                    logger.info(
                        "Worker %d: %d docs, %.2fB tokens",
                        worker_id,
                        result.doc_count,
                        result.token_total / 1e9,
                    )

                if result.token_total >= token_budget:
                    break

        result.shard_paths = writer.stats.shard_paths

    manifest_path = worker_dir / "manifest.parquet"
    if manifest_rows:
        table = pa.Table.from_pylist(manifest_rows, schema=MANIFEST_SCHEMA)
        pq.write_table(table, manifest_path)
    result.manifest_path = str(manifest_path)

    logger.info(
        "Worker %d finished: %d docs, %.2fB tokens, %d shards",
        worker_id,
        result.doc_count,
        result.token_total / 1e9,
        len(result.shard_paths),
    )
    return result


__all__ = ["WorkerResult", "process_shards"]
