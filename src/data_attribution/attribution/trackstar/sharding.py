"""Shared data utilities for the TrackStar attribution pipeline.

Provides JSONL splitting, zstandard decompression, shard discovery,
and path helpers for the reduce/score/aggregate workflow.
"""

import io
import json
import logging
from pathlib import Path

import zstandard as zstd

log = logging.getLogger(__name__)


def _open_lines(source: Path):
    if source.name.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        with open(source, "rb") as raw:
            with dctx.stream_reader(raw, read_across_frames=True) as reader:
                yield from io.TextIOWrapper(reader, encoding="utf-8")
    else:
        with open(source, encoding="utf-8") as f:
            yield from f


def split_jsonl(source: Path, parallelism: int, shard_dir: Path) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = [shard_dir / f"shard_{i:04d}.jsonl" for i in range(parallelism)]
    handles = [open(p, "w", encoding="utf-8") for p in shard_paths]
    counts = [0] * parallelism

    try:
        with open(source, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                shard_idx = i % parallelism
                handles[shard_idx].write(line if line.endswith("\n") else line + "\n")
                counts[shard_idx] += 1
    finally:
        for h in handles:
            h.close()

    total = sum(counts)
    if total == 0:
        log.error("Empty JSONL: %s", source)
        raise SystemExit(1)

    non_empty = []
    for path, count in zip(shard_paths, counts):
        if count > 0:
            non_empty.append(path)
            log.info("Wrote %s (%d lines)", path.name, count)
        else:
            path.unlink(missing_ok=True)

    log.info(
        "Split %d lines into %d shards from %s", total, len(non_empty), source.name
    )
    return non_empty


KEEP_FIELDS = {"id", "text"}


def _strip_fields(line: str) -> str:
    try:
        doc = json.loads(line)
        stripped = {k: v for k, v in doc.items() if k in KEEP_FIELDS}
        return json.dumps(stripped, ensure_ascii=False) + "\n"
    except json.JSONDecodeError:
        return line


def split_multi_jsonl(
    sources: list[Path], parallelism: int, shard_dir: Path
) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = [shard_dir / f"shard_{i:04d}.jsonl" for i in range(parallelism)]
    handles = [open(p, "w", encoding="utf-8") for p in shard_paths]
    counts = [0] * parallelism

    try:
        i = 0
        for source in sources:
            for line in _open_lines(source):
                if not line.strip():
                    continue
                cleaned = _strip_fields(line)
                shard_idx = i % parallelism
                handles[shard_idx].write(cleaned)
                counts[shard_idx] += 1
                i += 1
    finally:
        for h in handles:
            h.close()

    total = sum(counts)
    if total == 0:
        log.error("No lines found across %d source files", len(sources))
        raise SystemExit(1)

    non_empty = []
    for path, count in zip(shard_paths, counts):
        if count > 0:
            non_empty.append(path)
        else:
            path.unlink(missing_ok=True)

    log.info(
        "Consolidated %d lines from %d files into %d shards in %s",
        total,
        len(sources),
        len(non_empty),
        shard_dir,
    )
    return non_empty


def discover_queries(reduce_dir: Path) -> list[Path]:
    if not reduce_dir.exists():
        log.error("Reduce directory does not exist: %s", reduce_dir)
        raise SystemExit(1)

    queries = sorted(
        p for p in reduce_dir.iterdir() if p.is_dir() and p.name.startswith("queries_")
    )
    if not queries:
        log.error("No query directories found in %s", reduce_dir)
        raise SystemExit(1)

    log.info("Found %d query indexes in %s", len(queries), reduce_dir)
    return queries


def variant_run_dir(base_dir: Path, variant: str, run_id: str) -> Path:
    return base_dir / variant / run_id


def score_output_dir(
    score_run_dir: Path,
    query_name: str,
    source_name: str,
    shard_name: str,
) -> Path:
    return score_run_dir / query_name / source_name / shard_name


def result_output_path(
    results_dir: Path,
    variant: str,
    run_id: str,
    query_name: str,
    top_k: int,
) -> Path:
    return results_dir / variant / run_id / f"{query_name}_top{top_k}.jsonl"


def discover_score_queries(score_dir: Path) -> dict[str, list[Path]]:
    if not score_dir.exists():
        log.error("Score directory does not exist: %s", score_dir)
        raise SystemExit(1)

    query_shards: dict[str, list[Path]] = {}
    for query_dir in sorted(score_dir.iterdir()):
        if not query_dir.is_dir():
            continue
        if not query_dir.name.startswith("queries_"):
            log.warning("Skipping unexpected query directory: %s", query_dir.name)
            continue

        shard_dirs: list[Path] = []
        for source_dir in sorted(query_dir.iterdir()):
            if not source_dir.is_dir():
                log.warning(
                    "Skipping unexpected source entry in %s: %s",
                    query_dir.name,
                    source_dir.name,
                )
                continue
            for shard_dir in sorted(source_dir.iterdir()):
                if not shard_dir.is_dir():
                    log.warning(
                        "Skipping unexpected shard entry in %s/%s: %s",
                        query_dir.name,
                        source_dir.name,
                        shard_dir.name,
                    )
                    continue
                shard_dirs.append(shard_dir)

        if shard_dirs:
            query_shards[query_dir.name] = shard_dirs

    if not query_shards:
        log.error("No score directories found in %s", score_dir)
        raise SystemExit(1)

    log.info("Found %d scored queries in %s", len(query_shards), score_dir)
    return query_shards
