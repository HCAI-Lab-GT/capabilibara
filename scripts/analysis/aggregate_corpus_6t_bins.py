"""Aggregate the SOC-95 corpus manifest to per-(topic, format) bin composition.

Reads every per-shard parquet under the manifest root, groups by
``weborganizer_topic`` x ``weborganizer_format``, and emits one row per bin
with the document count and summed token mass. Both ``token_count`` and
``estimated_token_count`` are summed so downstream callers can pick the column
they need.

This is the validated recipe behind ``artifacts/corpus_6t_bin_composition.csv``.
Run it against the complete shard set to refresh that artifact.
"""

from __future__ import annotations

import argparse
import glob
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("aggregate_corpus_6t_bins")

READ_COLUMNS = [
    "weborganizer_topic",
    "weborganizer_format",
    "token_count",
    "estimated_token_count",
]


def _aggregate_one(path: str) -> list[tuple[object, object, int, int, int]]:
    table = pq.read_table(path, columns=READ_COLUMNS)
    ones = pa.array([1] * table.num_rows, type=pa.int64())
    table = table.append_column("doc_count", ones)
    grouped = table.group_by(
        ["weborganizer_topic", "weborganizer_format"]
    ).aggregate(
        [
            ("doc_count", "sum"),
            ("token_count", "sum"),
            ("estimated_token_count", "sum"),
        ]
    )
    topics = grouped.column("weborganizer_topic").to_pylist()
    formats = grouped.column("weborganizer_format").to_pylist()
    docs = grouped.column("doc_count_sum").to_pylist()
    toks = grouped.column("token_count_sum").to_pylist()
    ests = grouped.column("estimated_token_count_sum").to_pylist()
    return [
        (topics[i], formats[i], int(docs[i] or 0), int(toks[i] or 0), int(ests[i] or 0))
        for i in range(grouped.num_rows)
    ]


def aggregate(manifest_root: Path, workers: int) -> dict[tuple[object, object], list[int]]:
    files = sorted(glob.glob(str(manifest_root / "**" / "*.parquet"), recursive=True))
    logger.info("found %d parquet shards under %s", len(files), manifest_root)
    if not files:
        raise SystemExit(f"no parquet files under {manifest_root}")

    acc: dict[tuple[object, object], list[int]] = defaultdict(lambda: [0, 0, 0])
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_aggregate_one, f): f for f in files}
        for fut in as_completed(futures):
            for topic, fmt, docs, toks, ests in fut.result():
                row = acc[(topic, fmt)]
                row[0] += docs
                row[1] += toks
                row[2] += ests
            done += 1
            if done % 5000 == 0:
                logger.info("aggregated %d/%d shards", done, len(files))
    logger.info("aggregated %d/%d shards (complete)", done, len(files))
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    acc = aggregate(args.manifest_root, args.workers)

    import csv

    rows = sorted(
        ((t, f, v[0], v[1], v[2]) for (t, f), v in acc.items()),
        key=lambda r: (-(r[3]), str(r[0]), str(r[1])),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["topic_label", "format_label", "doc_count", "token_count_sum", "estimated_token_count_sum"])
        for t, f, dc, tc, ec in rows:
            writer.writerow(["" if t is None else t, "" if f is None else f, dc, tc, ec])

    total_docs = sum(v[0] for v in acc.values())
    total_tok = sum(v[1] for v in acc.values())
    total_est = sum(v[2] for v in acc.values())
    named = {k: v for k, v in acc.items() if k[0] is not None and k[1] is not None}
    fashion = acc.get(("fashion_and_beauty", "documentation"))
    logger.info("TOTAL docs=%d token_count=%d est_token=%d", total_docs, total_tok, total_est)
    logger.info("named-bin count=%d named docs=%d", len(named), sum(v[0] for v in named.values()))
    logger.info("fashion_and_beauty x documentation = %s", fashion)


if __name__ == "__main__":
    main()
