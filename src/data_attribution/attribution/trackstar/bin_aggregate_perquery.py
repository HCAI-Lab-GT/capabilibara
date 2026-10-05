"""Per-query bin aggregation for cross-benchmark comparability.

Keeps the query dimension during aggregation: for each bin, computes
per-query mean influence across docs, then reports median/IQR across
queries. The median is robust to query count differences, making
benchmarks with different numbers of queries directly comparable.
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_attribution.attribution.trackstar.bin_aggregate import (
    BENCHMARKS,
    load_manifest_bin_map,
    load_shard_doc_uuids,
)

log = logging.getLogger(__name__)


@dataclass
class PerQueryBinAccumulator:
    query_score_sums: np.ndarray
    query_abs_score_sums: np.ndarray
    doc_count: int = 0


def _new_accumulator(n_queries: int) -> PerQueryBinAccumulator:
    return PerQueryBinAccumulator(
        query_score_sums=np.zeros(n_queries, dtype=np.float64),
        query_abs_score_sums=np.zeros(n_queries, dtype=np.float64),
    )


def aggregate_shard_perquery(
    scores: np.ndarray,
    doc_uuids: list[str],
    bin_map: dict[str, tuple[str, str]],
    accumulators: dict[tuple[str, str], PerQueryBinAccumulator],
    n_queries: int,
) -> int:
    bin_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    missed = 0
    for i, uuid in enumerate(doc_uuids):
        bin_key = bin_map.get(uuid)
        if bin_key is None:
            missed += 1
            continue
        bin_indices[bin_key].append(i)

    for bin_key, indices in bin_indices.items():
        idx = np.array(indices)
        bin_scores = scores[idx, :]
        acc = accumulators.setdefault(bin_key, _new_accumulator(n_queries))
        acc.query_score_sums += bin_scores.sum(axis=0).astype(np.float64)
        acc.query_abs_score_sums += np.abs(bin_scores).sum(axis=0).astype(np.float64)
        acc.doc_count += len(indices)

    return missed


def finalize_bins_perquery(
    accumulators: dict[tuple[str, str], PerQueryBinAccumulator],
) -> pd.DataFrame:
    rows = []
    for (topic, fmt), acc in sorted(accumulators.items()):
        if acc.doc_count == 0:
            continue
        pq_means = acc.query_score_sums / acc.doc_count
        pq_abs_means = acc.query_abs_score_sums / acc.doc_count
        rows.append(
            {
                "topic_label": topic,
                "format_label": fmt,
                "median_influence": float(np.median(pq_means)),
                "mean_influence": float(np.mean(pq_means)),
                "p25_influence": float(np.percentile(pq_means, 25)),
                "p75_influence": float(np.percentile(pq_means, 75)),
                "std_influence": float(np.std(pq_means)),
                "median_abs_influence": float(np.median(pq_abs_means)),
                "mean_abs_influence": float(np.mean(pq_abs_means)),
                "doc_count": acc.doc_count,
            }
        )
    return pd.DataFrame(rows)


def aggregate_benchmark_perquery(
    benchmark_dir: Path,
    shard_dir: Path,
    bin_map: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    query_ids_path = benchmark_dir / "query_ids.json"
    with open(query_ids_path) as f:
        n_queries = len(json.load(f))
    log.info("Benchmark has %d queries", n_queries)

    accumulators: dict[tuple[str, str], PerQueryBinAccumulator] = {}
    total_docs = 0
    total_missed = 0

    shard_files = sorted(benchmark_dir.glob("shard_*.npy"))
    log.info("Found %d shard files in %s", len(shard_files), benchmark_dir)

    for npy_path in shard_files:
        shard_name = npy_path.stem
        jsonl_path = shard_dir / f"{shard_name}.jsonl"
        if not jsonl_path.exists():
            log.warning("Source JSONL missing: %s", jsonl_path)
            continue

        scores = np.load(npy_path, mmap_mode="r")
        doc_uuids = load_shard_doc_uuids(jsonl_path)

        if len(scores) != len(doc_uuids):
            log.error(
                "Length mismatch %s: %d scores vs %d docs",
                shard_name,
                len(scores),
                len(doc_uuids),
            )
            continue

        missed = aggregate_shard_perquery(
            scores, doc_uuids, bin_map, accumulators, n_queries
        )
        total_docs += len(doc_uuids)
        total_missed += missed
        log.info("%s: %d docs, %d missed", shard_name, len(doc_uuids), missed)
        del scores

    log.info(
        "Benchmark total: %d docs, %d missed (%.2f%%)",
        total_docs,
        total_missed,
        100.0 * total_missed / max(total_docs, 1),
    )
    return finalize_bins_perquery(accumulators)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Per-query bin aggregation for cross-benchmark comparability"
    )
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading manifest from %s", args.manifest)
    bin_map = load_manifest_bin_map(args.manifest)
    log.info("Manifest loaded: %d docs with bin labels", len(bin_map))

    for benchmark in args.benchmarks:
        benchmark_dir = args.scores_dir / benchmark
        if not benchmark_dir.exists():
            log.warning("Benchmark directory not found: %s", benchmark_dir)
            continue

        log.info("Processing %s", benchmark)
        df = aggregate_benchmark_perquery(benchmark_dir, args.shard_dir, bin_map)
        out_path = args.output_dir / f"{benchmark}_bin_scores_perquery.csv"
        df.to_csv(out_path, index=False)
        log.info("Wrote %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
