"""Collect bergson scores across shards, rank documents, and write top results.

For each query, aggregates scores from all shard directories, ranks
documents by score descending, and writes the top-k results as JSONL.

Usage:
    data-attribution-trackstar-aggregate \
        --scores-dir /path/to/scores --results-dir /path/to/results \
        --variant base --run-id <run_id> --top-k 100
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from bergson.data import load_scores
from datasets import load_from_disk

from data_attribution.attribution.trackstar.sharding import (
    discover_score_queries,
    result_output_path,
    variant_run_dir,
)

log = logging.getLogger(__name__)


def load_shard_scores(score_dir: Path) -> tuple[object, np.ndarray]:
    scores_obj = load_scores(score_dir)
    values = np.asarray(scores_obj.mmap["score_0"], dtype=np.float32)
    ds = load_from_disk(str(score_dir / "data.hf"))
    return ds, values


def collect_query_results(
    score_dirs: list[Path],
    top_k: int,
) -> list[dict]:
    all_datasets = []
    all_scores = []

    for score_dir in sorted(score_dirs):
        log.info("Loading %s", score_dir.name)
        ds, scores = load_shard_scores(score_dir)
        all_datasets.append(ds)
        all_scores.append(scores)

    combined_scores = np.concatenate(all_scores)
    log.info("Total documents across shards: %d", len(combined_scores))

    k = min(top_k, len(combined_scores))
    if k == 0:
        top_indices = np.array([], dtype=int)
    else:
        partial_indices = np.argpartition(combined_scores, -k)[-k:]
        top_indices = partial_indices[
            np.argsort(combined_scores[partial_indices])[::-1]
        ]

    shard_lengths = [len(scores) for scores in all_scores]
    shard_ends = np.cumsum(shard_lengths)

    results = []
    for rank, idx in enumerate(top_indices, 1):
        shard_idx = int(np.searchsorted(shard_ends, idx, side="right"))
        prev_end = int(shard_ends[shard_idx - 1]) if shard_idx > 0 else 0
        local_idx = int(idx - prev_end)

        row = dict(all_datasets[shard_idx][local_idx])
        row["rank"] = rank
        row["score"] = float(combined_scores[idx])
        results.append(row)

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Rank top documents from scored shards"
    )
    parser.add_argument(
        "--variant",
        choices=[
            "base",
            "base_32b",
            "base_olmobaseeval",
            "dclm_base",
            "instruct_base",
            "instruct_cot",
            "comma_1t",
            "comma_2t",
        ],
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=Path("scores"),
        help="Base directory for score outputs (default: scores/)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base directory for result outputs (default: results/)",
    )
    args = parser.parse_args()

    scores_dir = variant_run_dir(args.scores_dir, args.variant, args.run_id)
    results_dir = args.results_dir / args.variant / args.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    query_shards = discover_score_queries(scores_dir)
    log.info("Found %d queries in %s", len(query_shards), scores_dir)

    for query_name, shard_dirs in sorted(query_shards.items()):
        log.info("Processing %s (%d shards)", query_name, len(shard_dirs))
        results = collect_query_results(shard_dirs, args.top_k)

        output_path = result_output_path(
            args.results_dir,
            args.variant,
            args.run_id,
            query_name,
            args.top_k,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        log.info("Wrote %d rows to %s", len(results), output_path)

    log.info("Done. Results in %s", results_dir)


if __name__ == "__main__":
    main()
