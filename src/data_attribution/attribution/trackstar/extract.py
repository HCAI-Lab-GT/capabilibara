"""Extract all attribution scores and join with manifest labels.

Reads scores.bin from every shard directory, pairs with doc IDs from
source JSONL, and joins with the manifest to produce a combined parquet
with per-benchmark attribution scores and WebOrganizer bin labels.

Usage:
    data-attribution-trackstar-extract \
        --run-id 20260321T063351Z_267024 --variant base \
        --manifest /path/to/manifest.parquet \
        --shard-dir /path/to/stratified_shards \
        --scores-dir /path/to/scores \
        --results-dir /path/to/results
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_shard_scores(score_dir: Path) -> np.ndarray:
    info_path = score_dir / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    dtype = np.dtype(
        {
            "names": info["dtype"]["names"],
            "formats": info["dtype"]["formats"],
            "offsets": info["dtype"]["offsets"],
            "itemsize": info["dtype"]["itemsize"],
        }
    )
    mmap = np.memmap(score_dir / "scores.bin", dtype=dtype, mode="r")
    return np.asarray(mmap["score_0"], dtype=np.float32)


def load_shard_doc_ids(jsonl_path: Path) -> list[str]:
    doc_ids = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_ids.append(doc["id"])
    return doc_ids


def extract_query_scores(
    query_name: str,
    score_base: Path,
    shard_dir: Path,
) -> pd.DataFrame:
    query_score_dir = score_base / query_name
    if not query_score_dir.exists():
        log.error("Score directory not found: %s", query_score_dir)
        raise SystemExit(1)

    all_doc_ids = []
    all_scores = []

    for source_dir in sorted(query_score_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        for shard_path in sorted(source_dir.iterdir()):
            if not shard_path.is_dir():
                continue
            shard_name = shard_path.name
            jsonl_path = shard_dir / f"{shard_name}.jsonl"

            scores = load_shard_scores(shard_path)
            doc_ids = load_shard_doc_ids(jsonl_path)

            if len(scores) != len(doc_ids):
                log.error(
                    "Length mismatch for %s/%s: %d scores vs %d docs",
                    query_name,
                    shard_name,
                    len(scores),
                    len(doc_ids),
                )
                raise SystemExit(1)

            all_doc_ids.extend(doc_ids)
            all_scores.extend(scores.tolist())
            log.info("Loaded %s/%s: %d docs", query_name, shard_name, len(scores))

    return pd.DataFrame({"doc_id": all_doc_ids, f"score_{query_name}": all_scores})


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Extract all scores and join with manifest"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
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

    score_base = args.scores_dir / args.variant / args.run_id
    result_dir = args.results_dir / args.variant / args.run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(args.manifest)
    log.info("Manifest: %d rows, columns: %s", len(manifest), list(manifest.columns))

    query_dirs = sorted(
        p.name
        for p in score_base.iterdir()
        if p.is_dir() and p.name.startswith("queries_")
    )
    log.info("Found %d query directories: %s", len(query_dirs), query_dirs)

    combined = None
    for query_name in query_dirs:
        df = extract_query_scores(query_name, score_base, args.shard_dir)
        log.info("Extracted %s: %d rows", query_name, len(df))

        if combined is None:
            combined = df
        else:
            combined = combined.merge(df, on="doc_id", how="outer")

    combined = combined.merge(manifest, on="doc_id", how="left")

    null_bins = combined["bin_topic"].isna().sum()
    if null_bins > 0:
        log.warning("%d docs had no manifest match (null bin labels)", null_bins)

    out_path = result_dir / "all_scores_combined.parquet"
    combined.to_parquet(out_path, index=False)
    log.info(
        "Wrote %s (%d rows, %d columns)", out_path, len(combined), len(combined.columns)
    )

    score_cols = [c for c in combined.columns if c.startswith("score_")]
    for col in score_cols:
        s = combined[col]
        log.info(
            "%s: min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
            col,
            s.min(),
            s.max(),
            s.mean(),
            s.std(),
        )


if __name__ == "__main__":
    main()
