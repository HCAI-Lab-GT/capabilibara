"""Aggregate per-doc influence scores into 24x24 bin-level means.

Reads per-shard numpy score matrices, resolves doc IDs to (topic, format)
bins via the stratified sample manifest, writes one CSV per benchmark.
"""

import argparse
import io
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_attribution.attribution.trackstar.runtime_inputs import stable_file_bytes

log = logging.getLogger(__name__)

BENCHMARKS = (
    "queries_gsm8k",
    "queries_socialiqa",
    "queries_mmlu_social_science",
    "queries_mmlu_stem",
    "queries_arc_easy",
    "queries_arc_challenge",
    "queries_bbh_snarks",
    "queries_bbh_causal_judgement",
    "queries_bbh_sports_understanding",
)


@dataclass
class BinAccumulator:
    score_sum: float = 0.0
    abs_score_sum: float = 0.0
    score_sq_sum: float = 0.0
    doc_count: int = 0


def load_manifest_bin_map(
    manifest_path: Path,
    *,
    require_complete: bool = False,
    expected_sha256: str | None = None,
) -> dict[str, tuple[str, str]]:
    if expected_sha256 is None:
        df = pd.read_parquet(
            manifest_path, columns=["doc_id", "bin_topic", "bin_format"]
        )
    else:
        content = stable_file_bytes(manifest_path, expected_sha256=expected_sha256)
        df = pd.read_parquet(
            io.BytesIO(content), columns=["doc_id", "bin_topic", "bin_format"]
        )
    if require_complete:
        if bool(df[["doc_id", "bin_topic", "bin_format"]].isna().to_numpy().any()):
            raise ValueError("manifest has missing bin labels or document IDs")
        if df["doc_id"].duplicated().any():
            raise ValueError("manifest has duplicate doc_id values")
    df = df.dropna(subset=["bin_topic", "bin_format"])
    bin_map = dict(
        zip(
            (str(doc_id) for doc_id in df["doc_id"].tolist()),
            zip(
                df["bin_topic"].tolist(),
                df["bin_format"].tolist(),
                strict=True,
            ),
            strict=True,
        )
    )
    if require_complete and len(bin_map) != len(df):
        raise ValueError("manifest bin map does not cover every manifest row")
    return bin_map


def load_shard_doc_uuids(
    jsonl_path: Path, *, expected_sha256: str | None = None
) -> list[str]:
    uuids: list[str] = []
    if expected_sha256 is None:
        content = jsonl_path.read_bytes()
    else:
        content = stable_file_bytes(jsonl_path, expected_sha256=expected_sha256)
    for raw_line in content.split(b"\n"):
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        doc = json.loads(line)
        uuids.append(doc["id"])
    return uuids


def aggregate_shard(
    scores: np.ndarray,
    doc_uuids: list[str],
    bin_map: dict[str, tuple[str, str]],
    accumulators: dict[tuple[str, str], BinAccumulator],
    query_mask: np.ndarray | None = None,
) -> int:
    if query_mask is not None:
        scores = scores[:, query_mask]
    if scores.shape[1] == 0:
        return 0
    doc_means = scores.mean(axis=1).astype(np.float64)
    missed = 0
    for i, uuid in enumerate(doc_uuids):
        bin_key = bin_map.get(uuid)
        if bin_key is None:
            missed += 1
            continue
        val = float(doc_means[i])
        acc = accumulators.setdefault(bin_key, BinAccumulator())
        acc.score_sum += val
        acc.abs_score_sum += abs(val)
        acc.score_sq_sum += val * val
        acc.doc_count += 1
    return missed


def finalize_bins(
    accumulators: dict[tuple[str, str], BinAccumulator],
) -> pd.DataFrame:
    rows = []
    for (topic, fmt), acc in sorted(accumulators.items()):
        mean = acc.score_sum / acc.doc_count if acc.doc_count > 0 else 0.0
        abs_mean = acc.abs_score_sum / acc.doc_count if acc.doc_count > 0 else 0.0
        variance = (
            (acc.score_sq_sum / acc.doc_count - mean * mean)
            if acc.doc_count > 1
            else 0.0
        )
        std = math.sqrt(max(variance, 0.0))
        rows.append(
            {
                "topic_label": topic,
                "format_label": fmt,
                "mean_score": mean,
                "abs_mean_score": abs_mean,
                "doc_count": acc.doc_count,
                "std_score": std,
            }
        )
    return pd.DataFrame(rows)


def aggregate_benchmark(
    benchmark_dir: Path,
    shard_dir: Path,
    bin_map: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    accumulators: dict[tuple[str, str], BinAccumulator] = {}
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
            raise ValueError(
                f"Length mismatch {shard_name}: {len(scores)} score rows vs "
                f"{len(doc_uuids)} source docs. The whole shard would be silently "
                f"dropped. Pre-align with shard_trim.trim_shard_jsonl (keep the "
                f"index row count), or fix the source shard. "
                f"scores={npy_path} docs={jsonl_path}"
            )

        missed = aggregate_shard(scores, doc_uuids, bin_map, accumulators)
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
    return finalize_bins(accumulators)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Aggregate doc-level scores into 24x24 bin-level means"
    )
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=list(BENCHMARKS),
    )
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
        df = aggregate_benchmark(benchmark_dir, args.shard_dir, bin_map)
        out_path = args.output_dir / f"{benchmark}_bin_scores.csv"
        df.to_csv(out_path, index=False)
        log.info("Wrote %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
