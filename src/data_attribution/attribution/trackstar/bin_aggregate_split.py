"""Bin-level aggregation split by query correctness.

Extends the base bin_aggregate pipeline to produce separate CSVs
for queries the model answered correctly vs incorrectly. Reads
query metadata JSONL to determine the is_correct partition.

Usage:
    data-attribution-trackstar-bin-aggregate-split \
        --scores-dir /path/to/scores_full/base/<run_id> \
        --shard-dir /path/to/shards_10k/sample_10000_docs \
        --manifest /path/to/working_sample_manifest.parquet \
        --query-dir /path/to/queries/base \
        --output-dir artifacts/influence_bin_scores_split
"""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from data_attribution.attribution.trackstar.bin_aggregate import (
    BENCHMARKS,
    BinAccumulator,
    aggregate_shard,
    finalize_bins,
    load_manifest_bin_map,
    load_shard_doc_uuids,
)
from data_attribution.query_manifest_resolver import resolve_query_manifest_member

log = logging.getLogger(__name__)

QUERY_FILE_MAP: dict[str, str] = {
    "queries_gsm8k": "olmes_gsm8k.jsonl",
    "queries_socialiqa": "olmes_socialiqa.jsonl",
    "queries_mmlu_social_science": "olmes_mmlu_social_science.jsonl",
    "queries_mmlu_stem": "olmes_mmlu_stem.jsonl",
    "queries_arc_easy": "olmes_arc_easy.jsonl",
    "queries_arc_challenge": "olmes_arc_challenge.jsonl",
    "queries_arc_combined": "olmes_arc_combined.jsonl",
    "queries_bbh_snarks": "olmes_bbh_snarks.jsonl",
    "queries_bbh_causal_judgement": "olmes_bbh_causal_judgement.jsonl",
    "queries_bbh_sports_understanding": "olmes_bbh_sports_understanding.jsonl",
}

_COT_SUFFIX_MAP: dict[str, str] = {
    "gsm8k": "olmes_instruct_cot_gsm8k.jsonl",
    "socialiqa": "olmes_instruct_cot_socialiqa.jsonl",
    "mmlu_social_science": "olmes_instruct_cot_mmlu_social_science.jsonl",
    "mmlu_stem": "olmes_instruct_cot_mmlu_stem.jsonl",
}

_QUERY_FILE_PREFIXES: dict[str, str] = {
    "queries_olmes_instruct_cot_": "olmes_instruct_cot_",
    "queries_olmes_instruct_": "olmes_instruct_",
    "queries_base_": "olmes_base_",
    "queries_olmes_": "olmes_",
}


def _resolve_prefixed_query_file(benchmark: str) -> str | None:
    known_subsets = {key.replace("queries_", "") for key in QUERY_FILE_MAP}
    for benchmark_prefix, file_prefix in _QUERY_FILE_PREFIXES.items():
        if benchmark_prefix not in benchmark:
            continue
        subset = benchmark.rsplit(benchmark_prefix, maxsplit=1)[-1]
        if file_prefix == "olmes_base_":
            return f"{file_prefix}{subset}.jsonl"
        if subset in known_subsets:
            return f"{file_prefix}{subset}.jsonl"
    return None


def _resolve_holdout_olmes_query(
    benchmark: str,
    query_candidates: Sequence[Path] | None,
) -> str | None:
    prefix = "queries_olmes_" if benchmark.startswith("queries_olmes_") else "queries_"
    probe = benchmark[len(prefix) :]
    candidate_name = f"olmes_{probe}.jsonl"
    if query_candidates is not None:
        candidate_names = {c.name for c in query_candidates}
        if candidate_name in candidate_names:
            return candidate_name
        return None
    return candidate_name


def _resolve_result_group_query(
    benchmark: str, query_candidates: Sequence[Path]
) -> str | None:
    matches: list[str] = []
    for candidate in query_candidates:
        try:
            with candidate.open(encoding="utf-8") as stream:
                first = next((line for line in stream if line.strip()), None)
            row = json.loads(first) if first is not None else None
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict) and row.get("result_group") == benchmark:
            matches.append(candidate.name)
    if len(matches) > 1:
        raise ValueError(f"multiple frozen query files declare {benchmark!r}")
    return matches[0] if matches else None


def resolve_query_file(  # noqa: C901
    benchmark: str,
    query_candidates: Sequence[Path] | None = None,
) -> str | None:
    query_file = QUERY_FILE_MAP.get(benchmark)
    if query_file is not None:
        return query_file

    query_file = _resolve_prefixed_query_file(benchmark)
    if query_file is not None:
        return query_file

    for key, filename in QUERY_FILE_MAP.items():
        if benchmark.endswith(key):
            return filename

    for suffix, filename in _COT_SUFFIX_MAP.items():
        if benchmark.endswith(suffix):
            return filename

    if benchmark.startswith("queries_"):
        return _resolve_holdout_olmes_query(benchmark, query_candidates)

    if query_candidates is None:
        return None

    result_group_match = _resolve_result_group_query(benchmark, query_candidates)
    if result_group_match is not None:
        return result_group_match

    for candidate in query_candidates:
        for key in QUERY_FILE_MAP:
            subset = key.replace("queries_", "")
            if subset in benchmark and subset in candidate.name:
                return candidate.name
    return None


def load_correctness_mask(query_jsonl: Path) -> np.ndarray:
    query_jsonl = resolve_query_manifest_member(query_jsonl)
    mask: list[bool] = []
    with open(query_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mask.append(bool(row.get("is_correct", False)))
    return np.array(mask, dtype=bool)


def aggregate_benchmark_split(
    benchmark_dir: Path,
    shard_dir: Path,
    bin_map: dict[str, tuple[str, str]],
    correct_mask: np.ndarray,
) -> tuple:
    acc_correct: dict[tuple[str, str], BinAccumulator] = {}
    acc_incorrect: dict[tuple[str, str], BinAccumulator] = {}
    incorrect_mask = ~correct_mask
    total_docs = 0
    total_missed = 0

    shard_files = sorted(benchmark_dir.glob("shard_*.npy"))
    log.info("Found %d shard files in %s", len(shard_files), benchmark_dir)
    log.info(
        "Query split: %d correct, %d incorrect",
        int(correct_mask.sum()),
        int(incorrect_mask.sum()),
    )

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

        missed_correct = aggregate_shard(
            scores, doc_uuids, bin_map, acc_correct, query_mask=correct_mask
        )
        missed_incorrect = aggregate_shard(
            scores, doc_uuids, bin_map, acc_incorrect, query_mask=incorrect_mask
        )
        missed = max(missed_correct, missed_incorrect)
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
    return finalize_bins(acc_correct), finalize_bins(acc_incorrect)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Bin-level aggregation split by query correctness"
    )
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
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

        query_file = resolve_query_file(
            benchmark,
            query_candidates=sorted(args.query_dir.glob("*.jsonl")),
        )
        if query_file is None:
            log.warning("No query file mapping for %s", benchmark)
            continue

        query_path = resolve_query_manifest_member(args.query_dir / query_file)
        if not query_path.exists():
            log.warning("Query JSONL not found: %s", query_path)
            continue

        log.info("Processing %s", benchmark)
        correct_mask = load_correctness_mask(query_path)
        df_correct, df_incorrect = aggregate_benchmark_split(
            benchmark_dir, args.shard_dir, bin_map, correct_mask
        )

        correct_path = args.output_dir / f"{benchmark}_bin_scores_correct.csv"
        incorrect_path = args.output_dir / f"{benchmark}_bin_scores_incorrect.csv"
        df_correct.to_csv(correct_path, index=False)
        df_incorrect.to_csv(incorrect_path, index=False)
        log.info(
            "Wrote %s (%d rows) and %s (%d rows)",
            correct_path,
            len(df_correct),
            incorrect_path,
            len(df_incorrect),
        )


if __name__ == "__main__":
    main()
