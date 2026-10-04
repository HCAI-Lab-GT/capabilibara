import json
import logging
from pathlib import Path

import numpy as np

from data_attribution.attribution.trackstar.bin_aggregate_split import (
    aggregate_benchmark_split,
    load_correctness_mask,
    resolve_query_file,
)
from tests.data_attribution.query_manifest_test_utils import (
    write_strict_query_pointer,
)


def test_resolve_query_file_direct_mapping() -> None:
    assert resolve_query_file("queries_gsm8k") == "olmes_gsm8k.jsonl"
    assert resolve_query_file("queries_arc_challenge") == "olmes_arc_challenge.jsonl"
    assert resolve_query_file("queries_arc_combined") == "olmes_arc_combined.jsonl"


def test_resolve_query_file_arc_combined_full_benchmark_token() -> None:
    # Regression: the F5B ARC-rescore fused pass discovers benchmark tokens like
    # "comma_2t_queries_olmes_arc_combined"; before arc_combined was added to
    # QUERY_FILE_MAP the fused correctness-mask resolver raised FileNotFoundError
    # on it (arc_challenge/arc_easy resolved via the substring fallback but
    # arc_combined had no matching subset key).
    assert (
        resolve_query_file("comma_2t_queries_olmes_arc_combined")
        == "olmes_arc_combined.jsonl"
    )


def test_resolve_query_file_cot_suffix_mapping() -> None:
    assert (
        resolve_query_file("soc161_queries_olmes_instruct_cot_mmlu_stem")
        == "olmes_instruct_cot_mmlu_stem.jsonl"
    )
    assert (
        resolve_query_file("soc161_queries_olmes_instruct_cot_socialiqa")
        == "olmes_instruct_cot_socialiqa.jsonl"
    )


def test_resolve_query_file_standard_suffix_mapping() -> None:
    assert resolve_query_file("run_queries_gsm8k") == "olmes_gsm8k.jsonl"
    assert (
        resolve_query_file("soc170_queries_olmes_bbh_snarks")
        == "olmes_bbh_snarks.jsonl"
    )
    assert (
        resolve_query_file("soc170_queries_olmes_instruct_bbh_snarks")
        == "olmes_instruct_bbh_snarks.jsonl"
    )
    assert resolve_query_file("queries_base_medqa") == "olmes_base_medqa.jsonl"


def test_resolve_query_file_candidate_fallback() -> None:
    candidates = [Path("custom_socialiqa_eval.jsonl")]
    assert (
        resolve_query_file("custom_run_queries_socialiqa_extra", candidates)
        == "custom_socialiqa_eval.jsonl"
    )


def test_resolve_query_file_holdout_probe_with_candidate(tmp_path: Path) -> None:
    query_dir = tmp_path / "queries_tombench"
    query_dir.mkdir()
    (query_dir / "olmes_tombench.jsonl").touch()
    candidates = list(query_dir.glob("*.jsonl"))
    assert resolve_query_file("queries_tombench", candidates) == "olmes_tombench.jsonl"


def test_resolve_query_file_by_frozen_result_group(tmp_path: Path) -> None:
    query = tmp_path / "finance_mmlu_finance_bridge.jsonl"
    query.write_text(
        json.dumps({"result_group": "mmlu_finance_bridge", "query_id": "q1"}) + "\n"
    )

    assert resolve_query_file("mmlu_finance_bridge", [query]) == query.name


def test_resolve_query_file_core4_no_regression() -> None:
    assert resolve_query_file("queries_arc_challenge") == "olmes_arc_challenge.jsonl"
    assert resolve_query_file("queries_socialiqa") == "olmes_socialiqa.jsonl"


def test_resolve_query_file_dclm_full_mixture_core4_names() -> None:
    prefix = "dclm_full_mixture_queries_olmes_"
    assert resolve_query_file(f"{prefix}arc_challenge") == "olmes_arc_challenge.jsonl"
    assert (
        resolve_query_file(f"{prefix}mmlu_social_science")
        == "olmes_mmlu_social_science.jsonl"
    )
    assert resolve_query_file(f"{prefix}mmlu_stem") == "olmes_mmlu_stem.jsonl"
    assert resolve_query_file(f"{prefix}socialiqa") == "olmes_socialiqa.jsonl"


def test_resolve_query_file_unknown_returns_none() -> None:
    assert resolve_query_file("queries_completely_unknown_probe_xyz", []) is None


def test_load_correctness_mask_uses_immutable_strict_manifest_member(
    tmp_path: Path,
) -> None:
    compatibility, _immutable, _pointer = write_strict_query_pointer(
        tmp_path / "queries",
        member_name="olmes_instruct_demo.jsonl",
        immutable_rows=[{"is_correct": True}],
        compatibility_rows=[{"is_correct": False}],
    )

    mask = load_correctness_mask(compatibility)

    assert mask.tolist() == [True]


def _write_split_fixture(tmp_path: Path) -> tuple[Path, Path]:
    benchmark_dir = tmp_path / "scores" / "queries_demo"
    shard_dir = tmp_path / "shards"
    benchmark_dir.mkdir(parents=True)
    shard_dir.mkdir(parents=True)
    np.save(benchmark_dir / "shard_0000.npy", np.array([[1.0, 2.0], [3.0, 4.0]]))
    with open(shard_dir / "shard_0000.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "u0"}) + "\n")
        f.write(json.dumps({"id": "u1"}) + "\n")
    return benchmark_dir, shard_dir


def test_aggregate_benchmark_split_partitions_correct_and_incorrect(
    tmp_path: Path,
) -> None:
    benchmark_dir, shard_dir = _write_split_fixture(tmp_path)
    bin_map = {"u0": ("T", "F")}
    correct_mask = np.array([True, False])

    df_correct, df_incorrect = aggregate_benchmark_split(
        benchmark_dir, shard_dir, bin_map, correct_mask
    )

    assert list(df_correct["mean_score"]) == [1.0]
    assert list(df_incorrect["mean_score"]) == [2.0]
    assert int(df_correct["doc_count"].iloc[0]) == 1


def test_aggregate_benchmark_split_logs_missed_docs(tmp_path: Path, caplog) -> None:
    benchmark_dir, shard_dir = _write_split_fixture(tmp_path)
    bin_map = {"u0": ("T", "F")}
    correct_mask = np.array([True, False])

    with caplog.at_level(
        logging.INFO,
        logger="data_attribution.attribution.trackstar.bin_aggregate_split",
    ):
        aggregate_benchmark_split(benchmark_dir, shard_dir, bin_map, correct_mask)

    assert "1 missed" in caplog.text
    assert "50.00%" in caplog.text
