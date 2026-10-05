"""Tests for bin_aggregate.aggregate_benchmark.

Covers the silent whole-shard drop on length mismatch: it now raises ValueError
(was a log.error + continue that produced a misleadingly incomplete aggregation
with no error). Also covers the happy path so the base mean aggregator has
direct test coverage (it was previously only exercised transitively via the
fused equivalence oracle).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_attribution.attribution.trackstar.bin_aggregate import (
    aggregate_benchmark,
    load_manifest_bin_map,
    load_shard_doc_uuids,
)


def _write_shard(
    benchmark_dir: Path,
    shard_dir: Path,
    name: str,
    matrix: np.ndarray,
    uuids: list[str],
) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(benchmark_dir / f"{name}.npy", np.asarray(matrix, dtype=np.float32))
    with open(shard_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for uuid in uuids:
            f.write(json.dumps({"id": uuid}) + "\n")


class TestLengthMismatchTrap:
    def test_more_scores_than_docs_raises(self, tmp_path: Path):
        benchmark_dir = tmp_path / "scores" / "queries_demo"
        shard_dir = tmp_path / "shards"
        _write_shard(
            benchmark_dir,
            shard_dir,
            "shard_0000",
            np.zeros((3, 1)),  # 3 score rows
            ["u0", "u1"],  # 2 source docs
        )
        with pytest.raises(ValueError, match="Length mismatch shard_0000"):
            aggregate_benchmark(benchmark_dir, shard_dir, bin_map={"u0": ("T", "F")})

    def test_more_docs_than_scores_raises(self, tmp_path: Path):
        benchmark_dir = tmp_path / "scores" / "queries_demo"
        shard_dir = tmp_path / "shards"
        _write_shard(
            benchmark_dir,
            shard_dir,
            "shard_0000",
            np.zeros((1, 1)),  # 1 score row
            ["u0", "u1"],  # 2 source docs
        )
        with pytest.raises(ValueError, match="Length mismatch"):
            aggregate_benchmark(benchmark_dir, shard_dir, bin_map={})

    def test_missing_source_jsonl_skipped_not_raised(self, tmp_path: Path):
        # A missing JSONL is a different condition (logs + skip), not a mismatch.
        benchmark_dir = tmp_path / "scores" / "queries_demo"
        shard_dir = tmp_path / "shards"
        benchmark_dir.mkdir(parents=True)
        np.save(benchmark_dir / "shard_0000.npy", np.zeros((1, 1)))
        df = aggregate_benchmark(benchmark_dir, shard_dir, bin_map={})
        assert len(df) == 0


class TestHappyPath:
    def test_mean_score_per_bin(self, tmp_path: Path):
        benchmark_dir = tmp_path / "scores" / "queries_demo"
        shard_dir = tmp_path / "shards"
        _write_shard(
            benchmark_dir,
            shard_dir,
            "shard_0000",
            np.array([[1.0], [3.0]]),  # per-doc means across 1 query: 1.0, 3.0
            ["u0", "u1"],
        )
        bin_map = {
            "u0": ("software", "academic_writing"),
            "u1": ("software", "academic_writing"),
        }
        df = aggregate_benchmark(benchmark_dir, shard_dir, bin_map)
        row = df[df["topic_label"] == "software"].iloc[0]
        assert row["doc_count"] == 2
        assert abs(row["mean_score"] - 2.0) < 1e-6

    def test_docs_without_bin_entry_counted_as_missed(self, tmp_path: Path):
        benchmark_dir = tmp_path / "scores" / "queries_demo"
        shard_dir = tmp_path / "shards"
        _write_shard(
            benchmark_dir,
            shard_dir,
            "shard_0000",
            np.array([[5.0], [7.0]]),
            ["u0", "orphan"],  # orphan has no bin_map entry
        )
        df = aggregate_benchmark(
            benchmark_dir, shard_dir, bin_map={"u0": ("software", "academic_writing")}
        )
        row = df[df["topic_label"] == "software"].iloc[0]
        assert row["doc_count"] == 1
        assert abs(row["mean_score"] - 5.0) < 1e-6


class TestStrictManifestBinMap:
    def test_rejects_missing_bin_label(self, tmp_path: Path):
        manifest = tmp_path / "manifest.parquet"
        pd.DataFrame(
            {
                "doc_id": ["u0", "u1"],
                "bin_topic": ["topic", None],
                "bin_format": ["format", "format"],
            }
        ).to_parquet(manifest, index=False)

        with pytest.raises(ValueError, match="missing bin labels"):
            load_manifest_bin_map(manifest, require_complete=True)

    def test_rejects_duplicate_document_ids(self, tmp_path: Path):
        manifest = tmp_path / "manifest.parquet"
        pd.DataFrame(
            {
                "doc_id": ["u0", "u0"],
                "bin_topic": ["topic", "topic"],
                "bin_format": ["format", "format"],
            }
        ).to_parquet(manifest, index=False)

        with pytest.raises(ValueError, match="duplicate doc_id"):
            load_manifest_bin_map(manifest, require_complete=True)


def test_load_shard_doc_uuids_uses_lf_not_unicode_line_boundaries(
    tmp_path: Path,
) -> None:
    """U+2028 is valid inside a JSON string, not a JSONL record delimiter."""
    shard = tmp_path / "shard_0000.jsonl"
    shard.write_text(
        json.dumps({"id": "u0", "text": "first\u2028second"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"id": "u1"})
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()

    assert load_shard_doc_uuids(shard, expected_sha256=digest) == ["u0", "u1"]
