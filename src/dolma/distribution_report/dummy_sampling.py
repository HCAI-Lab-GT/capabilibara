"""Generate dummy sampling grids for local report smoke tests."""

from __future__ import annotations

import pandas as pd

from dolma.distribution_report.data_loader import build_grid
from dolma.distribution_report.sampling_loader import (
    CANONICAL_FORMAT_LABELS,
    CANONICAL_TOPIC_LABELS,
)
from dolma.pool_sample.sampling import (
    compute_bin_stats,
    generate_dummy_manifest,
    representative_sample,
    stratified_sample,
)


def _group_sampling_manifest(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["topic", "format"], as_index=False)
        .agg(doc_count=("doc_id", "size"), token_count=("token_count", "sum"))
        .rename(columns={"topic": "topic_label", "format": "format_label"})
    )
    grouped["topic_label"] = grouped["topic_label"].map(
        lambda value: f"__label__{value}"
    )
    grouped["format_label"] = grouped["format_label"].map(
        lambda value: f"__label__{value}"
    )
    return build_grid(
        grouped,
        topics=list(CANONICAL_TOPIC_LABELS),
        formats=list(CANONICAL_FORMAT_LABELS),
    )


def generate_dummy_sampling_grids() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = generate_dummy_manifest(n_docs=200_000)
    bin_stats = compute_bin_stats(manifest, target_docs_per_bin=5_000)
    stratified = stratified_sample(manifest, bin_stats, target_docs_per_bin=5_000)
    representative = representative_sample(
        manifest,
        stratified,
        token_target=50_000_000,
    )
    return _group_sampling_manifest(stratified), _group_sampling_manifest(
        representative
    )


__all__ = ["generate_dummy_sampling_grids"]
