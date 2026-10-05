"""Build reusable summary artifacts for WebOrganizer report outputs."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import cast

import pandas as pd

from dolma.distribution_report.data_loader import display_label
from dolma.distribution_report.sampling_data import build_share_difference_table
from dolma.distribution_report.sampling_loader import (
    CANONICAL_FORMAT_LABELS,
    CANONICAL_TOPIC_LABELS,
)

FIGURE_BASENAMES = (
    "fig_topic_doc_count",
    "fig_topic_token_count",
    "fig_format_doc_count",
    "fig_format_token_count",
    "fig_heatmap_doc_count",
    "fig_heatmap_token_count",
    "fig_treemap_topic",
    "fig_treemap_format",
    "fig_stratified_vs_representative",
    "fig_sampling_difference",
)
INTERACTIVE_FILENAMES = ("fig_treemap_interactive.html",)
TABLE_FILENAMES = (
    "table_concentration.csv",
    "table_concentration.tex",
    "top_bottom_bins.csv",
    "tab-top-bins.tabular-top.tex",
    "tab-top-bins.tabular-bottom.tex",
)
NOTE_FILENAME = "paper_figures_note.md"
METADATA_FILENAME = "report_metadata.json"


def _positive_range(values: pd.Series) -> dict[str, int] | None:
    positive = cast(pd.Series, values[values > 0])
    if positive.empty:
        return None
    return {"min": int(positive.min()), "max": int(positive.max())}


def _build_command(
    *,
    output_format: str,
    eda_dir: Path,
    output_dir: Path,
    run_label: str,
    representative_manifest: Path | None,
    stratified_manifest: Path | None,
    used_dummy: bool,
) -> str:
    parts = [
        "uv run data-attribution-weborganizer-report",
        f"--eda-dir {shlex.quote(str(eda_dir))}",
        f"--output-dir {shlex.quote(str(output_dir))}",
        f"--format {shlex.quote(output_format)}",
        f"--run-label {shlex.quote(run_label)}",
    ]
    if used_dummy:
        parts.append("--dummy")
    else:
        parts.append(
            f"--representative-manifest {shlex.quote(str(representative_manifest))}"
        )
        parts.append(f"--stratified-manifest {shlex.quote(str(stratified_manifest))}")
    return " \\\n  ".join(parts)


def _serialize_bin(row: pd.Series) -> dict[str, object]:
    topic_label = cast(str, row["topic_label"])
    format_label = cast(str, row["format_label"])
    return {
        "topic_label": topic_label,
        "format_label": format_label,
        "topic_display_label": display_label(topic_label),
        "format_display_label": display_label(format_label),
        "log2_share_ratio": round(float(cast(float, row["log2_share_ratio"])), 4),
        "stratified_share": round(float(cast(float, row["stratified_share"])), 6),
        "representative_share": round(
            float(cast(float, row["representative_share"])), 6
        ),
    }


def _build_scale_summary(
    figure_name: str,
    transform: str,
    values: pd.Series,
) -> dict[str, object]:
    positive_range = _positive_range(values)
    return {
        "figure_name": figure_name,
        "transform": transform,
        "positive_min": None if positive_range is None else positive_range["min"],
        "positive_max": None if positive_range is None else positive_range["max"],
        "empty_cells": "gray",
    }


def _build_output_inventory(formats: tuple[str, ...]) -> dict[str, object]:
    figure_files = {
        name: [f"{name}.{fmt}" for fmt in formats] for name in FIGURE_BASENAMES
    }
    generated_files = [
        filename for filenames in figure_files.values() for filename in filenames
    ]
    if "html" in formats:
        generated_files.extend(INTERACTIVE_FILENAMES)
    generated_files.extend(TABLE_FILENAMES)
    generated_files.extend([NOTE_FILENAME, METADATA_FILENAME])
    return {
        "formats": list(formats),
        "figure_files": figure_files,
        "table_files": list(TABLE_FILENAMES),
        "note_file": NOTE_FILENAME,
        "metadata_file": METADATA_FILENAME,
        "generated_files": sorted(generated_files),
    }


def build_report_summary(
    *,
    output_format: str,
    formats: tuple[str, ...],
    run_label: str,
    eda_dir: Path,
    output_dir: Path,
    joint_grid: pd.DataFrame,
    representative_grid: pd.DataFrame,
    stratified_grid: pd.DataFrame,
    representative_manifest: Path | None,
    stratified_manifest: Path | None,
    used_dummy: bool,
) -> dict[str, object]:
    diff_df = build_share_difference_table(stratified_grid, representative_grid)
    finite = cast(pd.DataFrame, diff_df[diff_df["log2_share_ratio"].notna()])
    over = finite.sort_values(by=["log2_share_ratio"], ascending=False).head(3)
    under = finite.sort_values(by=["log2_share_ratio"], ascending=True).head(3)
    sampling_values = cast(
        pd.Series,
        pd.concat(
            [representative_grid["token_count"], stratified_grid["token_count"]],
            ignore_index=True,
        ),
    )
    return {
        "report_name": "weborganizer-report",
        "run_label": run_label,
        "inputs": {
            "eda_dir": str(eda_dir),
            "output_dir": str(output_dir),
            "output_format": output_format,
            "representative_manifest": None
            if representative_manifest is None
            else str(representative_manifest),
            "stratified_manifest": None
            if stratified_manifest is None
            else str(stratified_manifest),
            "used_dummy": used_dummy,
            "canonical_remote_source": None
            if used_dummy
            else "https://huggingface.co/datasets/HCAI-Lab/archive-dolma3-pool-150b-samples",
            "reproduction_command": _build_command(
                output_format=output_format,
                eda_dir=eda_dir,
                output_dir=output_dir,
                run_label=run_label,
                representative_manifest=representative_manifest,
                stratified_manifest=stratified_manifest,
                used_dummy=used_dummy,
            ),
        },
        "taxonomy": {
            "name": "WebOrganizer",
            "topic_count": len(CANONICAL_TOPIC_LABELS),
            "format_count": len(CANONICAL_FORMAT_LABELS),
            "total_bins": len(CANONICAL_TOPIC_LABELS) * len(CANONICAL_FORMAT_LABELS),
            "topic_labels": list(CANONICAL_TOPIC_LABELS),
            "format_labels": list(CANONICAL_FORMAT_LABELS),
        },
        "heatmap_scales": {
            "fig_heatmap_doc_count": _build_scale_summary(
                "fig_heatmap_doc_count",
                "log10(doc_count)",
                cast(pd.Series, joint_grid["doc_count"]),
            ),
            "fig_heatmap_token_count": _build_scale_summary(
                "fig_heatmap_token_count",
                "log10(token_count_est)",
                cast(pd.Series, joint_grid["token_count_est"]),
            ),
            "fig_stratified_vs_representative": _build_scale_summary(
                "fig_stratified_vs_representative",
                "log10(token_count)",
                sampling_values,
            ),
            "fig_sampling_difference": {
                "figure_name": "fig_sampling_difference",
                "transform": "log2(stratified_share / representative_share)",
                "empty_cells": "gray",
            },
        },
        "comparison": {
            "most_overrepresented": [_serialize_bin(row) for _, row in over.iterrows()],
            "most_underrepresented": [
                _serialize_bin(row) for _, row in under.iterrows()
            ],
        },
        "outputs": _build_output_inventory(formats),
    }
