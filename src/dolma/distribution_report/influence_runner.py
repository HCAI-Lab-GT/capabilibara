"""Orchestrate influence figure generation from aggregated bin scores."""

from __future__ import annotations

import logging
from pathlib import Path

from dolma.distribution_report.influence_comparison import (
    fig_influence_difference,
    fig_influence_side_by_side,
)
from dolma.distribution_report.influence_composite import fig_influence_signed_2x2
from dolma.distribution_report.influence_composite_diffs import (
    fig_influence_diff_canonical_2panel,
    fig_influence_diff_correct_2x2,
)
from dolma.distribution_report.influence_figures import (
    fig_influence_absolute,
    fig_influence_signed,
)
from dolma.distribution_report.influence_loader import (
    BENCHMARK_DISPLAY_NAMES,
    canonical_axis_order,
    load_all_influence_grids,
)
from dolma.distribution_report.influence_facets import fig_influence_format_facets
from dolma.distribution_report.influence_histograms import (
    fig_influence_histogram,
    fig_influence_histogram_overlay,
)
from dolma.distribution_report.influence_marginals import (
    fig_influence_format_bars,
    fig_influence_topic_bars,
    fig_influence_topic_paired,
)
from dolma.distribution_report.influence_radar import fig_influence_radar
from dolma.distribution_report.influence_tables import (
    contrastive_bins_table,
    correctness_diff_table,
    top_bins_table,
)

logger = logging.getLogger(__name__)

CONTRASTIVE_PAIR = ("queries_socialiqa", "queries_gsm8k")
PRIMARY_BENCHMARKS = (
    "queries_socialiqa",
    "queries_mmlu_social_science",
    "queries_arc_challenge",
    "queries_mmlu_stem",
)
TOPIC_PAIR_FIGURES = (
    ("queries_socialiqa", "queries_arc_challenge"),
    ("queries_mmlu_social_science", "queries_mmlu_stem"),
)

# Benchmarks shown in the manuscript's cross-benchmark overlay (Fig 16).
# Excludes GSM8K and ARC-Easy per the Round 2 decision; includes BBH Snarks
# as the held-out validation set.
MANUSCRIPT_HIST_BENCHMARKS = (
    "queries_socialiqa",
    "queries_mmlu_social_science",
    "queries_arc_challenge",
    "queries_mmlu_stem",
    "queries_bbh_snarks",
)


def run_influence_figures(
    influence_dir: Path,
    output_dir: Path,
    formats: tuple[str, ...],
) -> None:
    logger.info("Generating influence heatmaps from %s", influence_dir)
    grids = load_all_influence_grids(influence_dir)

    canonical = canonical_axis_order()

    for key, grid in grids.items():
        fig_influence_absolute(grid, key, output_dir, formats)
        fig_influence_signed(grid, key, output_dir, formats)
        fig_influence_absolute(
            grid, key, output_dir, formats, axis_order=canonical, suffix="_canonical"
        )
        fig_influence_signed(
            grid, key, output_dir, formats, axis_order=canonical, suffix="_canonical"
        )
        fig_influence_topic_bars(grid, key, output_dir, formats)
        fig_influence_format_bars(grid, key, output_dir, formats)
        fig_influence_histogram(grid, key, output_dir, formats)
        fig_influence_format_facets(grid, key, output_dir, formats=formats)
        top_bins_table(grid, key, output_dir)

    if len(grids) > 1:
        fig_influence_radar(grids, output_dir, "abs_mean_score", formats)
        fig_influence_radar(grids, output_dir, "mean_score", formats)
        manuscript_grids = {
            k: v for k, v in grids.items() if k in MANUSCRIPT_HIST_BENCHMARKS
        }
        fig_influence_histogram_overlay(manuscript_grids, output_dir, formats)

    if all(key in grids for key in PRIMARY_BENCHMARKS):
        manuscript_grids = {key: grids[key] for key in PRIMARY_BENCHMARKS}
        fig_influence_signed_2x2(manuscript_grids, output_dir, formats)
        fig_influence_diff_canonical_2panel(manuscript_grids, output_dir, formats)

    for key_a, key_b in TOPIC_PAIR_FIGURES:
        if key_a in grids and key_b in grids:
            fig_influence_topic_paired(
                grids[key_a],
                grids[key_b],
                key_a,
                key_b,
                output_dir,
                formats,
            )

    key_a, key_b = CONTRASTIVE_PAIR
    if key_a in grids and key_b in grids:
        logger.info("Generating contrastive figures: %s vs %s", key_a, key_b)
        for value_col in ("abs_mean_score", "mean_score"):
            fig_influence_side_by_side(
                grids[key_a],
                grids[key_b],
                key_a,
                key_b,
                value_col,
                output_dir,
                formats,
            )
            fig_influence_side_by_side(
                grids[key_a],
                grids[key_b],
                key_a,
                key_b,
                value_col,
                output_dir,
                formats,
                axis_order=canonical,
                suffix="_canonical",
            )
        fig_influence_difference(
            grids[key_a],
            grids[key_b],
            key_a,
            key_b,
            output_dir,
            formats,
        )
        fig_influence_difference(
            grids[key_a],
            grids[key_b],
            key_a,
            key_b,
            output_dir,
            formats,
            axis_order=canonical,
            suffix="_canonical",
        )
        fig_influence_topic_paired(
            grids[key_a],
            grids[key_b],
            key_a,
            key_b,
            output_dir,
            formats,
        )
        contrastive_bins_table(grids[key_a], grids[key_b], key_a, key_b, output_dir)


def run_influence_split_figures(
    influence_split_dir: Path,
    output_dir: Path,
    formats: tuple[str, ...],
) -> None:
    logger.info(
        "Generating correct/incorrect influence figures from %s", influence_split_dir
    )
    correct_grids = {}
    incorrect_grids = {}
    for key in BENCHMARK_DISPLAY_NAMES:
        correct_path = influence_split_dir / f"{key}_bin_scores_correct.csv"
        incorrect_path = influence_split_dir / f"{key}_bin_scores_incorrect.csv"
        if correct_path.exists() and incorrect_path.exists():
            from dolma.distribution_report.influence_loader import load_influence_grid

            correct_grids[key] = load_influence_grid(correct_path)
            incorrect_grids[key] = load_influence_grid(incorrect_path)

    canonical = canonical_axis_order()
    for key in correct_grids:
        fig_influence_difference(
            correct_grids[key],
            incorrect_grids[key],
            f"{key}__correct",
            f"{key}__incorrect",
            output_dir,
            formats,
        )
        fig_influence_difference(
            correct_grids[key],
            incorrect_grids[key],
            f"{key}__correct",
            f"{key}__incorrect",
            output_dir,
            formats,
            axis_order=canonical,
            suffix="_canonical",
        )
        correctness_diff_table(
            correct_grids[key], incorrect_grids[key], key, output_dir
        )

    if all(key in correct_grids and key in incorrect_grids for key in PRIMARY_BENCHMARKS):
        fig_influence_diff_correct_2x2(
            {key: correct_grids[key] for key in PRIMARY_BENCHMARKS},
            {key: incorrect_grids[key] for key in PRIMARY_BENCHMARKS},
            output_dir,
            formats,
        )


__all__ = [
    "run_influence_figures",
    "run_influence_split_figures",
]
