"""Run the reusable WebOrganizer report pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from dolma.distribution_report.concentration_table import (
    build_table,
    write_csv,
    write_latex,
)
from dolma.distribution_report.data_loader import (
    build_grid,
    load_joint,
    load_marginals,
    resolve_eda_paths,
)
from dolma.distribution_report.dummy_sampling import generate_dummy_sampling_grids
from dolma.distribution_report.heatmap_figures import (
    fig_heatmap_doc_count,
    fig_heatmap_token_count,
)
from dolma.distribution_report.influence_runner import (
    run_influence_figures,
    run_influence_split_figures,
)
from dolma.distribution_report.marginal_figures import (
    fig_format_doc_count,
    fig_format_token_count,
    fig_topic_doc_count,
    fig_topic_token_count,
)
from dolma.distribution_report.metrics import concentration_stats
from dolma.distribution_report.output_note import write_output_note
from dolma.distribution_report.report_metadata import write_report_metadata
from dolma.distribution_report.report_summary import build_report_summary
from dolma.distribution_report.sampling_figures import fig_difference, fig_side_by_side
from dolma.distribution_report.sampling_loader import load_sampling_grid
from dolma.distribution_report.top_bins_table import (
    write_outputs as write_top_bins_outputs,
)
from dolma.distribution_report.treemap_figures import (
    fig_treemap_format,
    fig_treemap_interactive,
    fig_treemap_topic,
)

logger = logging.getLogger(__name__)


def _resolve_formats(output_format: str) -> tuple[str, ...]:
    if output_format == "both":
        return ("pdf", "png")
    if output_format == "all":
        return ("pdf", "png", "html")
    return (output_format,)


def _resolve_comparison_inputs(
    *,
    use_dummy: bool,
    representative_manifest: Path | None,
    stratified_manifest: Path | None,
) -> tuple[str, Path | None, Path | None]:
    if use_dummy:
        if representative_manifest or stratified_manifest:
            raise ValueError(
                "--dummy cannot be combined with --representative-manifest or "
                "--stratified-manifest."
            )
        return "dummy", None, None
    if bool(representative_manifest) != bool(stratified_manifest):
        raise ValueError(
            "Provide both --representative-manifest and --stratified-manifest."
        )
    if not representative_manifest or not stratified_manifest:
        raise ValueError(
            "Comparison figures require both --representative-manifest and "
            "--stratified-manifest, or use --dummy."
        )
    return "real", representative_manifest, stratified_manifest


def _resolve_run_label(run_label: str | None, output_dir: Path) -> str:
    candidate = output_dir.name if run_label is None else run_label
    normalized = candidate.strip()
    return normalized or "weborganizer-report"


def run_report(
    *,
    eda_dir: Path,
    output_dir: Path,
    output_format: str,
    run_label: str | None,
    use_dummy: bool,
    representative_manifest: Path | None,
    stratified_manifest: Path | None,
    influence_dir: Path | None = None,
    influence_split_dir: Path | None = None,
) -> int:
    comparison_mode, representative_manifest, stratified_manifest = (
        _resolve_comparison_inputs(
            use_dummy=use_dummy,
            representative_manifest=representative_manifest,
            stratified_manifest=stratified_manifest,
        )
    )
    resolve_eda_paths(eda_dir)
    formats = _resolve_formats(output_format)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_run_label = _resolve_run_label(run_label, output_dir)

    logger.info("Loading EDA data from %s", eda_dir)
    topic_df, format_df = load_marginals(eda_dir)
    joint_df = load_joint(eda_dir)
    grid_df = build_grid(joint_df)

    if comparison_mode == "dummy":
        logger.info("Generating dummy manifest for sampling comparison")
        stratified_grid, representative_grid = generate_dummy_sampling_grids()
    else:
        representative_manifest = cast(Path, representative_manifest)
        stratified_manifest = cast(Path, stratified_manifest)
        logger.info("Loading representative manifest from %s", representative_manifest)
        representative_grid = load_sampling_grid(representative_manifest)
        logger.info("Loading stratified manifest from %s", stratified_manifest)
        stratified_grid = load_sampling_grid(stratified_manifest)

    logger.info("Generating marginal bar charts")
    fig_topic_doc_count(topic_df, output_dir, formats=formats)
    fig_topic_token_count(topic_df, output_dir, formats=formats)
    fig_format_doc_count(format_df, output_dir, formats=formats)
    fig_format_token_count(format_df, output_dir, formats=formats)

    logger.info("Generating heatmaps")
    fig_heatmap_doc_count(grid_df, output_dir, formats=formats)
    fig_heatmap_token_count(grid_df, output_dir, formats=formats)

    logger.info("Generating treemap visualizations")
    fig_treemap_topic(topic_df, output_dir, formats=formats)
    fig_treemap_format(format_df, output_dir, formats=formats)
    if "html" in formats:
        fig_treemap_interactive(topic_df, format_df, grid_df, output_dir)

    logger.info("Computing concentration metrics")
    stats = concentration_stats(grid_df)
    table_df = build_table(stats)
    write_csv(table_df, output_dir)
    write_latex(table_df, output_dir)

    logger.info("Extracting top-20 and bottom-20 bins by token mass")
    write_top_bins_outputs(grid_df, output_dir, n=20)

    logger.info("Generating comparison heatmaps")
    fig_side_by_side(stratified_grid, representative_grid, output_dir, formats)
    fig_difference(stratified_grid, representative_grid, output_dir, formats)

    if influence_dir is not None:
        run_influence_figures(influence_dir, output_dir, formats)

    if influence_split_dir is not None:
        run_influence_split_figures(influence_split_dir, output_dir, formats)

    summary = build_report_summary(
        output_format=output_format,
        formats=formats,
        run_label=resolved_run_label,
        eda_dir=eda_dir,
        output_dir=output_dir,
        joint_grid=grid_df,
        representative_grid=representative_grid,
        stratified_grid=stratified_grid,
        representative_manifest=representative_manifest,
        stratified_manifest=stratified_manifest,
        used_dummy=comparison_mode == "dummy",
    )
    write_output_note(output_dir, summary)
    write_report_metadata(output_dir, summary)

    logger.info("All outputs saved to %s", output_dir)
    return 0


__all__ = ["run_report"]
