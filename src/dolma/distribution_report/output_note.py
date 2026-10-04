"""Write human-readable notes for WebOrganizer report outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _format_range(scale: dict[str, Any]) -> str:
    if scale.get("positive_min") is None:
        return "no positive cells"
    return f"{int(scale['positive_min']):,} to {int(scale['positive_max']):,}"


def _format_bin(bin_summary: dict[str, Any]) -> str:
    return (
        f"- {bin_summary['topic_display_label']} x "
        f"{bin_summary['format_display_label']}: log2 share ratio "
        f"{float(bin_summary['log2_share_ratio']):.2f}; stratified "
        f"{float(bin_summary['stratified_share']):.2%}, representative "
        f"{float(bin_summary['representative_share']):.2%}"
    )


def write_output_note(output_dir: Path, summary: dict[str, Any]) -> Path:
    note_path = output_dir / "paper_figures_note.md"
    inputs = summary["inputs"]
    scales = summary["heatmap_scales"]
    comparison = summary["comparison"]
    lines = [
        "# WebOrganizer report notes",
        "",
        "## Run",
        f"- Run label: `{summary['run_label']}`",
        f"- EDA directory: `{inputs['eda_dir']}`",
        f"- Output directory: `{inputs['output_dir']}`",
    ]
    if inputs["used_dummy"]:
        lines.append("- Sampling comparison inputs: generated via `--dummy`.")
    else:
        lines.append(
            f"- Sampling source: `{inputs.get('sampling_source', 'manifest-backed')}`"
        )
        lines.append(
            f"- Representative manifest: `{inputs['representative_manifest']}`"
        )
        lines.append(f"- Stratified manifest: `{inputs['stratified_manifest']}`")
    lines.extend(
        [
            "",
            "## Heatmap scales",
            (
                "- `fig_heatmap_doc_count`: Viridis on "
                f"`{scales['fig_heatmap_doc_count']['transform']}`; positive range "
                f"{_format_range(scales['fig_heatmap_doc_count'])}; empty bins are gray."
            ),
            (
                "- `fig_heatmap_token_count`: Viridis on "
                f"`{scales['fig_heatmap_token_count']['transform']}`; positive range "
                f"{_format_range(scales['fig_heatmap_token_count'])}; empty bins are gray."
            ),
            (
                "- `fig_stratified_vs_representative`: shared Viridis on "
                f"`{scales['fig_stratified_vs_representative']['transform']}` "
                f"across both panels; positive range "
                f"{_format_range(scales['fig_stratified_vs_representative'])}; "
                "empty bins are gray."
            ),
            (
                "- `fig_sampling_difference`: diverging scale on "
                f"`{scales['fig_sampling_difference']['transform']}`; red means "
                "over-represented in the stratified sample, blue means "
                "under-represented."
            ),
            "",
            "## Reproduction",
            "- Run the report locally or inside Slurm with:",
            "```bash",
            inputs["reproduction_command"],
            "```",
            "",
            "## Comparison interpretation",
        ]
    )
    if inputs["used_dummy"]:
        lines.append(
            "- Comparison interpretation is based on dummy manifests and is not "
            "paper-ready."
        )
    else:
        lines.append("- Most over-represented bins in stratified:")
        lines.extend(_format_bin(row) for row in comparison["most_overrepresented"])
        lines.append("- Most under-represented bins in stratified:")
        lines.extend(_format_bin(row) for row in comparison["most_underrepresented"])
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note_path


__all__ = ["write_output_note"]
