"""Concentration metrics table for paper appendix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_METRIC_LABELS = {
    "total_bins": "Total bins (24 x 24)",
    "empty_bins": "Empty bins (0 docs)",
    "near_empty_100": "Near-empty bins (<100 docs)",
    "near_empty_1000": "Near-empty bins (<1000 docs)",
    "gini_doc": "Gini coefficient (documents)",
    "gini_token": "Gini coefficient (tokens)",
    "effective_bins_doc": "Effective bins, documents (exp H)",
    "effective_bins_token": "Effective bins, tokens (exp H)",
    "top_10_token_share": "Top-10 bin token share",
    "top_50_token_share": "Top-50 bin token share",
    "top_100_token_share": "Top-100 bin token share",
}

_FORMAT_SPECS = {
    "total_bins": "{:,}",
    "empty_bins": "{:,}",
    "near_empty_100": "{:,}",
    "near_empty_1000": "{:,}",
    "gini_doc": "{:.4f}",
    "gini_token": "{:.4f}",
    "effective_bins_doc": "{:.1f}",
    "effective_bins_token": "{:.1f}",
    "top_10_token_share": "{:.1%}",
    "top_50_token_share": "{:.1%}",
    "top_100_token_share": "{:.1%}",
}


def build_table(stats: dict) -> pd.DataFrame:
    rows = []
    for key, label in _METRIC_LABELS.items():
        value = stats[key]
        fmt = _FORMAT_SPECS.get(key, "{}")
        rows.append({"Metric": label, "Value": fmt.format(value)})
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "table_concentration.csv"
    df.to_csv(path, index=False)
    return path


def write_latex(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "table_concentration.tex"
    latex = df.to_latex(index=False, column_format="lr", escape=True)
    path.write_text(latex)
    return path


__all__ = [
    "build_table",
    "write_csv",
    "write_latex",
]
