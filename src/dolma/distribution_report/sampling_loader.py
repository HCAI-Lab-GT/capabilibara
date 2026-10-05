"""Load and normalize real or dummy sampling manifests."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from dolma.constants import FORMATS, TOPICS
from dolma.manifest_paths import resolve_manifest_parquet_paths
from dolma.manifest_fields import compute_bin_id

logger = logging.getLogger(__name__)

TYPO_LABEL = "__label__electronics_and_hardare"
CANONICAL_LABEL = "__label__electronics_and_hardware"
CANONICAL_TOPIC_LABELS = tuple(f"__label__{topic}" for topic in TOPICS)
CANONICAL_FORMAT_LABELS = tuple(f"__label__{format_label}" for format_label in FORMATS)


def normalize_taxonomy_label(label: object) -> str:
    if pd.isna(label):
        raise ValueError("Encountered null taxonomy label in manifest.")
    normalized = str(label).strip().lower().replace(" ", "_")
    if not normalized:
        raise ValueError("Encountered empty taxonomy label in manifest.")
    if not normalized.startswith("__label__"):
        normalized = f"__label__{normalized}"
    return normalized.replace(TYPO_LABEL, CANONICAL_LABEL)


def _resolve_sampling_column(df: pd.DataFrame, *candidates: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Expected one of {candidates} in sampling manifest.")


def _drop_unsampleable_rows(
    df: pd.DataFrame,
    *,
    required_columns: tuple[str, ...],
    manifest_path: Path,
) -> pd.DataFrame:
    missing_mask = df.loc[:, required_columns].isnull().any(axis=1)
    if not missing_mask.any():
        return df
    dropped = int(missing_mask.sum())
    logger.warning(
        "Dropping %d unsampleable rows from %s due to null values in %s",
        dropped,
        manifest_path,
        ", ".join(required_columns),
    )
    return df.loc[~missing_mask].copy()


def _validate_labels(
    values: pd.Series,
    *,
    allowed: tuple[str, ...],
    label_name: str,
    manifest_path: Path,
) -> None:
    unknown = sorted(set(values.unique()) - set(allowed))
    if unknown:
        preview = ", ".join(unknown[:5])
        raise ValueError(f"Unknown {label_name} labels in {manifest_path}: {preview}")


def _load_manifest_frame(manifest_path: Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    paths = resolve_manifest_parquet_paths(manifest_path)
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _canonical_sampling_label(label: object) -> str:
    return normalize_taxonomy_label(label).removeprefix("__label__")


def load_sampleable_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    df = _load_manifest_frame(manifest_path)
    topic_col = _resolve_sampling_column(df, "weborganizer_topic", "topic")
    format_col = _resolve_sampling_column(df, "weborganizer_format", "format")
    token_col = _resolve_sampling_column(df, "token_count", "estimated_token_count")
    required = ("doc_id", token_col, topic_col, format_col)
    filter_columns = required + (("bin_id",) if "bin_id" in df.columns else ())
    manifest = _drop_unsampleable_rows(
        df.copy(),
        required_columns=filter_columns,
        manifest_path=manifest_path,
    )
    if manifest.empty:
        raise ValueError(
            f"No sampleable rows found in sampling manifest: {manifest_path}"
        )
    manifest["token_count"] = pd.to_numeric(manifest[token_col], errors="raise").astype(
        int
    )
    manifest["topic"] = manifest[topic_col].map(_canonical_sampling_label)
    manifest["format"] = manifest[format_col].map(_canonical_sampling_label)
    if "bin_id" in manifest.columns:
        manifest["bin_id"] = pd.to_numeric(manifest["bin_id"], errors="raise").astype(
            int
        )
    else:
        manifest["bin_id"] = manifest.apply(
            lambda row: compute_bin_id(row["topic"], row["format"]),
            axis=1,
        )
    _validate_labels(
        manifest["topic"].map(normalize_taxonomy_label),
        allowed=CANONICAL_TOPIC_LABELS,
        label_name="topic",
        manifest_path=manifest_path,
    )
    _validate_labels(
        manifest["format"].map(normalize_taxonomy_label),
        allowed=CANONICAL_FORMAT_LABELS,
        label_name="format",
        manifest_path=manifest_path,
    )
    return manifest


def load_sampling_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = load_sampleable_manifest(manifest_path)
    manifest = manifest.loc[:, ["doc_id", "token_count", "topic", "format"]].copy()
    manifest = manifest.rename(
        columns={"topic": "topic_label", "format": "format_label"}
    )
    manifest["topic_label"] = manifest["topic_label"].map(normalize_taxonomy_label)
    manifest["format_label"] = manifest["format_label"].map(normalize_taxonomy_label)
    _validate_labels(
        manifest["topic_label"],
        allowed=CANONICAL_TOPIC_LABELS,
        label_name="topic",
        manifest_path=manifest_path,
    )
    _validate_labels(
        manifest["format_label"],
        allowed=CANONICAL_FORMAT_LABELS,
        label_name="format",
        manifest_path=manifest_path,
    )
    return manifest


def build_sampling_grid(manifest_df: pd.DataFrame) -> pd.DataFrame:
    grouped = manifest_df.groupby(["topic_label", "format_label"], as_index=False).agg(
        doc_count=("doc_id", "size"), token_count=("token_count", "sum")
    )
    from dolma.distribution_report.data_loader import build_grid

    return build_grid(grouped)


def load_sampling_grid(manifest_path: Path) -> pd.DataFrame:
    return build_sampling_grid(load_sampling_manifest(manifest_path))


__all__ = [
    "CANONICAL_FORMAT_LABELS",
    "CANONICAL_TOPIC_LABELS",
    "build_sampling_grid",
    "load_sampleable_manifest",
    "load_sampling_grid",
    "load_sampling_manifest",
    "normalize_taxonomy_label",
]
