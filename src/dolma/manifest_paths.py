"""Helpers for reading manifest parquet files or datasets."""

from __future__ import annotations

from pathlib import Path


def resolve_manifest_parquet_paths(manifest_path: str | Path) -> list[Path]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest path not found: {path}")
    if path.is_file():
        return [path]

    data_dir = path / "data"
    if data_dir.is_dir():
        paths = sorted(data_dir.glob("*.parquet"))
        if paths:
            return paths

    paths = sorted(path.glob("*.parquet"))
    if paths:
        return paths

    raise ValueError(f"No parquet files found in directory: {path}")


def read_manifest_pandas(manifest_path: str | Path):
    import pandas as pd

    paths = resolve_manifest_parquet_paths(manifest_path)
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def scan_manifest_polars(manifest_path: str | Path):
    import polars as pl

    paths = [str(path) for path in resolve_manifest_parquet_paths(manifest_path)]
    return pl.scan_parquet(paths)


__all__ = [
    "read_manifest_pandas",
    "resolve_manifest_parquet_paths",
    "scan_manifest_polars",
]
