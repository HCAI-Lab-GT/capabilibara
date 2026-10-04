"""Deprecated compatibility wrapper for manifest-backed sampling."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from dolma.constants import (
    BIN_EMPTY,
    BIN_FULL,
    BIN_PARTIAL,
    BIN_SPARSE,
    FORMATS,
    TARGET_DOCS_PER_BIN,
    TOKEN_FLOOR_PER_BIN,
    TOPICS,
)
from dolma.pool_sample.manifest_cli import main as manifest_sample_main
from dolma.pool_sample.manifest_cli import parse_args
from dolma.pool_sample.sampling import (
    REPRESENTATIVE_TOKEN_TARGET,
    SEED,
    compute_bin_stats,
    generate_dummy_manifest,
    representative_sample,
    stratified_sample,
)

DEPRECATION_MESSAGE = (
    "dolma.pool_sample.stratified is deprecated. "
    "Use dolma.pool_sample.manifest_cli or the data-attribution-manifest-sample CLI."
)


def main(argv: Sequence[str] | None = None) -> int:
    warnings.warn(DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)
    return manifest_sample_main(argv)


__all__ = [
    "BIN_EMPTY",
    "BIN_FULL",
    "BIN_PARTIAL",
    "BIN_SPARSE",
    "DEPRECATION_MESSAGE",
    "FORMATS",
    "REPRESENTATIVE_TOKEN_TARGET",
    "SEED",
    "TARGET_DOCS_PER_BIN",
    "TOKEN_FLOOR_PER_BIN",
    "TOPICS",
    "compute_bin_stats",
    "generate_dummy_manifest",
    "main",
    "parse_args",
    "representative_sample",
    "stratified_sample",
]


if __name__ == "__main__":
    raise SystemExit(main())
