"""Dolma data utilities and enrichment tooling."""

import importlib

__all__ = [
    "DEFAULT_SPLIT",
    "DOLMA_6T_MIX_DATASET_ID",
    "DOLMA_DATASET_ID",
    "DOLMA_FIELDS",
    "SampleManifest",
    "approximate_token_count",
    "dolma_builder",
    "ensure_metadata",
    "materialize_sample",
    "prepare_dolma_shards",
    "project_record",
    "sample_documents",
    "stream_dolma",
    "write_jsonl",
    "write_manifest",
    "write_parquet",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "DOLMA_6T_MIX_DATASET_ID": ("constants", "DOLMA_6T_MIX_DATASET_ID"),
    "DOLMA_DATASET_ID": ("constants", "DOLMA_DATASET_ID"),
    "DOLMA_FIELDS": ("constants", "DOLMA_FIELDS"),
    "DEFAULT_SPLIT": ("constants", "DEFAULT_SPLIT"),
    "SampleManifest": ("outputs", "SampleManifest"),
    "write_jsonl": ("outputs", "write_jsonl"),
    "write_manifest": ("outputs", "write_manifest"),
    "write_parquet": ("outputs", "write_parquet"),
    "approximate_token_count": ("sample", "approximate_token_count"),
    "ensure_metadata": ("sample", "ensure_metadata"),
    "project_record": ("sample", "project_record"),
    "sample_documents": ("sample", "sample_documents"),
    "dolma_builder": ("shards", "dolma_builder"),
    "prepare_dolma_shards": ("shards", "prepare_dolma_shards"),
    "stream_dolma": ("shards", "stream_dolma"),
    "materialize_sample": ("materialize", "materialize_sample"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
