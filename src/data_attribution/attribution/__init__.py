"""Attribution utilities."""

from __future__ import annotations

import importlib

__all__ = [
    "AttributionRunConfig",
    "RunningStats",
    "compute_topk_streaming",
    "load_metadata",
    "load_query_gradients",
    "persist_outputs",
    "run_and_store",
    "run_attribution",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "AttributionRunConfig": ("types", "AttributionRunConfig"),
    "RunningStats": ("influence", "RunningStats"),
    "compute_topk_streaming": ("influence", "compute_topk_streaming"),
    "load_metadata": ("metadata", "load_metadata"),
    "load_query_gradients": ("gradients", "load_query_gradients"),
    "persist_outputs": ("records", "persist_outputs"),
    "run_and_store": ("run", "run_and_store"),
    "run_attribution": ("run", "run_attribution"),
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
