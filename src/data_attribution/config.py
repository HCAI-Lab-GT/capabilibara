"""Project configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_HF_CACHE_ROOT = PROJECT_ROOT / ".hf_cache"
ALLOWED_MODEL_IDS = (
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-1125-32B",
    "apple/DCLM-7B",
    "common-pile/comma-v0.1-1t",
    "common-pile/comma-v0.1-2t",
)
DEFAULT_MODEL_ID = ALLOWED_MODEL_IDS[0]
DEPRECATED_TRANSFORMERS_CACHE_VARS = (
    "PYTORCH_PRETRAINED_BERT_CACHE",
    "PYTORCH_TRANSFORMERS_CACHE",
    "TRANSFORMERS_CACHE",
)


@dataclass(frozen=True)
class HFCachePaths:
    root: Path
    hub: Path
    datasets: Path
    transformers: Path
    xet: Path


def _nonempty_environment(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def resolve_hf_cache_root(cache_root: Path | None = None) -> Path:
    if cache_root is not None:
        return Path(cache_root)
    env_root = _nonempty_environment("HF_CACHE_ROOT")
    if env_root:
        return Path(env_root)
    hf_home = _nonempty_environment("HF_HOME")
    if hf_home:
        return Path(hf_home)
    return DEFAULT_HF_CACHE_ROOT


def resolve_hf_datasets_cache(cache_dir: Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    env_cache = _nonempty_environment("HF_DATASETS_CACHE")
    if env_cache:
        return Path(env_cache)
    return Path.home() / ".cache" / "huggingface" / "datasets"


def configure_hf_cache(
    cache_root: Path | None = None, *, create_dirs: bool = False
) -> HFCachePaths:
    root = resolve_hf_cache_root(cache_root).expanduser()
    if cache_root is None:
        modern_hub = _nonempty_environment("HF_HUB_CACHE")
        legacy_hub = _nonempty_environment("HUGGINGFACE_HUB_CACHE")
        if modern_hub and legacy_hub:
            if Path(modern_hub).expanduser() != Path(legacy_hub).expanduser():
                raise ValueError(
                    "HF_HUB_CACHE and HUGGINGFACE_HUB_CACHE must match when both "
                    "are set"
                )
        hub = Path(modern_hub or legacy_hub or root / "hub").expanduser()
        datasets = Path(
            _nonempty_environment("HF_DATASETS_CACHE") or root / "datasets"
        ).expanduser()
        xet = Path(_nonempty_environment("HF_XET_CACHE") or root / "xet").expanduser()
    else:
        hub = root / "hub"
        datasets = root / "datasets"
        xet = root / "xet"
    paths = HFCachePaths(
        root=root,
        hub=hub,
        datasets=datasets,
        transformers=root / "transformers",
        xet=xet,
    )

    cache_environment = {
        "HF_HOME": paths.root,
        "HF_HUB_CACHE": paths.hub,
        "HUGGINGFACE_HUB_CACHE": paths.hub,
        "HF_DATASETS_CACHE": paths.datasets,
        "HF_XET_CACHE": paths.xet,
    }
    for name, path in cache_environment.items():
        if cache_root is not None or _nonempty_environment(name) is None:
            os.environ[name] = str(path)
    for env_var in DEPRECATED_TRANSFORMERS_CACHE_VARS:
        os.environ.pop(env_var, None)

    if create_dirs:
        for path in (
            paths.root,
            paths.hub,
            paths.datasets,
            paths.transformers,
            paths.xet,
        ):
            path.mkdir(parents=True, exist_ok=True)

    return paths


__all__ = [
    "ALLOWED_MODEL_IDS",
    "DEFAULT_HF_CACHE_ROOT",
    "DEFAULT_MODEL_ID",
    "DEPRECATED_TRANSFORMERS_CACHE_VARS",
    "PROJECT_ROOT",
    "HFCachePaths",
    "configure_hf_cache",
    "resolve_hf_cache_root",
    "resolve_hf_datasets_cache",
]
