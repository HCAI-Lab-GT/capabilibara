"""Dataclasses for attribution runs."""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Mapping, Sequence

import torch


@dataclass(frozen=True)
class QueryGradient:
    query_id: str
    gradient: torch.Tensor
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class AttributionRunConfig:
    """Configuration for an attribution sweep."""

    run_id: str
    query_gradients: Path
    index_path: Path
    output_dir: Path
    metadata_path: Path | None = None
    output_format: str = "parquet"
    top_k: int = 10
    query_id_field: str = "query_id"
    doc_id_field: str = "doc_id"
    metadata_join_keys: Sequence[str] = field(default_factory=tuple)
    batch_size: int = 100
    device: str | None = None
    unit_norm: bool = True
    started_at: _dt.datetime = field(
        default_factory=lambda: _dt.datetime.now(tz=_dt.UTC)
    )
    config: Mapping[str, object] = field(default_factory=dict)

    def to_serializable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["started_at"] = self.started_at.isoformat()
        payload["query_gradients"] = str(self.query_gradients)
        payload["index_path"] = str(self.index_path)
        payload["output_dir"] = str(self.output_dir)
        if self.metadata_path is not None:
            payload["metadata_path"] = str(self.metadata_path)
        payload["config"] = json.loads(json.dumps(self.config, default=str))
        return payload


__all__ = ["AttributionRunConfig", "QueryGradient"]
