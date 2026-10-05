"""Typed evidence records for controlled fused scores."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreShardSummary:
    shard_id: str
    row_count: int
    finite_value_count: int
    nonzero_value_count: int
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ScoreArtifactEvidence:
    run_spec_sha256: str
    run_spec_file_sha256: str
    checkpoint_role: str
    probe: str
    condition: str
    objective: str
    study_role: str
    producer: str
    reduction: str
    query_count: int
    query_order_sha256: str
    query_gradient_artifact_sha256: str
    query_gradient_evidence_sha256: str
    rendered_artifact_sha256: str
    rendered_bundle_sha256: str
    index_row_count: int
    corpus_sha256: str
    shard_order_sha256: str
    index_ordered_document_id_sha256: str
    shard_ids: tuple[str, ...]
    ordered_document_id_sha256: str
    document_metadata_file_sha256: str
    ranking_policy_sha256: str
    score_path: str
    score_uri: str
    score_byte_count: int
    score_file_sha256: str
    row_count: int
    finite_value_count: int
    nonzero_value_count: int
    shard_summaries: tuple[ScoreShardSummary, ...]
    evidence_sha256: str
    schema_version: str = "1"
    kind: str = "trackstar_fused_score"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shard_ids", tuple(self.shard_ids))
        object.__setattr__(self, "shard_summaries", tuple(self.shard_summaries))


__all__ = ["ScoreArtifactEvidence", "ScoreShardSummary"]
