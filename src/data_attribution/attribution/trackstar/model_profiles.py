"""Immutable model/corpus contracts for TrackStar score spaces."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal


class ProfileMismatchError(ValueError):
    """Raised when TrackStar components do not share one score-space contract."""


@dataclass(frozen=True, slots=True)
class TrackStarModelProfile:
    id: str
    model_id: str
    model_revision: str
    revision_match: Literal["exact", "prefix"]
    query_model_id: str
    corpus_scope: str
    corpus_revision: str
    sample_manifest_id: str
    label_variant: str
    variant: str
    prompt_protocol: str
    prompt_column: str
    native_context_length: int
    evaluation_max_length: int
    precision: str
    projection_dim: int
    token_batch: int
    fsdp_policy: str
    index_artifact_id: str
    preconditioner_artifact_id: str
    query_processor_artifact_id: str
    mixed_preconditioner_artifact_id: str
    total_shard_slots: int | None
    expected_realized_shards: int | None
    excluded_shards: tuple[str, ...]
    unit_normalize: bool
    truncation: bool
    index_preconditioned: bool
    query_preconditioned: bool
    score_precondition: bool
    shard_document_exceptions: tuple[tuple[str, int], ...] = ()
    execution_gated: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def matches_resolved_revision(self, resolved_revision: str) -> bool:
        if self.revision_match == "exact":
            return resolved_revision == self.model_revision
        return resolved_revision.startswith(self.model_revision)


def _profile(
    *,
    id: str,
    model_id: str,
    model_revision: str,
    revision_match: Literal["exact", "prefix"],
    corpus_scope: str,
    corpus_revision: str,
    sample_manifest_id: str,
    label_variant: str,
    variant: str,
    native_context_length: int,
    index_artifact_id: str,
    preconditioner_artifact_id: str,
    total_shard_slots: int | None,
    evaluation_max_length: int = 8192,
    fsdp_policy: str = "disabled",
    shard_document_exceptions: tuple[tuple[str, int], ...] = (),
    execution_gated: bool = False,
) -> TrackStarModelProfile:
    return TrackStarModelProfile(
        id=id,
        model_id=model_id,
        model_revision=model_revision,
        revision_match=revision_match,
        query_model_id=model_id,
        corpus_scope=corpus_scope,
        corpus_revision=corpus_revision,
        sample_manifest_id=sample_manifest_id,
        label_variant=label_variant,
        variant=variant,
        prompt_protocol="finance0 cloze",
        prompt_column="text",
        native_context_length=native_context_length,
        evaluation_max_length=evaluation_max_length,
        precision="fp32",
        projection_dim=16,
        token_batch=4096,
        fsdp_policy=fsdp_policy,
        index_artifact_id=index_artifact_id,
        preconditioner_artifact_id=preconditioner_artifact_id,
        query_processor_artifact_id=f"{preconditioner_artifact_id}#query_preconditioner",
        mixed_preconditioner_artifact_id=(
            f"{preconditioner_artifact_id}#mixed_preconditioner"
        ),
        total_shard_slots=total_shard_slots,
        expected_realized_shards=total_shard_slots,
        excluded_shards=(),
        unit_normalize=True,
        truncation=True,
        index_preconditioned=True,
        query_preconditioned=True,
        score_precondition=False,
        shard_document_exceptions=shard_document_exceptions,
        execution_gated=execution_gated,
    )


_PROFILES = {
    profile.id: profile
    for profile in (
        _profile(
            id="olmo3_7b_dolma3",
            model_id="allenai/Olmo-3-1025-7B",
            model_revision="a81bae42db3975be1671e27b9c9a56da1a9f980f",
            revision_match="exact",
            corpus_scope="Dolma3 sample_10000_docs 5678621-document working sample",
            corpus_revision="20260326T163642Z_1102443",
            sample_manifest_id="sample_10000_docs/working_sample_manifest.parquet",
            label_variant="base",
            variant="base",
            native_context_length=8192,
            index_artifact_id="HCAI-Lab/trackstar-gradient-index-base",
            preconditioner_artifact_id=(
                "HCAI-Lab/trackstar-preconditioners/olmo-3-1025-7b"
            ),
            total_shard_slots=316,
        ),
        _profile(
            id="comma_2t_common_pile",
            model_id="common-pile/comma-v0.1-2t",
            model_revision="3fba893",
            revision_match="prefix",
            corpus_scope="Common Pile v0.1 2T released-mixture working sample",
            corpus_revision="Common Pile v0.1",
            sample_manifest_id="comma_working_sample_manifest_v1",
            label_variant="comma_2t",
            variant="comma_2t",
            native_context_length=16384,
            index_artifact_id="HCAI-Lab/comma-v01-2t-gradient-index",
            preconditioner_artifact_id="HCAI-Lab/comma-v01-2t-preconditioner",
            total_shard_slots=196,
        ),
        _profile(
            id="olmo3_32b_dolma3",
            model_id="allenai/Olmo-3-1125-32B",
            model_revision="c2b61da",
            revision_match="prefix",
            corpus_scope="Dolma3 sample_10000_docs with documented shard_0042 exception",
            corpus_revision="20260326T163642Z_1102443",
            sample_manifest_id="sample_10000_docs/working_sample_manifest.parquet",
            label_variant="base_32b",
            variant="base_32b",
            native_context_length=8192,
            index_artifact_id=(
                "HCAI-Lab/trackstar-gradient-index-olmo3-32b-base-preconditioned"
            ),
            preconditioner_artifact_id="32B mixed preconditioner",
            total_shard_slots=316,
            fsdp_policy="required",
            shard_document_exceptions=(("shard_0042", 17_969),),
            execution_gated=True,
        ),
        _profile(
            id="dclm_7b_released_mixture",
            model_id="apple/DCLM-7B",
            model_revision="c85bfa16",
            revision_match="prefix",
            corpus_scope=(
                "DCLM-7B released mixture: DCLM-Baseline + StarCoderData + ProofPile2"
            ),
            corpus_revision="dclm-7b-released-mixture-v1",
            sample_manifest_id="dclm_released_mixture_working_sample_v1",
            label_variant="dclm_base",
            variant="dclm_base",
            native_context_length=2048,
            evaluation_max_length=2048,
            index_artifact_id=(
                "urn:social-data-attribution:dclm-7b-released-mixture-gradient-index"
            ),
            preconditioner_artifact_id=(
                "urn:social-data-attribution:dclm-7b-released-mixture-preconditioner"
            ),
            total_shard_slots=None,
            execution_gated=True,
        ),
    )
}


_SCORE_SPACE_FIELDS = (
    "model_id",
    "model_revision",
    "revision_match",
    "corpus_scope",
    "corpus_revision",
    "sample_manifest_id",
    "index_artifact_id",
    "preconditioner_artifact_id",
    "prompt_protocol",
    "prompt_column",
    "evaluation_max_length",
    "token_batch",
    "variant",
    "label_variant",
    "query_model_id",
    "precision",
    "projection_dim",
    "unit_normalize",
    "truncation",
    "index_preconditioned",
    "query_preconditioned",
    "query_processor_artifact_id",
    "mixed_preconditioner_artifact_id",
    "score_precondition",
    "total_shard_slots",
    "expected_realized_shards",
    "excluded_shards",
    "shard_document_exceptions",
)


def get_model_profile(profile_id: str) -> TrackStarModelProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        known = ", ".join(sorted(_PROFILES))
        raise KeyError(
            f"unknown TrackStar model profile {profile_id!r}; known: {known}"
        ) from exc


def model_profile_variants(*, include_gated: bool = True) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                profile.variant
                for profile in _PROFILES.values()
                if include_gated or not profile.execution_gated
            }
        )
    )


def assert_profile_compatible(
    expected: TrackStarModelProfile,
    actual: TrackStarModelProfile,
) -> None:
    mismatches = [
        field
        for field in _SCORE_SPACE_FIELDS
        if getattr(expected, field) != getattr(actual, field)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}={getattr(actual, field)!r} (expected {getattr(expected, field)!r})"
            for field in mismatches
        )
        raise ProfileMismatchError(f"TrackStar profile mismatch: {details}")


def _load_verified_resolution(
    profile: TrackStarModelProfile, resolution_manifest: Path
) -> tuple[str, Path]:
    path = Path(resolution_manifest)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileMismatchError(
            f"invalid model resolution manifest: {path}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ProfileMismatchError("model resolution manifest has invalid schema")
    expected = {
        "profile_id": profile.id,
        "profile_digest": profile.digest,
        "logical_model_id": profile.model_id,
        "requested_revision": profile.model_revision,
        "index_artifact_id": profile.index_artifact_id,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ProfileMismatchError(
                f"model resolution {field} does not match profile"
            )
    index_metadata_digest = payload.get("index_metadata_digest")
    if (
        not isinstance(index_metadata_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", index_metadata_digest) is None
    ):
        raise ProfileMismatchError(
            "model resolution index_metadata_digest is missing or invalid"
        )
    index_metadata_revision = payload.get("index_metadata_revision")
    if not isinstance(index_metadata_revision, str) or not index_metadata_revision:
        raise ProfileMismatchError(
            "model resolution index_metadata_revision is missing or invalid"
        )
    resolved_at = payload.get("resolved_at")
    try:
        resolved_timestamp = datetime.fromisoformat(str(resolved_at))
    except ValueError as exc:
        raise ProfileMismatchError(
            "model resolution resolved_at is missing or invalid"
        ) from exc
    if not isinstance(resolved_at, str) or resolved_timestamp.tzinfo is None:
        raise ProfileMismatchError("model resolution resolved_at is missing or invalid")
    resolved_sha = payload.get("resolved_full_sha")
    if (
        not isinstance(resolved_sha, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", resolved_sha) is None
        or not profile.matches_resolved_revision(resolved_sha)
    ):
        raise ProfileMismatchError("model resolution full SHA does not match profile")
    snapshot_value = payload.get("immutable_local_snapshot_path")
    if not isinstance(snapshot_value, str):
        raise ProfileMismatchError("model resolution snapshot path is missing")
    snapshot = Path(snapshot_value)
    if (
        not snapshot.is_absolute()
        or not snapshot.is_dir()
        or snapshot.name != resolved_sha
    ):
        raise ProfileMismatchError(
            "model resolution snapshot is not an immutable full-SHA directory"
        )
    return resolved_sha, snapshot


def profile_execution_environment(
    profile: TrackStarModelProfile, resolution_manifest: Path
) -> dict[str, str]:
    """Return launcher values bound to a verified immutable local snapshot."""

    resolved_sha, snapshot = _load_verified_resolution(profile, resolution_manifest)

    return {
        "MODEL": str(snapshot),
        "LOGICAL_MODEL_ID": profile.model_id,
        "MODEL_REVISION": resolved_sha,
        "REQUESTED_MODEL_REVISION": profile.model_revision,
        "VARIANT": profile.variant,
        "PROMPT_COLUMN": profile.prompt_column,
        "PRECISION": profile.precision,
        "PROJECTION_DIM": str(profile.projection_dim),
        "TOKEN_BATCH": str(profile.token_batch),
        "FSDP": "1" if profile.fsdp_policy == "required" else "0",
    }


__all__ = [
    "ProfileMismatchError",
    "TrackStarModelProfile",
    "assert_profile_compatible",
    "get_model_profile",
    "model_profile_variants",
    "profile_execution_environment",
]
