"""Immutable records for controlled query-condition artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from data_attribution.artifact_integrity import sha256_json

type QueryCondition = Literal["model_completion", "reference_answer", "prompt_only"]
type QueryObjective = Literal[
    "completion_target", "joint_sequence", "prompt_next_token"
]
type StudyRole = Literal["primary", "sensitivity", "replication"]
type QueryRendering = Literal["canonical_plain"]
type CompletionTiePolicy = Literal["reject", "lowest_candidate_index"]
type RenderingSetting = str | int | bool | None

SCHEMA_VERSION = "1"
QUERY_CONDITIONS: tuple[QueryCondition, ...] = (
    "model_completion",
    "reference_answer",
    "prompt_only",
)
QUERY_OBJECTIVES: tuple[QueryObjective, ...] = (
    "completion_target",
    "joint_sequence",
    "prompt_next_token",
)
STUDY_ROLES: tuple[StudyRole, ...] = ("primary", "sensitivity", "replication")
QUERY_RENDERINGS: tuple[QueryRendering, ...] = ("canonical_plain",)
COMPLETION_TIE_POLICIES: tuple[CompletionTiePolicy, ...] = (
    "reject",
    "lowest_candidate_index",
)


def _freeze_mapping[T](values: Mapping[str, T]) -> Mapping[str, T]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class ArtifactLineage:
    artifact_id: str
    uri: str
    sha256: str
    revision: str | None = None


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_index: int
    continuation: str
    scores: Mapping[str, float]
    token_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", _freeze_mapping(self.scores))


@dataclass(frozen=True)
class QueryEvidence:
    query_id: str
    probe: str
    task_core: str
    native_id: str
    prompt: str
    candidates: tuple[CandidateEvidence, ...]
    model_id: str
    model_revision: str
    task_revision: str
    lineage: tuple[ArtifactLineage, ...]
    document_id: str | None = None
    reference_index: int | None = None
    correctness: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "lineage", tuple(self.lineage))
        object.__setattr__(self, "correctness", _freeze_mapping(self.correctness))


@dataclass(frozen=True)
class ReferenceSourceAnchor:
    revision: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if type(self.revision) is not str or not self.revision.strip():
            raise ValueError("Reference source requires a nonempty revision")
        if (
            type(self.artifact_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.artifact_sha256) is None
        ):
            raise ValueError("Reference source requires a lowercase 64-hex SHA-256")


@dataclass(frozen=True)
class ReferenceInputBase:
    query_id: str
    source: ReferenceSourceAnchor

    def __post_init__(self) -> None:
        if type(self.query_id) is not str or not self.query_id.strip():
            raise ValueError("Reference input requires a query ID")
        if type(self.source) is not ReferenceSourceAnchor:
            raise ValueError("Reference input requires a source trust anchor")


@dataclass(frozen=True)
class ParsedAnswerReferenceInput(ReferenceInputBase):
    source_answer: str
    ordered_source_choices: tuple[str, ...]
    corroborating_answer: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_string_tuple(self.ordered_source_choices, "ordered source choices")


@dataclass(frozen=True)
class EthicsOrderingReferenceInput(ReferenceInputBase):
    roles: tuple[str, ...]
    ordered_source_values: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_string_tuple(self.roles, "ETHICS roles")
        _require_string_tuple(self.ordered_source_values, "ETHICS source values")


@dataclass(frozen=True)
class HumanConsensusReferenceInput(ReferenceInputBase):
    orientation: tuple[str, ...]
    human_response: float
    threshold: float
    comparator: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_string_tuple(self.orientation, "MoralExcept orientation")


@dataclass(frozen=True)
class ExplicitIndexReferenceInput(ReferenceInputBase):
    source_key: str
    source_index: int


@dataclass(frozen=True)
class UnavailableReferenceInput(ReferenceInputBase):
    reason: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("Unavailable reference input requires a reason")


@dataclass(frozen=True)
class ReferenceSelection:
    query_id: str
    candidate_index: int
    continuation: str
    basis: str
    source: ReferenceSourceAnchor


@dataclass(frozen=True)
class ReferenceExclusion:
    query_id: str
    probe: str
    reason: str
    reference_policy: str
    source: ReferenceSourceAnchor


type ReferenceInput = (
    ParsedAnswerReferenceInput
    | EthicsOrderingReferenceInput
    | HumanConsensusReferenceInput
    | ExplicitIndexReferenceInput
    | UnavailableReferenceInput
)

REFERENCE_INPUT_KINDS = (
    "parsed_answer",
    "ethics_ordering",
    "human_consensus",
    "explicit_index",
    "unavailable",
)


def reference_input_snapshot(
    source: ReferenceInput,
) -> tuple[str, dict[str, object]]:
    if type(source) is ParsedAnswerReferenceInput:
        return "parsed_answer", {
            "query_id": source.query_id,
            "source": _source_snapshot(source.source),
            "source_answer": source.source_answer,
            "ordered_source_choices": list(source.ordered_source_choices),
            "corroborating_answer": source.corroborating_answer,
        }
    if type(source) is EthicsOrderingReferenceInput:
        return "ethics_ordering", {
            "query_id": source.query_id,
            "source": _source_snapshot(source.source),
            "roles": list(source.roles),
            "ordered_source_values": list(source.ordered_source_values),
        }
    if type(source) is HumanConsensusReferenceInput:
        return "human_consensus", {
            "query_id": source.query_id,
            "source": _source_snapshot(source.source),
            "orientation": list(source.orientation),
            "human_response": source.human_response,
            "threshold": source.threshold,
            "comparator": source.comparator,
        }
    if type(source) is ExplicitIndexReferenceInput:
        return "explicit_index", {
            "query_id": source.query_id,
            "source": _source_snapshot(source.source),
            "source_key": source.source_key,
            "source_index": source.source_index,
        }
    if type(source) is UnavailableReferenceInput:
        return "unavailable", {
            "query_id": source.query_id,
            "source": _source_snapshot(source.source),
            "reason": source.reason,
        }
    raise ValueError("Unsupported reference input type")


def decode_reference_input(kind: str, payload: Mapping[str, object]) -> ReferenceInput:
    if kind not in REFERENCE_INPUT_KINDS:
        raise ValueError("Unknown serialized reference input kind")
    query_id = _snapshot_text(payload, "query_id")
    source = _decode_source_snapshot(payload)
    if kind == "parsed_answer":
        _snapshot_keys(
            payload,
            {
                "query_id",
                "source",
                "source_answer",
                "ordered_source_choices",
                "corroborating_answer",
            },
        )
        return ParsedAnswerReferenceInput(
            query_id,
            source,
            _snapshot_text(payload, "source_answer"),
            _snapshot_string_tuple(payload, "ordered_source_choices"),
            _snapshot_optional_text(payload, "corroborating_answer"),
        )
    if kind == "ethics_ordering":
        _snapshot_keys(
            payload, {"query_id", "source", "roles", "ordered_source_values"}
        )
        return EthicsOrderingReferenceInput(
            query_id,
            source,
            _snapshot_string_tuple(payload, "roles"),
            _snapshot_string_tuple(payload, "ordered_source_values"),
        )
    if kind == "human_consensus":
        _snapshot_keys(
            payload,
            {
                "query_id",
                "source",
                "orientation",
                "human_response",
                "threshold",
                "comparator",
            },
        )
        return HumanConsensusReferenceInput(
            query_id,
            source,
            _snapshot_string_tuple(payload, "orientation"),
            _snapshot_number(payload, "human_response"),
            _snapshot_number(payload, "threshold"),
            _snapshot_text(payload, "comparator"),
        )
    if kind == "explicit_index":
        _snapshot_keys(payload, {"query_id", "source", "source_key", "source_index"})
        index = payload.get("source_index")
        if type(index) is not int:
            raise ValueError("Serialized reference index must be an integer")
        return ExplicitIndexReferenceInput(
            query_id, source, _snapshot_text(payload, "source_key"), index
        )
    _snapshot_keys(payload, {"query_id", "source", "reason"})
    return UnavailableReferenceInput(
        query_id, source, _snapshot_text(payload, "reason")
    )


def _source_snapshot(source: ReferenceSourceAnchor) -> dict[str, str]:
    return {"revision": source.revision, "artifact_sha256": source.artifact_sha256}


def _decode_source_snapshot(payload: Mapping[str, object]) -> ReferenceSourceAnchor:
    value = payload.get("source")
    if not isinstance(value, Mapping):
        raise ValueError("Serialized reference input has no source anchor")
    _snapshot_keys(value, {"revision", "artifact_sha256"})
    return ReferenceSourceAnchor(
        _snapshot_text(value, "revision"), _snapshot_text(value, "artifact_sha256")
    )


def _snapshot_keys(payload: Mapping[str, object], expected: set[str]) -> None:
    if set(payload) != expected:
        raise ValueError("Serialized reference input has invalid fields")


def _snapshot_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if type(value) is not str:
        raise ValueError(f"Serialized reference input field {key!r} must be text")
    return value


def _snapshot_optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and type(value) is not str:
        raise ValueError(f"Serialized reference input field {key!r} must be text")
    return value


def _snapshot_string_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if type(value) not in (list, tuple) or any(
        type(item) is not str for item in cast(list[object] | tuple[object, ...], value)
    ):
        raise ValueError(f"Serialized reference input field {key!r} must be text list")
    return tuple(cast(list[str] | tuple[str, ...], value))


def _snapshot_number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if type(value) not in (int, float):
        raise ValueError(f"Serialized reference input field {key!r} must be numeric")
    return float(cast(int | float, value))


@dataclass(frozen=True)
class ReferenceResolution:
    inputs: tuple[ReferenceInput, ...]
    records: tuple[ReferenceSelection, ...]
    exclusions: tuple[ReferenceExclusion, ...]

    def __post_init__(self) -> None:
        _require_tuple(self.inputs, "reference resolution inputs")
        _require_tuple(self.records, "reference resolution records")
        _require_tuple(self.exclusions, "reference resolution exclusions")


def reference_exclusion_sha256(exclusion: ReferenceExclusion) -> str:
    return sha256_json(exclusion)


def _require_string_tuple(values: tuple[object, ...], label: str) -> None:
    _require_tuple(values, label)
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError(f"{label} must contain nonempty strings")


def _require_tuple(values: object, label: str) -> None:
    if type(values) is not tuple:
        raise ValueError(f"{label} must be an immutable tuple")


@dataclass(frozen=True)
class CompileQueryRequest:
    probe: str
    condition: QueryCondition
    objective: QueryObjective
    evidence: tuple[QueryEvidence, ...]
    selection_rule: str | None = None
    reference_inputs: tuple[ReferenceInput, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "reference_inputs", tuple(self.reference_inputs))


@dataclass(frozen=True)
class CanonicalQueryRecord:
    query_id: str
    probe: str
    task_core: str
    prompt: str
    target: str | None
    target_source: str | None
    condition: QueryCondition
    selection_rule: str | None
    lineage: tuple[ArtifactLineage, ...]
    base_completion_tie_policy: CompletionTiePolicy
    reference_basis: str | None = None
    reference_index: int | None = None
    reference_continuation: str | None = None
    reference_source_revision: str | None = None
    reference_source_sha256: str | None = None
    reference_input_kind: str | None = None
    reference_input_payload: Mapping[str, object] | None = None
    reference_candidate_continuations: tuple[str, ...] | None = None
    base_completion_index: int | None = None
    base_completion_correct: bool | None = None
    base_completion_model_id: str | None = None
    base_completion_model_revision: str | None = None
    base_completion_selection_rule: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lineage", tuple(self.lineage))
        if self.reference_input_payload is not None:
            if not isinstance(self.reference_input_payload, Mapping):
                raise ValueError("Reference input payload must be a mapping")
            object.__setattr__(
                self,
                "reference_input_payload",
                _freeze_mapping(self.reference_input_payload),
            )
        if self.reference_candidate_continuations is not None:
            object.__setattr__(
                self,
                "reference_candidate_continuations",
                tuple(self.reference_candidate_continuations),
            )


@dataclass(frozen=True)
class CompiledQueryBundle:
    probe: str
    condition: QueryCondition
    objective: QueryObjective
    selection_rule: str | None
    completion_tie_policy: CompletionTiePolicy
    rendering: QueryRendering
    study_role: StudyRole
    records: tuple[CanonicalQueryRecord, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))


@dataclass(frozen=True)
class OffsetOverlapEvidence:
    token_start: int
    token_end: int
    token_ids: tuple[int, ...]
    source_start: int
    source_end: int
    grouped_decode: str
    individual_decodes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "individual_decodes", tuple(self.individual_decodes))


@dataclass(frozen=True)
class RenderedQueryRecord:
    query_id: str
    probe: str
    condition: QueryCondition
    objective: QueryObjective
    selection_rule: str | None
    rendering: QueryRendering
    study_role: StudyRole
    canonical_record_sha256: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    offset_overlap_evidence: tuple[OffsetOverlapEvidence, ...]
    special_tokens_mask: tuple[int, ...]
    length: int
    loss_token_start: int
    loss_token_end: int
    prompt_character_length: int
    text_character_length: int
    tokenizer_id: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_ids", tuple(self.input_ids))
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(
            self, "offset_mapping", tuple(tuple(pair) for pair in self.offset_mapping)
        )
        object.__setattr__(
            self, "offset_overlap_evidence", tuple(self.offset_overlap_evidence)
        )
        object.__setattr__(self, "special_tokens_mask", tuple(self.special_tokens_mask))


@dataclass(frozen=True)
class RenderedQueryBundle:
    probe: str
    condition: QueryCondition
    objective: QueryObjective
    selection_rule: str | None
    rendering: QueryRendering
    study_role: StudyRole
    tokenizer_id: str
    tokenizer_revision: str
    compiled_bundle_sha256: str
    rendering_settings: Mapping[str, RenderingSetting]
    records: tuple[RenderedQueryRecord, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rendering_settings", _freeze_mapping(self.rendering_settings)
        )
        object.__setattr__(self, "records", tuple(self.records))


type QueryBundle = CompiledQueryBundle | RenderedQueryBundle
