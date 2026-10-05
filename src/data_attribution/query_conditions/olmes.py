"""Strict OLMES source-manifest loading and query-evidence normalization."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import unquote, urlsplit

from data_attribution.artifact_integrity import (
    parse_json_object,
    require_equal,
    sha256_file,
    sha256_json,
)
from data_attribution.query_conditions.io import (
    count_jsonl_objects,
    object_list_field,
    optional_field,
    read_jsonl_objects,
    require_keys,
    required_field,
)
from data_attribution.query_conditions.types import (
    ArtifactLineage,
    CandidateEvidence,
    QueryEvidence,
)


type EvidenceMode = Literal["raw_requests", "verified_reconstruction"]
type RevisionEvidenceKind = Literal["resolved_snapshot", "loading_content_attestation"]

SOURCE_MANIFEST_VERSION = "1"
CPQC_ROW_COUNTS = {
    "bbh_causal_judgement": 187,
    "bbh_disambiguation_qa": 250,
    "moralexceptqa_rbqa": 148,
}
_HF_CACHE_MODEL_COMPONENT_ALIASES = frozenset(
    {
        (
            "allenai/Olmo-3-1025-7B",
            "a81bae42db3975be1671e27b9c9a56da1a9f980f",
            "models--allenai--OLMo-3-1025-7B",
        )
    }
)


@dataclass(frozen=True)
class SourceArtifact:
    path: Path
    sha256: str
    complete: bool
    record_count: int | None


@dataclass(frozen=True)
class ModelRevisionEvidence:
    kind: RevisionEvidenceKind
    artifact: SourceArtifact


@dataclass(frozen=True)
class ReconstructionEvidence:
    olmes_commit: str
    olmes_patch_hashes: tuple[str, ...]
    dataset_revision: str | None
    local_data_sha256: str | None
    reconstruction_code_sha256: str
    sample_group_count: int
    reconstructed_requests: SourceArtifact
    recorded_inputs: SourceArtifact


@dataclass(frozen=True)
class OlmesTaskSource:
    source_id: str
    probe: str
    task_core: str
    task_alias: str
    evidence_mode: EvidenceMode
    expected_rows: int
    model_id: str
    model_revision: str
    task_hash: str
    model_hash: str
    task_config_sha256: str
    model_config_sha256: str
    predictions: SourceArtifact
    requests: SourceArtifact | None
    metrics: SourceArtifact
    model_revision_evidence: ModelRevisionEvidence
    reconstruction: ReconstructionEvidence | None


_ARTIFACT_FIELDS = frozenset({"path", "sha256", "complete", "record_count"})
_SOURCE_FIELDS = frozenset(
    {
        "source_id",
        "probe",
        "task_core",
        "task_alias",
        "evidence_mode",
        "expected_rows",
        "model_id",
        "model_revision",
        "task_hash",
        "model_hash",
        "task_config_sha256",
        "model_config_sha256",
        "predictions",
        "requests",
        "metrics",
        "model_revision_evidence",
        "reconstruction",
    }
)
_RECONSTRUCTION_FIELDS = frozenset(
    {
        "olmes_commit",
        "olmes_patch_hashes",
        "dataset_revision",
        "local_data_sha256",
        "reconstruction_code_sha256",
        "sample_group_count",
        "reconstructed_requests",
        "recorded_inputs",
    }
)


def decode_source(payload: dict[str, object], root: Path) -> OlmesTaskSource:
    require_keys(payload, _SOURCE_FIELDS, "OLMES source")
    source = OlmesTaskSource(
        source_id=_text(payload, "source_id"),
        probe=_text(payload, "probe"),
        task_core=_text(payload, "task_core"),
        task_alias=_text(payload, "task_alias"),
        evidence_mode=_evidence_mode(payload),
        expected_rows=required_field(payload, "expected_rows", int, "OLMES source"),
        model_id=_text(payload, "model_id"),
        model_revision=_text(payload, "model_revision"),
        task_hash=_text(payload, "task_hash"),
        model_hash=_text(payload, "model_hash"),
        task_config_sha256=_sha_field(payload, "task_config_sha256", "OLMES source"),
        model_config_sha256=_sha_field(payload, "model_config_sha256", "OLMES source"),
        predictions=_artifact_field(payload, "predictions", root),
        requests=_optional_artifact(payload, "requests", root),
        metrics=_artifact_field(payload, "metrics", root),
        model_revision_evidence=_revision_evidence(payload, root),
        reconstruction=_optional_reconstruction(payload, root),
    )
    validate_source_contract(source)
    return source


def _artifact_field(payload: dict[str, object], key: str, root: Path) -> SourceArtifact:
    value = required_field(payload, key, dict, f"{key} artifact")
    require_keys(value, _ARTIFACT_FIELDS, f"{key} artifact")
    path = _resolve(root, required_field(value, "path", str, f"{key} artifact"))
    return SourceArtifact(
        path=path,
        sha256=_sha_field(value, "sha256", f"{key} artifact"),
        complete=required_field(value, "complete", bool, f"{key} artifact"),
        record_count=optional_field(value, "record_count", int, f"{key} artifact"),
    )


def _revision_evidence(payload: dict[str, object], root: Path) -> ModelRevisionEvidence:
    value = payload["model_revision_evidence"]
    if not isinstance(value, dict):
        raise ValueError("OLMES source requires model revision evidence")
    require_keys(value, frozenset({"kind", "artifact"}), "model revision evidence")
    kind = required_field(value, "kind", str, "model revision evidence")
    if kind not in ("resolved_snapshot", "loading_content_attestation"):
        raise ValueError("Invalid model revision evidence kind")
    artifact = _artifact_field(value, "artifact", root)
    return ModelRevisionEvidence(cast(RevisionEvidenceKind, kind), artifact)


def _optional_reconstruction(
    payload: dict[str, object], root: Path
) -> ReconstructionEvidence | None:
    value = payload["reconstruction"]
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Invalid reconstruction evidence")
    require_keys(value, _RECONSTRUCTION_FIELDS, "reconstruction evidence")
    patches = required_field(
        value, "olmes_patch_hashes", list, "reconstruction evidence"
    )
    if not patches or any(not isinstance(item, str) for item in patches):
        raise ValueError("Invalid reconstruction patch hashes")
    dataset = optional_field(value, "dataset_revision", str, "reconstruction evidence")
    local_hash = optional_field(
        value, "local_data_sha256", str, "reconstruction evidence"
    )
    if dataset is not None and not dataset.strip():
        raise ValueError("reconstruction requires a non-empty dataset revision")
    if dataset is None and local_hash is None:
        raise ValueError("reconstruction requires dataset revision or local data hash")
    return _build_reconstruction(value, root, tuple(patches), dataset, local_hash)


def _build_reconstruction(
    payload: dict[str, object],
    root: Path,
    patches: tuple[str, ...],
    dataset: str | None,
    local_hash: str | None,
) -> ReconstructionEvidence:
    for item in patches:
        _require_sha(item, "reconstruction patch hash")
    if local_hash is not None:
        _require_sha(local_hash, "local data hash")
    return ReconstructionEvidence(
        olmes_commit=required_field(
            payload, "olmes_commit", str, "reconstruction evidence"
        ),
        olmes_patch_hashes=patches,
        dataset_revision=dataset,
        local_data_sha256=local_hash,
        reconstruction_code_sha256=_sha_field(
            payload, "reconstruction_code_sha256", "reconstruction evidence"
        ),
        sample_group_count=required_field(
            payload, "sample_group_count", int, "reconstruction evidence"
        ),
        reconstructed_requests=_artifact_field(payload, "reconstructed_requests", root),
        recorded_inputs=_artifact_field(payload, "recorded_inputs", root),
    )


def _optional_artifact(
    payload: dict[str, object], key: str, root: Path
) -> SourceArtifact | None:
    return None if payload[key] is None else _artifact_field(payload, key, root)


def _evidence_mode(payload: dict[str, object]) -> EvidenceMode:
    value = _text(payload, "evidence_mode")
    if value not in ("raw_requests", "verified_reconstruction"):
        raise ValueError("Invalid OLMES source evidence_mode")
    return cast(EvidenceMode, value)


def _text(payload: dict[str, object], key: str) -> str:
    value = required_field(payload, key, str, "OLMES source")
    if not value.strip():
        raise ValueError(f"OLMES source requires non-empty {key}")
    return value


def _sha_field(payload: dict[str, object], key: str, context: str) -> str:
    value = required_field(payload, key, str, context)
    _require_sha(value, key)
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def validate_model_revision_evidence(source: OlmesTaskSource) -> None:
    evidence = source.model_revision_evidence
    if evidence.kind == "resolved_snapshot":
        payload = _resolved_snapshot_payload(source)
        require_equal(payload.get("model_id"), source.model_id, "attested model id")
        require_equal(
            payload.get("model_revision"),
            source.model_revision,
            "attested model revision",
        )
        _validate_snapshot_uri(payload.get("snapshot_uri"), source.model_revision)
        return
    _validate_loading_attestation(source)


def _resolved_snapshot_payload(source: OlmesTaskSource) -> dict[str, object]:
    path = source.model_revision_evidence.artifact.path
    payload = parse_json_object(path.read_text(encoding="utf-8"), path, 1)
    require_keys(payload, _evidence_fields(), "model revision evidence")
    return payload


def _evidence_fields() -> frozenset[str]:
    return frozenset({"model_id", "model_revision", "snapshot_uri"})


def _validate_loading_attestation(source: OlmesTaskSource) -> None:
    from data_attribution.attribution.trackstar.lineage import (
        ModelContentAttestation,
        load_verified_attestation,
    )

    artifact = source.model_revision_evidence.artifact
    if artifact.record_count is not None:
        raise ValueError("Loading attestation artifact record_count must be null")
    attestation = load_verified_attestation(artifact.path)
    if type(attestation) is not ModelContentAttestation:
        raise ValueError("Loading evidence requires an exact model content attestation")
    if (
        attestation.lineage_basis != "loading_content_attestation"
        or attestation.raw_revision is not None
    ):
        raise ValueError(
            "Loading evidence requires loading_content_attestation lineage with null "
            "raw revision"
        )
    require_equal(attestation.model_id, source.model_id, "attested model ID")
    require_equal(
        attestation.build_revision,
        source.model_revision,
        "attested build revision",
    )
    require_equal(
        attestation.artifact_sha256,
        source.model_config_sha256,
        "attested model config artifact SHA-256",
    )


def _validate_snapshot_uri(value: object, revision: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("Invalid resolved snapshot_uri")
    if not _pins_revision(value, revision):
        raise ValueError("Resolved snapshot_uri must pin declared model revision")


def _pins_revision(value: str, revision: str) -> bool:
    parsed = urlsplit(value)
    path = unquote(parsed.path)
    location = unquote(f"{parsed.netloc}{parsed.path}")
    path_segments = tuple(segment for segment in path.split("/") if segment)
    prefix, separator, suffix = location.rpartition("@")
    has_locator = bool(parsed.scheme or path.startswith("/") or "/" in path or prefix)
    return has_locator and (
        revision in path_segments or bool(separator and suffix == revision)
    )


def validate_source_artifacts(source: OlmesTaskSource) -> None:
    for label, artifact in _source_artifacts(source):
        validate_artifact_fields(label, artifact)
        _validate_artifact_contents(label, artifact)


def validate_artifact_fields(label: str, artifact: object) -> None:
    if type(artifact) is not SourceArtifact:
        raise ValueError(f"Invalid {label} artifact type")
    if not isinstance(artifact.path, Path):
        raise ValueError(f"Invalid {label} path type")
    if not _is_hex(artifact.sha256, 64):
        raise ValueError(f"Invalid {label} artifact SHA-256")
    if type(artifact.complete) is not bool:
        raise ValueError(f"Invalid {label} complete type")
    count = artifact.record_count
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError(f"Invalid {label} record_count type")


def _source_artifacts(source: OlmesTaskSource) -> list[tuple[str, SourceArtifact]]:
    artifacts = [
        ("predictions", source.predictions),
        ("metrics", source.metrics),
        ("model_revision_evidence", source.model_revision_evidence.artifact),
    ]
    if source.requests is not None:
        artifacts.append(("requests", source.requests))
    if source.reconstruction is not None:
        artifacts.extend(
            (
                (
                    "reconstructed_requests",
                    source.reconstruction.reconstructed_requests,
                ),
                ("recorded_inputs", source.reconstruction.recorded_inputs),
            )
        )
    return artifacts


def _validate_artifact_contents(label: str, artifact: SourceArtifact) -> None:
    if not artifact.path.is_file():
        raise ValueError(f"Missing {label} artifact: {artifact.path}")
    if sha256_file(artifact.path) != artifact.sha256:
        raise ValueError(f"Invalid {label} SHA-256")
    count = artifact.record_count
    if count is not None and count_jsonl_objects(artifact.path) != count:
        raise ValueError(f"Invalid {label} record count")


_EVIDENCE_MODES = ("raw_requests", "verified_reconstruction")
_REVISION_KINDS = ("resolved_snapshot", "loading_content_attestation")


def validate_runtime_source_contract(source: OlmesTaskSource) -> None:
    if type(source) is not OlmesTaskSource:
        raise ValueError("Invalid OLMES source type")
    _validate_source_fields(source)
    _validate_revision_evidence(source.model_revision_evidence)
    _validate_mode_contract(source)
    _validate_cpqc_contract(source)
    validate_source_artifacts(source)
    validate_model_revision_evidence(source)


def _validate_source_fields(source: OlmesTaskSource) -> None:
    for field in (
        "source_id",
        "probe",
        "task_core",
        "task_alias",
        "model_id",
        "model_revision",
        "task_hash",
        "model_hash",
    ):
        _require_text(getattr(source, field), field)
    if Path(source.model_id).is_absolute():
        raise ValueError("OLMES source requires a canonical model ID")
    if (
        type(source.evidence_mode) is not str
        or source.evidence_mode not in _EVIDENCE_MODES
    ):
        raise ValueError("Invalid evidence_mode")
    _require_positive_int(source.expected_rows, "expected_rows")
    _require_sha(source.task_config_sha256, "task_config_sha256")
    _require_sha(source.model_config_sha256, "model_config_sha256")
    validate_artifact_fields("predictions", source.predictions)
    validate_artifact_fields("metrics", source.metrics)
    if source.predictions.record_count != source.expected_rows:
        raise ValueError("Predictions record_count must equal expected_rows")


def _validate_revision_evidence(evidence: object) -> None:
    if type(evidence) is not ModelRevisionEvidence:
        raise ValueError("Invalid model revision evidence type")
    if type(evidence.kind) is not str or evidence.kind not in _REVISION_KINDS:
        raise ValueError("Invalid revision evidence kind")
    validate_artifact_fields("model_revision_evidence", evidence.artifact)
    if not evidence.artifact.complete:
        raise ValueError("OLMES source requires complete model revision evidence")


def _validate_mode_contract(source: OlmesTaskSource) -> None:
    if source.evidence_mode == "raw_requests":
        _validate_raw_mode(source)
        return
    if (
        source.requests is not None
        or type(source.reconstruction) is not ReconstructionEvidence
    ):
        raise ValueError("verified_reconstruction requires reconstruction evidence")
    _validate_reconstruction(source.reconstruction)
    artifacts = (
        source.predictions,
        source.metrics,
        source.reconstruction.reconstructed_requests,
        source.reconstruction.recorded_inputs,
    )
    if not all(artifact.complete for artifact in artifacts):
        raise ValueError(
            "verified_reconstruction requires complete reconstruction artifacts"
        )


def _validate_raw_mode(source: OlmesTaskSource) -> None:
    if type(source.requests) is not SourceArtifact or source.reconstruction is not None:
        raise ValueError(
            "raw_requests requires complete predictions, requests, and metrics"
        )
    validate_artifact_fields("requests", source.requests)
    if not all(
        artifact.complete
        for artifact in (source.predictions, source.requests, source.metrics)
    ):
        raise ValueError(
            "raw_requests requires complete predictions, requests, and metrics"
        )


def _validate_reconstruction(reconstruction: ReconstructionEvidence) -> None:
    if not _is_hex(reconstruction.olmes_commit, 40):
        raise ValueError("Invalid reconstruction OLMES commit")
    patches = reconstruction.olmes_patch_hashes
    if type(patches) is not tuple:
        raise ValueError("Invalid olmes_patch_hashes type")
    if not patches:
        raise ValueError("Invalid reconstruction patch hashes")
    for patch in patches:
        _require_sha(patch, "reconstruction patch hash")
    _validate_reconstruction_data(reconstruction)
    _require_sha(reconstruction.reconstruction_code_sha256, "reconstruction code hash")
    _require_positive_int(reconstruction.sample_group_count, "sample_group_count")
    validate_artifact_fields(
        "reconstructed_requests", reconstruction.reconstructed_requests
    )
    validate_artifact_fields("recorded_inputs", reconstruction.recorded_inputs)


def _validate_reconstruction_data(reconstruction: ReconstructionEvidence) -> None:
    dataset = reconstruction.dataset_revision
    if dataset is not None and (type(dataset) is not str or not dataset.strip()):
        raise ValueError("reconstruction requires a non-empty dataset revision")
    local_hash = reconstruction.local_data_sha256
    if local_hash is not None:
        _require_sha(local_hash, "local data hash")
    if dataset is None and local_hash is None:
        raise ValueError("reconstruction requires dataset revision or local data hash")


def _validate_cpqc_contract(source: OlmesTaskSource) -> None:
    probe_count = CPQC_ROW_COUNTS.get(source.probe)
    task_count = CPQC_ROW_COUNTS.get(source.task_core)
    if (
        probe_count is not None or task_count is not None
    ) and source.probe != source.task_core:
        raise ValueError("CPQC task identity requires matching probe and task_core")
    required = probe_count if probe_count is not None else task_count
    if required is None:
        return
    if source.expected_rows != required:
        raise ValueError(
            f"{source.probe} reconstruction requires exactly {required} rows"
        )
    if source.evidence_mode != "verified_reconstruction":
        raise ValueError(f"{source.probe} requires verified_reconstruction")
    if source.reconstruction is None or source.reconstruction.sample_group_count != 3:
        raise ValueError("CPQC reconstruction requires three sample groups")


def _require_text(value: object, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"Invalid non-empty {label}")


def _require_positive_int(value: object, label: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"Invalid {label}")


def _require_sha(value: object, label: str) -> None:
    if not _is_hex(value, 64):
        raise ValueError(f"Invalid {label} SHA-256")


def _is_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_source_contract(source: OlmesTaskSource) -> None:
    validate_runtime_source_contract(source)


type RequestKey = tuple[str, str, int]
type GroupKey = tuple[str, str]


@dataclass(frozen=True)
class RecordedEnvelope:
    identity: GroupKey
    task_name: str
    label: int
    rows: tuple[dict[str, object], ...]


def read_request_rows(path: Path) -> list[dict[str, object]]:
    return read_jsonl_objects(path)


def read_recorded_envelopes(path: Path) -> tuple[RecordedEnvelope, ...]:
    return tuple(_decode_recorded_envelope(row) for row in read_jsonl_objects(path))


def request_key(row: dict[str, object]) -> RequestKey:
    native_id = scalar_identifier(row.get("native_id"), "native_id")
    document_id = scalar_identifier(row.get("doc_id"), "document id")
    candidate_index = row.get("idx")
    if type(candidate_index) is not int:
        raise ValueError("Request requires native_id, document id, and candidate index")
    return native_id, document_id, candidate_index


def scalar_identifier(value: object, label: str) -> str:
    if type(value) not in (str, int):
        raise ValueError(f"{label} must be a scalar string or integer")
    return str(value)


def request_text(row: dict[str, object]) -> tuple[str, str]:
    request = row.get("request")
    if not isinstance(request, dict):
        raise ValueError("Request payload must be an object")
    context = request.get("context")
    continuation = request.get("continuation")
    if type(context) is not str or type(continuation) is not str:
        raise ValueError("Request requires exact string context and continuation")
    return context, continuation


def _decode_recorded_envelope(envelope: dict[str, object]) -> RecordedEnvelope:
    require_keys(
        envelope,
        frozenset({"doc", "task_name", "doc_id", "native_id", "label", "requests"}),
        "recorded input envelope",
    )
    task_name = envelope["task_name"]
    label = envelope["label"]
    if type(task_name) is not str:
        raise ValueError("Invalid recorded task name")
    if type(label) is not int:
        raise ValueError("Invalid recorded label")
    identity = (
        scalar_identifier(envelope["native_id"], "native_id"),
        scalar_identifier(envelope["doc_id"], "document id"),
    )
    requests = envelope["requests"]
    if not isinstance(requests, list) or not requests:
        raise ValueError("Recorded input envelope has an empty candidate group")
    rows = _flatten_recorded_requests(envelope, requests)
    return RecordedEnvelope(identity, task_name, label, tuple(rows))


def _flatten_recorded_requests(
    envelope: dict[str, object], requests: list[object]
) -> list[dict[str, object]]:
    shared = {
        key: envelope[key] for key in ("task_name", "doc_id", "native_id", "label")
    }
    flattened: list[dict[str, object]] = []
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("Recorded input request must be an object")
        require_keys(
            request,
            frozenset({"request_type", "request", "idx"}),
            "recorded input request",
        )
        flattened.append({**shared, **request})
    return flattened


def verify_reconstruction(source: OlmesTaskSource) -> Path:
    validate_source_contract(source)
    if (
        source.evidence_mode != "verified_reconstruction"
        or source.reconstruction is None
    ):
        raise ValueError("Source does not contain verified reconstruction evidence")
    spec = source.reconstruction
    reconstructed = _indexed_requests(spec.reconstructed_requests.path)
    envelopes = read_recorded_envelopes(spec.recorded_inputs.path)
    _validate_envelope_count(envelopes, spec.sample_group_count)
    _validate_envelope_identities(envelopes)
    _validate_envelope_rosters(reconstructed, envelopes)
    recorded = _indexed_rows(_recorded_rows(envelopes))
    _compare_recorded_inputs(reconstructed, recorded)
    _validate_group_count(reconstructed, source.expected_rows, "reconstructed")
    return spec.reconstructed_requests.path


def read_recorded_rows(path: Path) -> list[dict[str, object]]:
    return _recorded_rows(read_recorded_envelopes(path))


def _indexed_requests(path: Path) -> dict[RequestKey, dict[str, object]]:
    return _indexed_rows(read_request_rows(path))


def _indexed_rows(
    rows: list[dict[str, object]],
) -> dict[RequestKey, dict[str, object]]:
    indexed: dict[RequestKey, dict[str, object]] = {}
    for row in rows:
        key = request_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate request join key: {key}")
        request_text(row)
        indexed[key] = row
    return indexed


def _validate_group_count(
    rows: dict[RequestKey, dict[str, object]], expected: int, label: str
) -> None:
    groups: dict[GroupKey, list[int]] = {}
    for native_id, document_id, candidate_index in rows:
        groups.setdefault((native_id, document_id), []).append(candidate_index)
    if len(groups) != expected:
        raise ValueError(
            f"Invalid {label} sample groups: expected {expected}, found {len(groups)}"
        )
    for key, indices in groups.items():
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"Incomplete candidates for {label} group {key}")


def _compare_recorded_inputs(
    reconstructed: dict[RequestKey, dict[str, object]],
    recorded: dict[RequestKey, dict[str, object]],
) -> None:
    for key, recorded_row in recorded.items():
        reconstructed_row = reconstructed.get(key)
        if reconstructed_row is None:
            raise ValueError(f"Missing recorded input join for {key}")
        expected_context, expected_continuation = request_text(recorded_row)
        actual_context, actual_continuation = request_text(reconstructed_row)
        _compare_recorded_metadata(reconstructed_row, recorded_row, key)
        if actual_context.encode("utf-8") != expected_context.encode("utf-8"):
            raise ValueError(f"Mismatched recorded context bytes for {key}")
        if actual_continuation.encode("utf-8") != expected_continuation.encode("utf-8"):
            raise ValueError(f"Mismatched recorded continuation bytes for {key}")


def _validate_envelope_count(
    envelopes: tuple[RecordedEnvelope, ...], expected: int
) -> None:
    if len(envelopes) != expected:
        raise ValueError(
            f"Invalid physical recorded envelope count for sample groups: expected {expected}, "
            f"found {len(envelopes)}"
        )


def _validate_envelope_identities(envelopes: tuple[RecordedEnvelope, ...]) -> None:
    identities = [envelope.identity for envelope in envelopes]
    if len(identities) != len(set(identities)):
        raise ValueError("Found duplicate recorded envelope identity")


def _validate_envelope_rosters(
    reconstructed: dict[RequestKey, dict[str, object]],
    envelopes: tuple[RecordedEnvelope, ...],
) -> None:
    for envelope in envelopes:
        expected = {key[2] for key in reconstructed if key[:2] == envelope.identity}
        if not expected:
            raise ValueError(f"Missing recorded input join for {envelope.identity}")
        actual = [request_key(row)[2] for row in envelope.rows]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(
                f"Recorded sample lacks the complete candidate roster for {envelope.identity}"
            )


def _recorded_rows(envelopes: tuple[RecordedEnvelope, ...]) -> list[dict[str, object]]:
    return [row for envelope in envelopes for row in envelope.rows]


def _compare_recorded_metadata(
    reconstructed: dict[str, object],
    recorded: dict[str, object],
    key: RequestKey,
) -> None:
    fields = (
        ("label", "label"),
        ("task_name", "task name"),
        ("request_type", "request type"),
    )
    for field, label in fields:
        actual = recorded.get(field)
        expected = reconstructed.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Mismatched recorded {label} for {key}")


LOG_2_OF_E = 1.44269504089
_PREDICTED_FIELDS = {
    "predicted_index_raw": "raw",
    "predicted_index_per_token": "per_token",
    "predicted_index_per_char": "per_character",
    "predicted_index_per_byte": "per_byte",
    "predicted_index_uncond": "unconditioned",
}
_ACCURACY_FIELDS = {
    "acc_raw": "raw",
    "acc_per_token": "per_token",
    "acc_per_char": "per_character",
    "acc_per_byte": "per_byte",
    "acc_uncond": "unconditioned",
}


def normalize_candidates(
    outputs: list[dict[str, object]], requests: list[dict[str, object]]
) -> tuple[tuple[CandidateEvidence, ...], dict[str, tuple[int, ...]]]:
    if len(outputs) != len(requests):
        raise ValueError("Candidate and model output counts differ")
    if not outputs:
        raise ValueError("Evidence contains an empty candidate output group")
    unconditioned = ["sum_logits_uncond" in output for output in outputs]
    if any(unconditioned) and not all(unconditioned):
        raise ValueError("Incomplete unconditioned score evidence")
    candidates = tuple(
        _candidate(index, output, request)
        for index, (output, request) in enumerate(zip(outputs, requests, strict=True))
    )
    winning_sets = {
        rule: _winning_indices(candidates, rule, minimize=rule == "per_byte")
        for rule in candidates[0].scores
    }
    return candidates, winning_sets


def validate_stored_metrics(
    metrics: dict[str, object],
    winning_sets: dict[str, tuple[int, ...]],
    gold: int,
) -> dict[str, bool]:
    if (
        metrics.get("predicted_index_per_char") is None
        or metrics.get("acc_per_char") is None
    ):
        raise ValueError("Missing per-character evidence")
    stored_indices: dict[str, int] = {}
    for field, rule in _PREDICTED_FIELDS.items():
        index = _validate_predicted(metrics.get(field), winning_sets, rule, field)
        if index is not None:
            stored_indices[rule] = index
    for field, rule in _ACCURACY_FIELDS.items():
        _validate_accuracy(metrics.get(field), stored_indices.get(rule), gold, field)
    return {rule: index == gold for rule, index in stored_indices.items()}


def _candidate(
    index: int, output: dict[str, object], request: dict[str, object]
) -> CandidateEvidence:
    _, continuation = request_text(request)
    if "sum_logits" not in output:
        raise ValueError("Missing per-character evidence: sum_logits")
    score = _number(output["sum_logits"], "sum_logits")
    tokens = output.get("num_tokens")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("Missing per-token evidence: num_tokens")
    scores = _scores(score, tokens, continuation, output)
    _validate_normalizations(output, scores)
    return CandidateEvidence(index, continuation, scores, tokens)


def _scores(
    score: float, tokens: int, continuation: str, output: dict[str, object]
) -> dict[str, float]:
    scores = {
        "raw": score,
        "per_character": score / max(len(continuation), 1),
        "per_token": score / max(tokens, 1),
        "per_byte": -LOG_2_OF_E * (score / max(len(continuation.encode("utf-8")), 1)),
    }
    if "sum_logits_uncond" in output:
        unconditioned = _number(output["sum_logits_uncond"], "sum_logits_uncond")
        scores["unconditioned"] = score - unconditioned
    return scores


def _validate_normalizations(
    output: dict[str, object], scores: dict[str, float]
) -> None:
    stored = {
        "logits_per_token": "per_token",
        "logits_per_char": "per_character",
        "bits_per_byte": "per_byte",
    }
    for field, rule in stored.items():
        if field in output and _number(output[field], field) != scores[rule]:
            raise ValueError(f"Stored {field} does not match exact reconstruction")


def _winning_indices(
    candidates: tuple[CandidateEvidence, ...], rule: str, *, minimize: bool
) -> tuple[int, ...]:
    values = [candidate.scores[rule] for candidate in candidates]
    winning_value = min(values) if minimize else max(values)
    return tuple(
        candidate.candidate_index
        for candidate in candidates
        if candidate.scores[rule] == winning_value
    )


def _validate_predicted(
    value: object,
    winning_sets: dict[str, tuple[int, ...]],
    rule: str,
    field: str,
) -> int | None:
    if value is None:
        return None
    winning = winning_sets.get(rule)
    if type(value) is not int or winning is None or value not in winning:
        raise ValueError(f"Stored {field} is outside exact winning set {winning}")
    return value


def _validate_accuracy(
    value: object, stored_index: int | None, gold: int, field: str
) -> None:
    if value is None:
        return
    if type(value) not in (bool, int) or value not in (0, 1) or stored_index is None:
        raise ValueError(f"Invalid stored {field}")
    expected = stored_index == gold
    if bool(value) != expected:
        raise ValueError(f"Stored {field} does not match stored predicted index")


def _number(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"Invalid numeric {field}")
    number = cast(int | float, value)
    if not math.isfinite(number):
        raise ValueError(f"Invalid numeric {field}")
    return float(number)


@dataclass(frozen=True)
class ValidatedTask:
    task_name: str
    uncond_docid_offset: int | None


def validate_metrics(source: OlmesTaskSource) -> ValidatedTask:
    path = source.metrics.path
    payload = parse_json_object(path.read_text(encoding="utf-8"), path, 1)
    task_config = _object(payload.get("task_config"), "task_config")
    model_config = _object(payload.get("model_config"), "model_config")
    require_equal(payload.get("task_hash"), source.task_hash, "task hash")
    require_equal(payload.get("model_hash"), source.model_hash, "model hash")
    require_equal(
        sha256_json(task_config), source.task_config_sha256, "task config hash"
    )
    require_equal(
        sha256_json(model_config), source.model_config_sha256, "model config hash"
    )
    require_equal(task_config.get("task_core"), source.task_core, "task core")
    task_name = task_config.get("task_name")
    if type(task_name) is not str:
        raise ValueError("Invalid task name")
    require_equal(payload.get("task_name"), task_name, "task name")
    metadata = _object(task_config.get("metadata"), "task metadata")
    require_equal(metadata.get("alias"), source.task_alias, "task alias")
    _validate_model_config(source, model_config)
    return ValidatedTask(task_name, _unconditioned_offset(task_config))


def lineage_for_source(source: OlmesTaskSource) -> tuple[ArtifactLineage, ...]:
    artifacts: list[tuple[str, SourceArtifact, str | None]] = [
        ("metrics", source.metrics, source.task_hash),
        (
            "model_revision_evidence",
            source.model_revision_evidence.artifact,
            source.model_revision,
        ),
        ("predictions", source.predictions, source.task_hash),
    ]
    if source.requests is not None:
        artifacts.append(("requests", source.requests, source.task_hash))
    if source.reconstruction is not None:
        artifacts.extend(_reconstruction_artifacts(source))
    lineage = [_lineage(source.source_id, *item) for item in artifacts]
    lineage.extend(_semantic_lineage(source))
    return tuple(sorted(lineage, key=lambda item: item.artifact_id))


def _validate_model_config(
    source: OlmesTaskSource, model_config: dict[str, object]
) -> None:
    model = model_config.get("model")
    tokenizer = model_config.get("tokenizer")
    if model == source.model_id:
        if tokenizer not in (None, source.model_id):
            raise ValueError("Canonical model and tokenizer locators must match")
    else:
        _validate_snapshot_model_locators(source, model, tokenizer)
    revision = model_config.get("revision")
    if revision is not None:
        require_equal(revision, source.model_revision, "model revision")
    validate_model_revision_evidence(source)


def _validate_snapshot_model_locators(
    source: OlmesTaskSource, model: object, tokenizer: object
) -> None:
    if model != tokenizer or type(model) is not str:
        raise ValueError("Snapshot model and tokenizer locators must match")
    if source.model_revision_evidence.kind != "resolved_snapshot":
        raise ValueError("Snapshot locator requires resolved snapshot evidence")
    payload = _resolved_snapshot_payload(source)
    if model != payload.get("snapshot_uri"):
        raise ValueError("Snapshot locator must match resolved snapshot evidence")
    _validate_hf_cache_snapshot_path(source, model)


def _validate_hf_cache_snapshot_path(source: OlmesTaskSource, locator: str) -> None:
    path = Path(locator)
    if not path.is_absolute() or str(path) != locator:
        raise ValueError(
            "Snapshot locator must be an absolute Hugging Face cache snapshot"
        )
    canonical_component = f"models--{source.model_id.replace('/', '--')}"
    component = path.parts[-3] if len(path.parts) >= 3 else ""
    component_is_valid = (
        component == canonical_component
        or (
            source.model_id,
            source.model_revision,
            component,
        )
        in _HF_CACHE_MODEL_COMPONENT_ALIASES
    )
    suffix_is_valid = path.parts[-2:] == (
        "snapshots",
        source.model_revision,
    )
    if (
        "." in path.parts
        or ".." in path.parts
        or not component_is_valid
        or not suffix_is_valid
    ):
        raise ValueError(
            "Snapshot locator must encode the canonical model ID and revision"
        )


def _unconditioned_offset(task_config: dict[str, object]) -> int | None:
    metric_kwargs = task_config.get("metric_kwargs")
    if metric_kwargs is None:
        return None
    if not isinstance(metric_kwargs, dict):
        raise ValueError("Invalid task metric_kwargs")
    offset = metric_kwargs.get("uncond_docid_offset")
    if offset is None:
        return None
    if type(offset) is not int or offset < 1:
        raise ValueError("Invalid uncond_docid_offset")
    return offset


def _reconstruction_artifacts(
    source: OlmesTaskSource,
) -> tuple[tuple[str, SourceArtifact, str | None], ...]:
    reconstruction = source.reconstruction
    if reconstruction is None:
        return ()
    return (
        (
            "reconstructed_requests",
            reconstruction.reconstructed_requests,
            reconstruction.dataset_revision,
        ),
        ("recorded_inputs", reconstruction.recorded_inputs, source.task_hash),
    )


def _semantic_lineage(source: OlmesTaskSource) -> list[ArtifactLineage]:
    values: list[tuple[str, str, str | None]] = [
        ("task_config", source.task_config_sha256, source.task_hash),
        ("model_config", source.model_config_sha256, source.model_revision),
    ]
    reconstruction = source.reconstruction
    if reconstruction is not None:
        values.append(
            ("reconstruction_code", reconstruction.reconstruction_code_sha256, None)
        )
        if reconstruction.local_data_sha256 is not None:
            values.append(("local_data", reconstruction.local_data_sha256, None))
        values.extend(
            (f"olmes_patch:{sha256}", sha256, reconstruction.olmes_commit)
            for sha256 in sorted(set(reconstruction.olmes_patch_hashes))
        )
    return [_manifest_lineage(source.source_id, *value) for value in values]


def _manifest_lineage(
    source_id: str, name: str, sha256: str, revision: str | None
) -> ArtifactLineage:
    return ArtifactLineage(
        artifact_id=f"{source_id}:{name}",
        uri=f"manifest://{source_id}/{name}",
        sha256=sha256,
        revision=revision,
    )


def _lineage(
    source_id: str, name: str, artifact: SourceArtifact, revision: str | None
) -> ArtifactLineage:
    return ArtifactLineage(
        artifact_id=f"{source_id}:{name}",
        uri=str(artifact.path),
        sha256=artifact.sha256,
        revision=revision,
    )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"Invalid {label} object")
    return value


type PredictionRows = dict[GroupKey, dict[str, object]]
type RequestGroups = dict[GroupKey, list[dict[str, object]]]


def group_requests(path: Path) -> RequestGroups:
    groups: RequestGroups = {}
    for row in read_request_rows(path):
        native_id, document_id, _ = request_key(row)
        groups.setdefault((native_id, document_id), []).append(row)
    for key, rows in groups.items():
        rows.sort(key=lambda item: request_key(item)[2])
        indices = [request_key(item)[2] for item in rows]
        if indices != list(range(len(rows))):
            raise ValueError(
                f"Request group {key} requires contiguous candidate indices"
            )
    return groups


def validate_request_coverage(
    source: OlmesTaskSource,
    predictions: PredictionRows,
    groups: RequestGroups,
    task: ValidatedTask,
) -> None:
    has_unconditioned = _has_unconditioned_scores(predictions)
    _validate_unconditioned_mode(source, task, has_unconditioned)
    prediction_keys = set(predictions)
    expected = set(prediction_keys)
    unconditioned: dict[GroupKey, GroupKey] = {}
    if source.evidence_mode == "raw_requests" and task.uncond_docid_offset:
        unconditioned = _derive_unconditioned_keys(
            prediction_keys, task.uncond_docid_offset
        )
        expected.update(unconditioned.values())
    missing = expected - groups.keys()
    extra = groups.keys() - expected
    if missing:
        raise ValueError(f"Missing request groups: {sorted(missing)}")
    if extra:
        raise ValueError(f"Found unexpected request groups: {sorted(extra)}")
    for rows in groups.values():
        validate_candidate_requests(rows, task.task_name)
    _validate_unconditioned(predictions, groups, unconditioned)


def _validate_unconditioned_mode(
    source: OlmesTaskSource, task: ValidatedTask, has_unconditioned: bool
) -> None:
    if has_unconditioned and source.evidence_mode != "raw_requests":
        raise ValueError("unconditioned scores require raw request evidence")
    if has_unconditioned and task.uncond_docid_offset is None:
        raise ValueError(
            "unconditioned scores require a declared uncond_docid_offset and request roster"
        )


def _derive_unconditioned_keys(
    prediction_keys: set[GroupKey], offset: int
) -> dict[GroupKey, GroupKey]:
    mapping = {key: _offset_key(key, offset) for key in prediction_keys}
    derived_keys = list(mapping.values())
    if len(derived_keys) != len(prediction_keys):
        raise ValueError("Unconditioned request mapping is incomplete")
    if len(set(derived_keys)) != len(derived_keys):
        raise ValueError("Unconditioned request mapping is non-injective")
    overlap = prediction_keys.intersection(derived_keys)
    if overlap:
        raise ValueError(
            "Unconditioned request mapping overlaps conditional request keys: "
            f"{sorted(overlap)}"
        )
    return mapping


def _has_unconditioned_scores(predictions: PredictionRows) -> bool:
    for row in predictions.values():
        outputs = row.get("model_output")
        if isinstance(outputs, list) and any(
            isinstance(output, dict) and "sum_logits_uncond" in output
            for output in outputs
        ):
            return True
    return False


def validate_candidate_requests(rows: list[dict[str, object]], task_name: str) -> None:
    if not rows:
        raise ValueError("Evidence contains an empty candidate request group")
    contexts = [request_text(row)[0] for row in rows]
    if len(set(contexts)) != 1:
        raise ValueError("Candidate requests require identical contexts")
    for row in rows:
        if row.get("request_type") != "loglikelihood":
            raise ValueError("Invalid request type for multiple-choice evidence")
        require_equal(row.get("task_name"), task_name, "request task name")


def _validate_unconditioned(
    predictions: PredictionRows,
    groups: RequestGroups,
    unconditioned: dict[GroupKey, GroupKey],
) -> None:
    for key, unconditioned_key in unconditioned.items():
        outputs = predictions[key].get("model_output")
        if (
            not isinstance(outputs, list)
            or not outputs
            or any(
                not isinstance(output, dict) or "sum_logits_uncond" not in output
                for output in outputs
            )
        ):
            raise ValueError(f"Missing unconditioned prediction evidence for {key}")
        _compare_rosters(groups[key], groups[unconditioned_key], key)


def _compare_rosters(
    conditional: list[dict[str, object]],
    unconditioned: list[dict[str, object]],
    key: GroupKey,
) -> None:
    if len(conditional) != len(unconditioned):
        raise ValueError(f"Incomplete unconditioned request roster for {key}")
    for first, second in zip(conditional, unconditioned, strict=True):
        if request_key(first)[2] != request_key(second)[2]:
            raise ValueError(f"Mismatched unconditioned candidate index for {key}")
        if request_text(first)[1].encode() != request_text(second)[1].encode():
            raise ValueError(f"Mismatched unconditioned continuation bytes for {key}")
        require_equal(second.get("label"), first.get("label"), "unconditioned label")


def _offset_key(key: GroupKey, offset: int) -> GroupKey:
    try:
        document_id = int(key[1])
    except ValueError as error:
        raise ValueError(
            f"Unconditioned offset requires numeric document id: {key[1]}"
        ) from error
    return key[0], str(document_id + offset)


def load_olmes_source_manifest(path: Path) -> tuple[OlmesTaskSource, ...]:
    payload = parse_json_object(path.read_text(encoding="utf-8"), path, 1)
    require_keys(payload, frozenset({"schema_version", "sources"}), "source manifest")
    version = required_field(payload, "schema_version", str, "source manifest")
    if version != SOURCE_MANIFEST_VERSION:
        raise ValueError("Unsupported source manifest schema version")
    items = object_list_field(payload, "sources", "source manifest")
    sources = tuple(decode_source(item, path.parent) for item in items)
    identities = [(item.probe, item.task_core) for item in sources]
    if len(identities) != len(set(identities)):
        raise ValueError("Source manifest contains duplicate probe/task sources")
    task_cores = [item.task_core for item in sources]
    if len(task_cores) != len(set(task_cores)):
        raise ValueError("Source manifest contains duplicate task sources")
    return tuple(
        sorted(sources, key=lambda item: (item.probe, item.task_core, item.source_id))
    )


@dataclass(frozen=True)
class _BuildContext:
    source: OlmesTaskSource
    groups: RequestGroups
    native_counts: Counter[str]
    task: ValidatedTask
    lineage: tuple[ArtifactLineage, ...]


def load_olmes_evidence(source: OlmesTaskSource) -> list[QueryEvidence]:
    validate_source_contract(source)
    task = validate_metrics(source)
    request_path = (
        source.requests.path if source.requests else verify_reconstruction(source)
    )
    request_groups = group_requests(request_path)
    predictions = read_jsonl_objects(source.predictions.path)
    if len(predictions) != source.expected_rows:
        raise ValueError("Prediction count does not match expected_rows")
    keyed = _prediction_rows(predictions, source)
    validate_request_coverage(source, keyed, request_groups, task)
    native_counts = Counter(native_id for native_id, _ in keyed)
    lineage = lineage_for_source(source)
    context = _BuildContext(source, request_groups, native_counts, task, lineage)
    evidence = [_build_evidence(context, key, row) for key, row in keyed.items()]
    _validate_query_ids(evidence)
    validate_source_artifacts(source)
    return sorted(evidence, key=lambda item: item.query_id)


def _build_evidence(
    context: _BuildContext, key: tuple[str, str], row: dict[str, object]
) -> QueryEvidence:
    requests = context.groups.get(key)
    if requests is None:
        raise ValueError(f"Missing request group for prediction {key}")
    candidates, gold, correctness = _normalize_row(row, requests)
    source = context.source
    query_id = _query_id(source.task_core, key, context.native_counts[key[0]])
    return QueryEvidence(
        query_id=query_id,
        probe=source.probe,
        task_core=source.task_core,
        native_id=key[0],
        document_id=key[1],
        prompt=request_text(requests[0])[0],
        candidates=candidates,
        model_id=source.model_id,
        model_revision=source.model_revision,
        task_revision=source.task_hash,
        lineage=context.lineage,
        reference_index=gold,
        correctness=correctness,
    )


def _normalize_row(
    row: dict[str, object], requests: list[dict[str, object]]
) -> tuple[tuple[CandidateEvidence, ...], int, dict[str, bool]]:
    metrics = _object(row.get("metrics"), "prediction metrics")
    outputs = _object_list(row.get("model_output"), "model_output")
    gold = _gold_index(row, metrics, requests, len(outputs))
    continuations = [request_text(request)[1] for request in requests]
    if len(set(continuations)) != len(continuations):
        raise ValueError("Evidence contains duplicate candidates")
    candidates, winning_sets = normalize_candidates(outputs, requests)
    correctness = validate_stored_metrics(metrics, winning_sets, gold)
    return candidates, gold, correctness


def _prediction_rows(
    rows: list[dict[str, object]], source: OlmesTaskSource
) -> dict[tuple[str, str], dict[str, object]]:
    keyed: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        require_equal(row.get("task_hash"), source.task_hash, "prediction task hash")
        require_equal(row.get("model_hash"), source.model_hash, "prediction model hash")
        key = _identity(row)
        if key in keyed:
            raise ValueError(f"Ambiguous duplicate prediction identity: {key}")
        keyed[key] = row
    return keyed


def _gold_index(
    row: dict[str, object],
    metrics: dict[str, object],
    requests: list[dict[str, object]],
    candidate_count: int,
) -> int:
    values = [row.get("label"), metrics.get("correct_choice")]
    values.extend(request.get("label") for request in requests)
    if any(type(value) is not int for value in values) or len(set(values)) != 1:
        raise ValueError("Prediction has inconsistent gold labels or correct_choice")
    gold = values[0]
    if gold not in range(candidate_count):
        raise ValueError("Reference index is outside the candidate range")
    return cast(int, gold)


def _identity(row: dict[str, object]) -> tuple[str, str]:
    native_id = scalar_identifier(row.get("native_id"), "native_id")
    document_id = scalar_identifier(row.get("doc_id"), "document id")
    return native_id, document_id


def _query_id(task_core: str, key: tuple[str, str], native_count: int) -> str:
    task = _query_component(task_core)
    native = _query_component(key[0])
    if native_count == 1:
        return f"{task}:{native}"
    if key[1] == "":
        raise ValueError(f"Ambiguous duplicate native ID: {key[0]}")
    return f"{task}:{native}:{_query_component(key[1])}"


def _query_component(value: str) -> str:
    return value.replace("%", "%25").replace(":", "%3A")


def _validate_query_ids(evidence: list[QueryEvidence]) -> None:
    counts = Counter(item.query_id for item in evidence)
    duplicates = sorted(query_id for query_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Found duplicate generated query IDs: {duplicates}")


def _object_list(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"Invalid {label} list")
    return value
