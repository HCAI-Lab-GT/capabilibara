"""Shared model-lineage records and identity validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from types import MappingProxyType
from typing import Literal


type LineageBasis = Literal["pinned_revision", "loading_content_attestation"]
type LoadingFileKind = Literal["git_blob", "lfs_object", "safetensors_index"]

LINEAGE_SCHEMA_VERSION = "1"


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("Repository JSON objects require string keys")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and isfinite(value):
        return value
    raise ValueError("Repository JSON contains an unsupported value")


@dataclass(frozen=True)
class RepoFile:
    path: str
    size: int
    git_oid: str
    lfs_sha256: str | None = None
    index_payload: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.index_payload is not None:
            object.__setattr__(self, "index_payload", _freeze_json(self.index_payload))


@dataclass(frozen=True)
class RepoSnapshot:
    revision: str
    files: tuple[RepoFile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))


@dataclass(frozen=True)
class RepoSnapshots:
    build_time: RepoSnapshot
    requested: RepoSnapshot


@dataclass(frozen=True)
class TotalParametersCorrection:
    build_value: int
    requested_value: int


@dataclass(frozen=True)
class ModelContentRequest:
    artifact_id: str
    artifact_uri: str
    artifact_sha256: str
    build_timestamp: str
    model_id: str
    raw_revision: str | None
    build_revision: str
    requested_revision: str
    ignored_documentation_paths: tuple[str, ...] = ()
    total_parameters_correction: TotalParametersCorrection | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ignored_documentation_paths", tuple(self.ignored_documentation_paths)
        )


@dataclass(frozen=True)
class NormalizedLoadingFile:
    path: str
    size: int
    kind: LoadingFileKind
    content_id: str


@dataclass(frozen=True)
class ModelContentAttestation:
    artifact_id: str
    artifact_uri: str
    artifact_sha256: str
    build_timestamp: str
    model_id: str
    raw_revision: str | None
    build_revision: str
    requested_revision: str
    lineage_basis: LineageBasis
    build_snapshot: RepoSnapshot
    requested_snapshot: RepoSnapshot
    ignored_documentation_paths: tuple[str, ...]
    metadata_normalizations: tuple[TotalParametersCorrection, ...]
    normalized_files: tuple[NormalizedLoadingFile, ...]
    loading_content_sha256: str
    attestation_sha256: str
    schema_version: str = LINEAGE_SCHEMA_VERSION
    kind: Literal["model_content"] = "model_content"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ignored_documentation_paths", tuple(self.ignored_documentation_paths)
        )
        object.__setattr__(
            self, "metadata_normalizations", tuple(self.metadata_normalizations)
        )
        object.__setattr__(self, "normalized_files", tuple(self.normalized_files))


def validate_model_request(request: ModelContentRequest) -> LineageBasis:
    if type(request) is not ModelContentRequest:
        raise ValueError("Model content requires an immutable request")
    for value, label in (
        (request.artifact_id, "artifact ID"),
        (request.artifact_uri, "artifact URI"),
        (request.model_id, "model ID"),
    ):
        _validate_nonempty(value, label)
    if "://" not in request.artifact_uri:
        raise ValueError("Model content requires a durable artifact URI")
    validate_hex(request.artifact_sha256, 64, "artifact SHA-256")
    validate_hex(request.build_revision, 40, "build revision")
    validate_hex(request.requested_revision, 40, "requested revision")
    _validate_timestamp(request.build_timestamp)
    _validate_correction(request.total_parameters_correction)
    if request.raw_revision is None:
        return "loading_content_attestation"
    validate_hex(request.raw_revision, 40, "raw revision")
    if request.raw_revision != request.build_revision:
        raise ValueError("Raw revision conflicts with explicit build revision")
    return "pinned_revision"


def validate_hex(value: object, length: int, label: str) -> None:
    if type(value) is not str or len(value) != length:
        raise ValueError(f"Invalid {label}")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"Invalid {label}")


def _validate_correction(correction: TotalParametersCorrection | None) -> None:
    if correction is None:
        return
    if type(correction) is not TotalParametersCorrection:
        raise ValueError("Invalid total_parameters correction declaration")
    values = (correction.build_value, correction.requested_value)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Invalid total_parameters correction declaration")
    if correction.build_value == correction.requested_value:
        raise ValueError("total_parameters correction values must differ")


def _validate_nonempty(value: object, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"Invalid nonempty {label}")


def _validate_timestamp(value: object) -> None:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError("Build timestamp must be UTC ISO-8601")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("Build timestamp must be UTC ISO-8601") from error
    if parsed.tzinfo != UTC:
        raise ValueError("Build timestamp must be UTC ISO-8601")
