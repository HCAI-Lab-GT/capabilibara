"""Public interface and shared records for TrackStar lineage attestations."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from data_attribution.artifact_integrity import (
    canonical_json_bytes,
    parse_json_object,
)
from data_attribution.attribution.trackstar.lineage_records import (
    LINEAGE_SCHEMA_VERSION,
    LineageBasis,
    ModelContentAttestation,
    ModelContentRequest,
    NormalizedLoadingFile,
    RepoFile,
    RepoSnapshot,
    RepoSnapshots,
    TotalParametersCorrection,
    validate_hex,
)


def build_model_content_attestation(
    request: ModelContentRequest, snapshots: RepoSnapshots
) -> ModelContentAttestation:
    from data_attribution.attribution.trackstar.lineage_model import (
        build_model_content_attestation as build,
    )

    return build(request, snapshots)


from data_attribution.attribution.trackstar.lineage_preconditioner import (  # noqa: E402
    ArtifactRef,
    ModuleShape,
    PreconditionerAttestation,
    PreconditionerSettings,
    RootArtifactFile,
)

type LineageAttestation = ModelContentAttestation | PreconditionerAttestation


def build_preconditioner_attestation(
    root: Path,
    source_refs: tuple[ArtifactRef, ...],
    model_attestation: ModelContentAttestation,
) -> PreconditionerAttestation:
    from data_attribution.attribution.trackstar.lineage_preconditioner import (
        build_preconditioner_attestation as build,
    )

    return build(root, source_refs, model_attestation)


def write_attestation(attestation: LineageAttestation, path: Path) -> None:
    _validate_attestation(attestation)
    if path.exists():
        raise ValueError(f"Refusing to overwrite lineage attestation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    temporary.write_bytes(canonical_json_bytes(attestation) + b"\n")
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(
            f"Refusing to overwrite lineage attestation: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    if load_verified_attestation(path) != attestation:
        raise ValueError("Written lineage attestation failed verification")


def load_verified_attestation(path: Path) -> LineageAttestation:
    from data_attribution.attribution.trackstar.lineage_model import (
        decode_model_attestation,
    )
    from data_attribution.attribution.trackstar.lineage_preconditioner import (
        decode_preconditioner_attestation,
    )

    payload = parse_json_object(path.read_text(encoding="utf-8"), path, 1)
    if payload.get("kind") == "model_content":
        result: LineageAttestation = decode_model_attestation(payload)
    elif payload.get("kind") == "preconditioner":
        result = decode_preconditioner_attestation(payload)
    else:
        raise ValueError("Unknown lineage attestation kind")
    _validate_attestation(result)
    return result


def _validate_attestation(attestation: LineageAttestation) -> None:
    from data_attribution.attribution.trackstar.lineage_model import (
        model_attestation_sha256,
        validate_model_attestation,
    )
    from data_attribution.attribution.trackstar.lineage_preconditioner import (
        validate_preconditioner_attestation,
    )

    if attestation.schema_version != LINEAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported lineage attestation schema version")
    if type(attestation) is ModelContentAttestation:
        if attestation.attestation_sha256 != model_attestation_sha256(attestation):
            raise ValueError("Invalid lineage attestation hash")
        validate_model_attestation(attestation)
    elif type(attestation) is PreconditionerAttestation:
        validate_preconditioner_attestation(attestation)
    else:
        raise ValueError("Unsupported lineage attestation type")


__all__ = [
    "ArtifactRef",
    "LineageBasis",
    "LineageAttestation",
    "ModuleShape",
    "ModelContentAttestation",
    "ModelContentRequest",
    "NormalizedLoadingFile",
    "PreconditionerAttestation",
    "PreconditionerSettings",
    "RepoFile",
    "RepoSnapshot",
    "RepoSnapshots",
    "RootArtifactFile",
    "TotalParametersCorrection",
    "build_model_content_attestation",
    "build_preconditioner_attestation",
    "load_verified_attestation",
    "validate_hex",
    "write_attestation",
]
