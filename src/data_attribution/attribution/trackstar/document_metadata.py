"""Capture and verify document-to-bin metadata for controlled scoring."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_attribution.artifact_integrity import sha256_json

REQUIRED_COLUMNS = ("doc_id", "bin_topic", "bin_format")
_MAX_METADATA_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class DocumentMetadataArtifactRef:
    path: str
    durable_uri: str
    byte_count: int
    file_sha256: str
    schema_sha256: str
    row_count: int
    ordered_document_id_sha256: str
    metadata_sha256: str


def capture_document_metadata(
    path: Path, durable_uri: str
) -> DocumentMetadataArtifactRef:
    reference, _ = _inspect(path, durable_uri)
    return reference


def load_document_metadata(
    reference: DocumentMetadataArtifactRef,
    expected_document_ids: tuple[str, ...],
) -> dict[str, tuple[str, str]]:
    if type(reference) is not DocumentMetadataArtifactRef:
        raise ValueError("Invalid document metadata reference")
    actual, mapping = _inspect(Path(reference.path), reference.durable_uri)
    if actual != reference:
        raise ValueError("Document metadata reference differs from file")
    _require_exact_roster(mapping, expected_document_ids)
    return mapping


def capture_and_load_document_metadata(
    path: Path, durable_uri: str, expected_document_ids: tuple[str, ...]
) -> tuple[DocumentMetadataArtifactRef, dict[str, tuple[str, str]]]:
    """Stable-read, hash, parse, and roster-check metadata exactly once."""
    reference, mapping = _inspect(path, durable_uri)
    _require_exact_roster(mapping, expected_document_ids)
    return reference, mapping


def capture_and_load_document_metadata_for_join_ids(
    path: Path, durable_uri: str, expected_join_ids: tuple[str, ...]
) -> tuple[DocumentMetadataArtifactRef, dict[str, tuple[str, str]]]:
    """Load metadata keyed by a source field that may repeat across occurrences."""
    reference, mapping, rows = _inspect_with_rows(path, durable_uri)
    _require_exact_join_roster(rows, expected_join_ids)
    return reference, mapping


def _inspect(
    path: Path, durable_uri: str
) -> tuple[DocumentMetadataArtifactRef, dict[str, tuple[str, str]]]:
    reference, mapping, _ = _inspect_with_rows(path, durable_uri)
    return reference, mapping


def _inspect_with_rows(
    path: Path, durable_uri: str
) -> tuple[
    DocumentMetadataArtifactRef,
    dict[str, tuple[str, str]],
    tuple[tuple[str, str, str], ...],
]:
    _validate_location(path, durable_uri)
    with _open_stable_file(path) as stream:
        raw = stream.read(_MAX_METADATA_BYTES + 1)
        if len(raw) > _MAX_METADATA_BYTES:
            raise ValueError("Document metadata exceeds the bounded read contract")
        file_sha256 = hashlib.sha256(raw).hexdigest()
        byte_count = len(raw)
        try:
            parquet = pq.ParquetFile(pa.BufferReader(raw))
            schema = parquet.schema_arrow
            _validate_schema(schema)
            rows, mapping = _read_rows(parquet)
        except (OSError, pa.ArrowException) as error:
            raise ValueError(f"Invalid document metadata Parquet: {path}") from error
    reference = _reference(path, durable_uri, byte_count, file_sha256, schema, rows)
    return reference, mapping, rows


def _open_stable_file(path: Path):
    from data_attribution.attribution.trackstar.lineage_model import (
        open_stable_regular_file,
    )

    return open_stable_regular_file(path)


def _validate_location(path: Path, durable_uri: str) -> None:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("Document metadata path must be absolute")
    if type(durable_uri) is not str or "://" not in durable_uri:
        raise ValueError("Document metadata requires a durable URI")


def _validate_schema(schema: pa.Schema) -> None:
    if len(schema.names) != len(set(schema.names)):
        raise ValueError("Document metadata has duplicate columns")
    missing = sorted(set(REQUIRED_COLUMNS).difference(schema.names))
    if missing:
        raise ValueError(f"Document metadata lacks required columns: {missing}")
    for name in REQUIRED_COLUMNS:
        data_type = schema.field(name).type
        if not (pa.types.is_string(data_type) or pa.types.is_large_string(data_type)):
            raise ValueError("Document metadata required columns must be strings")


def _read_rows(
    parquet: pq.ParquetFile,
) -> tuple[tuple[tuple[str, str, str], ...], dict[str, tuple[str, str]]]:
    rows: list[tuple[str, str, str]] = []
    mapping: dict[str, tuple[str, str]] = {}
    for batch in parquet.iter_batches(columns=list(REQUIRED_COLUMNS)):
        values = batch.to_pydict()
        for doc_id, topic, fmt in zip(*(values[name] for name in REQUIRED_COLUMNS)):
            row = _validate_row(doc_id, topic, fmt)
            assignment = row[1:]
            previous = mapping.get(row[0])
            if previous is not None and previous != assignment:
                raise ValueError(f"Conflicting document metadata for ID: {row[0]}")
            mapping[row[0]] = assignment
            rows.append(row)
    if not rows:
        raise ValueError("Document metadata must contain at least one row")
    return tuple(rows), mapping


def _validate_row(doc_id, topic, fmt) -> tuple[str, str, str]:
    values = (doc_id, topic, fmt)
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError("Document metadata values must be nonempty strings")
    return values


def _reference(path, uri, byte_count, file_sha256, schema, rows):
    ordered_ids = hashlib.sha256()
    metadata = hashlib.sha256()
    for row in rows:
        _update_part(ordered_ids, row[0])
        for value in row:
            _update_part(metadata, value)
    schema_view = tuple(
        (field.name, str(field.type), field.nullable) for field in schema
    )
    return DocumentMetadataArtifactRef(
        str(path.absolute()),
        uri,
        byte_count,
        file_sha256,
        sha256_json(schema_view),
        len(rows),
        ordered_ids.hexdigest(),
        metadata.hexdigest(),
    )


def _update_part(digest, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _require_exact_roster(mapping, expected: tuple[str, ...]) -> None:
    if any(type(value) is not str or not value for value in expected):
        raise ValueError("Invalid expected document-ID roster")
    if len(expected) != len(set(expected)) or set(expected) != set(mapping):
        raise ValueError("Document metadata differs from document-ID roster")


def _require_exact_join_roster(
    rows: tuple[tuple[str, str, str], ...], expected: tuple[str, ...]
) -> None:
    if any(type(value) is not str or not value for value in expected):
        raise ValueError("Invalid expected document-ID roster")
    if Counter(row[0] for row in rows) != Counter(expected):
        raise ValueError("Document metadata differs from source metadata join roster")


__all__ = [
    "DocumentMetadataArtifactRef",
    "capture_document_metadata",
    "capture_and_load_document_metadata",
    "capture_and_load_document_metadata_for_join_ids",
    "load_document_metadata",
]
