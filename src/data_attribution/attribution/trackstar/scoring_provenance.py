"""Generic ordered-input identity helpers and static scoring provenance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
from typing import Literal, cast

from data_attribution.artifact_integrity import (
    canonical_json_bytes,
    parse_json_object,
    sha256_json,
)
from data_attribution.attribution.trackstar.lineage import validate_hex
from data_attribution.attribution.trackstar.lineage_model import (
    has_symlink_component,
    open_stable_regular_file,
    require_integer,
    require_keys,
    require_object,
    require_text,
)


InventoryKind = Literal["source", "query", "document"]


@dataclass(frozen=True)
class InventoryEvidence:
    """Hash and row count for one ordered source, query, or document inventory."""

    kind: InventoryKind
    sha256: str
    row_count: int

    def __post_init__(self) -> None:
        if self.kind not in ("source", "query", "document"):
            raise ValueError("Invalid scoring inventory kind")
        validate_hex(self.sha256, 64, "scoring inventory SHA-256")
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("Scoring inventory row count must be positive")


@dataclass(frozen=True)
class ScoringProvenance:
    """Self-hashed binding for the three rosters that define a score join."""

    source: InventoryEvidence
    query: InventoryEvidence
    document: InventoryEvidence
    receipt_sha256: str
    schema_version: Literal["1"] = "1"
    kind: Literal["trackstar_scoring_provenance"] = "trackstar_scoring_provenance"


def scoring_provenance_sha256(value: ScoringProvenance) -> str:
    """Hash a scoring provenance record without its self-hash field."""

    return sha256_json(replace(value, receipt_sha256=""))


def build_scoring_provenance(
    source: InventoryEvidence,
    query: InventoryEvidence,
    document: InventoryEvidence,
) -> ScoringProvenance:
    """Build a static score-join binding from three ordered inventories."""

    _validate_inventory_kind(source, "source")
    _validate_inventory_kind(query, "query")
    _validate_inventory_kind(document, "document")
    unsigned = ScoringProvenance(source, query, document, "")
    result = replace(unsigned, receipt_sha256=scoring_provenance_sha256(unsigned))
    validate_provenance(result)
    return result


def validate_provenance(value: ScoringProvenance) -> None:
    """Validate the static score-join schema and self-hash."""

    if (
        type(value) is not ScoringProvenance
        or value.schema_version != "1"
        or value.kind != "trackstar_scoring_provenance"
    ):
        raise ValueError("Unsupported scoring provenance")
    _validate_inventory_kind(value.source, "source")
    _validate_inventory_kind(value.query, "query")
    _validate_inventory_kind(value.document, "document")
    validate_hex(value.receipt_sha256, 64, "scoring provenance SHA-256")
    if value.receipt_sha256 != scoring_provenance_sha256(value):
        raise ValueError("Invalid scoring provenance self-hash")


def serialize_provenance(value: ScoringProvenance) -> bytes:
    """Serialize one validated scoring provenance record as canonical JSONL."""

    validate_provenance(value)
    return canonical_json_bytes(value) + b"\n"


def decode_provenance(payload: bytes | str) -> ScoringProvenance:
    """Decode one strict scoring provenance record."""

    value = _payload(payload, "scoring provenance")
    require_keys(value, ScoringProvenance, "scoring provenance")
    result = ScoringProvenance(
        _decode_inventory(require_object(value, "source"), "source"),
        _decode_inventory(require_object(value, "query"), "query"),
        _decode_inventory(require_object(value, "document"), "document"),
        require_text(value, "receipt_sha256"),
        schema_version=cast(Literal["1"], require_text(value, "schema_version")),
        kind=cast(
            Literal["trackstar_scoring_provenance"],
            require_text(value, "kind"),
        ),
    )
    validate_provenance(result)
    return result


def write_provenance(value: ScoringProvenance, path: Path) -> None:
    """Write one absent-only, read-only scoring provenance record."""

    _write_absent(path, serialize_provenance(value))
    if load_provenance(path) != value:
        raise ValueError("Written scoring provenance failed verification")


def load_provenance(path: Path) -> ScoringProvenance:
    """Load and validate one scoring provenance record."""

    with open_stable_regular_file(path) as stream:
        return decode_provenance(stream.read())


def load_verified_provenance(
    path: Path,
    source: InventoryEvidence,
    query: InventoryEvidence,
    document: InventoryEvidence,
) -> ScoringProvenance:
    """Load provenance only when it matches the caller's current rosters."""

    value = load_provenance(path)
    expected = build_scoring_provenance(source, query, document)
    if value != expected:
        raise ValueError("Scoring provenance differs from current inventories")
    return value


def ordered_doc_roster_sha256(
    paths: Sequence[Path], *, state_dir: Path | None = None
) -> tuple[str, int]:
    """Hash source-order document IDs across an ordered file roster."""

    ordered = tuple(path.absolute() for path in paths)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("source path roster must be nonempty and unique")
    root = Path(tempfile.gettempdir()) if state_dir is None else state_dir.absolute()
    database, database_path = _state_database(root)
    digest = hashlib.sha256()
    count = 0
    try:
        for path in ordered:
            count = _hash_source_path(
                path, digest=digest, database=database, count=count
            )
        if count == 0:
            raise ValueError("source roster must contain at least one row")
        return digest.hexdigest(), count
    finally:
        try:
            database.close()
        finally:
            _remove_database_files(database_path)


def ordered_doc_id_sha256(path: Path) -> tuple[str, int]:
    """Hash document IDs in one source file."""

    return ordered_doc_roster_sha256((path,))


def ordered_query_id_sha256(path: Path) -> tuple[str, int]:
    """Validate one query/correctness JSONL file and hash its ordered IDs."""

    with open_stable_regular_file(path) as stream:
        return ordered_query_id_content_sha256(stream.read())


def ordered_query_id_content_sha256(content: bytes) -> tuple[str, int]:
    """Validate query/correctness bytes and hash their ordered query IDs."""

    if not content.strip():
        raise ValueError("correctness slice must not be empty")
    digest = hashlib.sha256()
    count = 0
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        try:
            raw = json.loads(raw_line, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"correctness row {line_number} is invalid JSON"
            ) from error
        if type(raw) is not dict:
            raise ValueError(f"correctness row {line_number} must be an object")
        row = cast(dict[str, object], raw)
        query_id = row.get("query_id")
        if type(query_id) is not str or not query_id:
            raise ValueError("correctness query_id must be a nonempty string")
        if type(row.get("is_correct")) is not bool:
            raise ValueError("correctness is_correct must be a boolean")
        encoded = query_id.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        count += 1
    if not count:
        raise ValueError("correctness slice must not be empty")
    return digest.hexdigest(), count


def _hash_source_path(
    path: Path,
    *,
    digest: hashlib._Hash,
    database: sqlite3.Connection,
    count: int,
) -> int:
    with open_stable_regular_file(path) as source:
        for line_number, raw_line in enumerate(source, start=1):
            doc_id = _document_id(raw_line, line_number=line_number)
            try:
                database.execute("INSERT INTO document_ids VALUES (?)", (doc_id,))
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    f"source roster contains duplicate id: {doc_id}"
                ) from error
            encoded = doc_id.encode("utf-8")
            digest.update(struct.pack(">Q", len(encoded)))
            digest.update(encoded)
            count += 1
    return count


def _document_id(raw_line: bytes, *, line_number: int) -> str:
    if not raw_line.strip():
        raise ValueError(f"source shard contains a blank row at {line_number}")
    try:
        raw = json.loads(raw_line, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"source shard row {line_number} is invalid JSON") from error
    if type(raw) is not dict:
        raise ValueError(f"source shard row {line_number} must be a JSON object")
    row = cast(dict[str, object], raw)
    if set(row) != {"id", "text"}:
        raise ValueError(
            f"source shard row {line_number} must contain exactly id and text"
        )
    doc_id = row["id"]
    if type(doc_id) is not str or not doc_id:
        raise ValueError(f"source shard row {line_number} id must be a nonempty string")
    return doc_id


def _state_database(state_dir: Path) -> tuple[sqlite3.Connection, Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise ValueError("scoring provenance state directory is invalid")
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".ordered-doc-roster-", suffix=".sqlite3", dir=state_dir
    )
    descriptor_path = Path(raw_path)
    database: sqlite3.Connection | None = None
    try:
        database = sqlite3.connect(descriptor_path)
        database.execute(
            "CREATE TABLE document_ids (id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        return database, descriptor_path
    except BaseException:
        if database is not None:
            database.close()
        descriptor_path.unlink(missing_ok=True)
        raise
    finally:
        if database is not None:
            # The successful caller owns the open connection and closes it later.
            pass
        else:
            Path(raw_path).unlink(missing_ok=True)
        # mkstemp returns an open descriptor; sqlite opens the path separately.
        os.close(descriptor)


def _remove_database_files(path: Path) -> None:
    for suffix in ("", "-journal", "-shm", "-wal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _decode_inventory(
    value: dict[str, object], expected_kind: InventoryKind
) -> InventoryEvidence:
    require_keys(value, InventoryEvidence, "scoring inventory")
    result = InventoryEvidence(
        cast(InventoryKind, require_text(value, "kind")),
        require_text(value, "sha256"),
        require_integer(value, "row_count"),
    )
    _validate_inventory_kind(result, expected_kind)
    return result


def _validate_inventory_kind(value: InventoryEvidence, expected: InventoryKind) -> None:
    if type(value) is not InventoryEvidence or value.kind != expected:
        raise ValueError(f"Scoring provenance {expected} inventory is invalid")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _payload(payload: bytes | str, label: str) -> dict[str, object]:
    if type(payload) is bytes:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} is not UTF-8") from error
    elif type(payload) is str:
        text = payload
    else:
        raise TypeError(f"{label} payload must be bytes or text")
    return parse_json_object(text, Path(f"<{label}>"), 1)


def _write_absent(path: Path, payload: bytes) -> None:
    if not isinstance(path, Path) or path.exists() or path.is_symlink():
        raise ValueError(f"Scoring provenance already exists: {path}")
    if has_symlink_component(path.parent):
        raise ValueError("Scoring provenance path contains a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    path.chmod(0o444)


__all__ = [
    "InventoryEvidence",
    "ScoringProvenance",
    "build_scoring_provenance",
    "decode_provenance",
    "load_provenance",
    "load_verified_provenance",
    "ordered_doc_id_sha256",
    "ordered_doc_roster_sha256",
    "ordered_query_id_content_sha256",
    "ordered_query_id_sha256",
    "scoring_provenance_sha256",
    "serialize_provenance",
    "validate_provenance",
    "write_provenance",
]
