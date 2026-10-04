"""Verified JSONL I/O for compiled and rendered query bundles."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, TypedDict, cast

import data_attribution.query_conditions.compiler as contracts
from data_attribution.artifact_integrity import (
    canonical_json_bytes,
    jsonable,
    parse_json_object,
    require_equal,
    sha256_file,
    sha256_json,
)
from data_attribution.query_conditions.types import (
    COMPLETION_TIE_POLICIES,
    QUERY_CONDITIONS,
    QUERY_OBJECTIVES,
    QUERY_RENDERINGS,
    SCHEMA_VERSION,
    STUDY_ROLES,
    ArtifactLineage,
    CandidateEvidence,
    CanonicalQueryRecord,
    CompiledQueryBundle,
    OffsetOverlapEvidence,
    QueryEvidence,
    QueryBundle,
    RenderedQueryRecord,
    RenderedQueryBundle,
    RenderingSetting,
    decode_reference_input,
)

type ArtifactKind = Literal["compiled", "rendered"]
type BundleKind = Literal["compiled", "rendered"]

COMPILED_RECORDS_NAME = "canonical_queries.jsonl"
COMPILED_MANIFEST_NAME = "compiled_bundle.json"
RENDERED_RECORDS_NAME = "queries.jsonl"
RENDERED_MANIFEST_NAME = "rendered_bundle.json"

_ENVELOPE_KEYS = {"schema_version", "record_sha256"}
_COMPILED_BUNDLE_FIELDS = frozenset(
    field.name for field in fields(CompiledQueryBundle)
) - {"records"}
_RENDERED_BUNDLE_FIELDS = frozenset(
    field.name for field in fields(RenderedQueryBundle)
) - {"records"}
_MANIFEST_FIELDS = frozenset(
    {
        "bundle_sha256",
        "bundle_type",
        "record_count",
        "record_hashes",
        "records_file",
        "records_sha256",
    }
)
_CANONICAL_FIELDS = frozenset(field.name for field in fields(CanonicalQueryRecord))
_RENDERED_FIELDS = frozenset(field.name for field in fields(RenderedQueryRecord))
_OVERLAP_EVIDENCE_FIELDS = frozenset(
    field.name for field in fields(OffsetOverlapEvidence)
)
_OBJECTIVES = {
    "model_completion": {"completion_target", "joint_sequence"},
    "reference_answer": {"completion_target", "joint_sequence"},
    "prompt_only": {"prompt_next_token"},
}
_RENDERED_AXES = (
    "probe condition objective selection_rule rendering study_role "
    "tokenizer_id tokenizer_revision"
).split()
_BINDING_AXES = "probe condition objective selection_rule rendering study_role".split()


@dataclass(frozen=True)
class BundleArtifact:
    records_bytes: bytes
    manifest: dict[str, object]
    manifest_bytes: bytes
    bundle_sha256: str


class _CanonicalDetails(TypedDict):
    reference_basis: str | None
    reference_index: int | None
    reference_continuation: str | None
    reference_source_revision: str | None
    reference_source_sha256: str | None
    reference_input_kind: str | None
    reference_input_payload: dict[str, object] | None
    reference_candidate_continuations: tuple[str, ...] | None
    base_completion_index: int | None
    base_completion_correct: bool | None
    base_completion_model_id: str | None
    base_completion_model_revision: str | None
    base_completion_selection_rule: str | None


def build_compiled_artifact(bundle: CompiledQueryBundle) -> BundleArtifact:
    return _build_artifact(bundle, "compiled", COMPILED_RECORDS_NAME)


def build_rendered_artifact(bundle: RenderedQueryBundle) -> BundleArtifact:
    return _build_artifact(bundle, "rendered", RENDERED_RECORDS_NAME)


def compiled_bundle_sha256(bundle: CompiledQueryBundle) -> str:
    return build_compiled_artifact(bundle).bundle_sha256


def validate_rendering_settings(settings: Mapping[str, object]) -> int:
    expected = {
        "add_special_tokens": True,
        "padding": False,
        "truncation": False,
        "return_offsets_mapping": True,
        "return_special_tokens_mask": True,
        "return_attention_mask": False,
        "clean_up_tokenization_spaces": False,
    }
    if set(settings) != {*expected, "max_length"}:
        raise ValueError("Rendered bundle has invalid rendering settings")
    for key, value in expected.items():
        if type(settings[key]) is not bool:
            raise ValueError(f"Rendered setting {key} must be an exact boolean")
        if settings[key] is not value:
            raise ValueError("Rendered bundle has invalid rendering settings")
    limit = settings["max_length"]
    if type(limit) is not int or not 1 <= limit <= 4096:
        raise ValueError("Rendered max_length must be between 1 and 4096")
    return limit


def validate_lineage(lineage: tuple[ArtifactLineage, ...]) -> None:
    if not lineage:
        raise ValueError("Canonical record requires artifact lineage")
    for item in lineage:
        if type(item) is not ArtifactLineage:
            raise ValueError("Canonical record has invalid artifact lineage")
        if type(item.artifact_id) is not str or not item.artifact_id.strip():
            raise ValueError("Canonical artifact lineage requires an ID")
        if type(item.uri) is not str or not item.uri.strip():
            raise ValueError("Canonical artifact lineage requires a URI")
        _validate_sha256(item.sha256, "artifact lineage hash")


def _build_artifact(
    bundle: QueryBundle, kind: ArtifactKind, records_name: str
) -> BundleArtifact:
    data = jsonable(bundle)
    if type(data) is not dict:
        raise TypeError("Bundle serialization must produce a mapping")
    record_data = data.pop("records")
    if type(record_data) is not list:
        raise TypeError("Bundle records must serialize to a list")
    rows = [_record_envelope(record) for record in record_data]
    records_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    manifest = _manifest(data, rows, records_bytes, records_name, kind)
    bundle_hash = sha256_json(manifest)
    manifest["bundle_sha256"] = bundle_hash
    return BundleArtifact(
        records_bytes,
        manifest,
        canonical_json_bytes(manifest) + b"\n",
        bundle_hash,
    )


def _record_envelope(record: object) -> dict[str, object]:
    payload = jsonable(record)
    if type(payload) is not dict:
        raise TypeError("Record serialization must produce a mapping")
    return {
        "schema_version": SCHEMA_VERSION,
        **payload,
        "record_sha256": sha256_json(payload),
    }


def _manifest(
    data: dict[str, object],
    rows: list[dict[str, object]],
    records_bytes: bytes,
    records_name: str,
    kind: ArtifactKind,
) -> dict[str, object]:
    return {
        **data,
        "bundle_type": kind,
        "records_file": records_name,
        "record_count": len(rows),
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "record_hashes": [row["record_sha256"] for row in rows],
    }


def require_keys(
    payload: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if missing := expected - payload.keys():
        raise ValueError(f"Invalid {context} schema: missing {sorted(missing)}")
    if extra := payload.keys() - expected:
        raise ValueError(f"Invalid {context} schema: unknown {sorted(extra)}")


def required_field[T](
    payload: Mapping[str, object], key: str, expected: type[T], context: str
) -> T:
    value = payload[key]
    if type(value) is not expected:
        raise ValueError(f"Invalid {context} schema: field {key!r}")
    return cast(T, value)


def optional_field[T](
    payload: Mapping[str, object], key: str, expected: type[T], context: str
) -> T | None:
    value = payload[key]
    if value is None:
        return None
    if type(value) is not expected:
        raise ValueError(f"Invalid {context} schema: field {key!r}")
    return cast(T, value)


def list_field[T](
    payload: Mapping[str, object], key: str, item_type: type[T], context: str
) -> list[T]:
    value = required_field(payload, key, list, context)
    if any(type(item) is not item_type for item in value):
        raise ValueError(f"Invalid {context} schema: field {key!r} items")
    return cast(list[T], value)


def _optional_string_tuple(
    payload: Mapping[str, object], key: str, context: str
) -> tuple[str, ...] | None:
    value = payload[key]
    if value is None:
        return None
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"Invalid {context} schema: field {key!r} items")
    return tuple(value)


def object_list_field(
    payload: Mapping[str, object], key: str, context: str
) -> list[dict[str, object]]:
    value = list_field(payload, key, dict, context)
    if any(any(type(name) is not str for name in item) for item in value):
        raise ValueError(f"Invalid {context} schema: field {key!r} object keys")
    return cast(list[dict[str, object]], value)


def choice[T](
    payload: Mapping[str, object], key: str, choices: tuple[T, ...], context: str
) -> T:
    value = required_field(payload, key, str, context)
    for selected in choices:
        if value == selected:
            return selected
    raise ValueError(f"Invalid {context} schema: field {key!r}")


def settings_field(
    payload: Mapping[str, object], context: str
) -> dict[str, RenderingSetting]:
    value = required_field(payload, "rendering_settings", dict, context)
    allowed = (str, int, bool, type(None))
    if any(type(key) is not str for key in value) or any(
        type(item) not in allowed for item in value.values()
    ):
        raise ValueError(f"Invalid {context} schema: field 'rendering_settings'")
    return cast(dict[str, RenderingSetting], value)


def validate_manifest(payload: Mapping[str, object], kind: BundleKind) -> None:
    context = f"{kind} manifest"
    bundle_fields = (
        _COMPILED_BUNDLE_FIELDS if kind == "compiled" else _RENDERED_BUNDLE_FIELDS
    )
    require_keys(payload, bundle_fields | _MANIFEST_FIELDS, context)
    required_field(payload, "record_count", int, "record_count")
    list_field(payload, "record_hashes", str, context)
    for key in ("bundle_sha256", "bundle_type", "records_file", "records_sha256"):
        required_field(payload, key, str, context)
    required_field(payload, "probe", str, context)
    choice(payload, "condition", QUERY_CONDITIONS, context)
    choice(payload, "objective", QUERY_OBJECTIVES, context)
    optional_field(payload, "selection_rule", str, context)
    choice(payload, "rendering", QUERY_RENDERINGS, context)
    choice(payload, "study_role", STUDY_ROLES, context)
    required_field(payload, "schema_version", str, context)
    if kind == "compiled":
        choice(payload, "completion_tie_policy", COMPLETION_TIE_POLICIES, context)
    if kind == "rendered":
        for key in ("tokenizer_id", "tokenizer_revision", "compiled_bundle_sha256"):
            required_field(payload, key, str, context)
        settings_field(payload, context)


def decode_artifact_lineage(payload: Mapping[str, object]) -> ArtifactLineage:
    context = "artifact lineage"
    require_keys(
        payload, frozenset({"artifact_id", "uri", "sha256", "revision"}), context
    )
    return ArtifactLineage(
        artifact_id=required_field(payload, "artifact_id", str, context),
        uri=required_field(payload, "uri", str, context),
        sha256=required_field(payload, "sha256", str, context),
        revision=optional_field(payload, "revision", str, context),
    )


def _details(payload: Mapping[str, object], context: str) -> _CanonicalDetails:
    return {
        "reference_basis": optional_field(payload, "reference_basis", str, context),
        "reference_index": optional_field(payload, "reference_index", int, context),
        "reference_continuation": optional_field(
            payload, "reference_continuation", str, context
        ),
        "reference_source_revision": optional_field(
            payload, "reference_source_revision", str, context
        ),
        "reference_source_sha256": optional_field(
            payload, "reference_source_sha256", str, context
        ),
        "reference_input_kind": optional_field(
            payload, "reference_input_kind", str, context
        ),
        "reference_input_payload": optional_field(
            payload, "reference_input_payload", dict, context
        ),
        "reference_candidate_continuations": _optional_string_tuple(
            payload, "reference_candidate_continuations", context
        ),
        "base_completion_index": optional_field(
            payload, "base_completion_index", int, context
        ),
        "base_completion_correct": optional_field(
            payload, "base_completion_correct", bool, context
        ),
        "base_completion_model_id": optional_field(
            payload, "base_completion_model_id", str, context
        ),
        "base_completion_model_revision": optional_field(
            payload, "base_completion_model_revision", str, context
        ),
        "base_completion_selection_rule": optional_field(
            payload, "base_completion_selection_rule", str, context
        ),
    }


def decode_canonical_record(payload: Mapping[str, object]) -> CanonicalQueryRecord:
    context = "canonical record"
    require_keys(payload, _CANONICAL_FIELDS, context)
    lineage = tuple(
        decode_artifact_lineage(item)
        for item in object_list_field(payload, "lineage", context)
    )
    details = _details(payload, context)
    return CanonicalQueryRecord(
        query_id=required_field(payload, "query_id", str, context),
        probe=required_field(payload, "probe", str, context),
        task_core=required_field(payload, "task_core", str, context),
        prompt=required_field(payload, "prompt", str, context),
        target=optional_field(payload, "target", str, context),
        target_source=optional_field(payload, "target_source", str, context),
        condition=choice(payload, "condition", QUERY_CONDITIONS, context),
        selection_rule=optional_field(payload, "selection_rule", str, context),
        lineage=lineage,
        base_completion_tie_policy=choice(
            payload, "base_completion_tie_policy", COMPLETION_TIE_POLICIES, context
        ),
        **details,
    )


def _offset_mapping(
    payload: Mapping[str, object], context: str
) -> tuple[tuple[int, int], ...]:
    pairs = list_field(payload, "offset_mapping", list, context)
    if any(
        len(pair) != 2 or any(type(value) is not int for value in pair)
        for pair in pairs
    ):
        raise ValueError(f"Invalid {context} schema: field 'offset_mapping'")
    return tuple((pair[0], pair[1]) for pair in pairs)


def _offset_overlap_evidence(
    payload: Mapping[str, object], context: str
) -> tuple[OffsetOverlapEvidence, ...]:
    result: list[OffsetOverlapEvidence] = []
    for item in object_list_field(payload, "offset_overlap_evidence", context):
        evidence_context = "rendered offset overlap evidence"
        require_keys(item, _OVERLAP_EVIDENCE_FIELDS, evidence_context)
        result.append(
            OffsetOverlapEvidence(
                token_start=required_field(item, "token_start", int, evidence_context),
                token_end=required_field(item, "token_end", int, evidence_context),
                token_ids=tuple(list_field(item, "token_ids", int, evidence_context)),
                source_start=required_field(
                    item, "source_start", int, evidence_context
                ),
                source_end=required_field(item, "source_end", int, evidence_context),
                grouped_decode=required_field(
                    item, "grouped_decode", str, evidence_context
                ),
                individual_decodes=tuple(
                    list_field(item, "individual_decodes", str, evidence_context)
                ),
            )
        )
    return tuple(result)


def decode_rendered_record(payload: Mapping[str, object]) -> RenderedQueryRecord:
    context = "rendered record"
    require_keys(payload, _RENDERED_FIELDS, context)
    mask = tuple(list_field(payload, "special_tokens_mask", int, context))
    prompt_length = required_field(payload, "prompt_character_length", int, context)
    return RenderedQueryRecord(
        query_id=required_field(payload, "query_id", str, context),
        probe=required_field(payload, "probe", str, context),
        condition=choice(payload, "condition", QUERY_CONDITIONS, context),
        objective=choice(payload, "objective", QUERY_OBJECTIVES, context),
        selection_rule=optional_field(payload, "selection_rule", str, context),
        rendering=choice(payload, "rendering", QUERY_RENDERINGS, context),
        study_role=choice(payload, "study_role", STUDY_ROLES, context),
        canonical_record_sha256=required_field(
            payload, "canonical_record_sha256", str, context
        ),
        input_ids=tuple(list_field(payload, "input_ids", int, context)),
        labels=tuple(list_field(payload, "labels", int, context)),
        offset_mapping=_offset_mapping(payload, context),
        offset_overlap_evidence=_offset_overlap_evidence(payload, context),
        special_tokens_mask=mask,
        length=required_field(payload, "length", int, context),
        loss_token_start=required_field(payload, "loss_token_start", int, context),
        loss_token_end=required_field(payload, "loss_token_end", int, context),
        prompt_character_length=prompt_length,
        text_character_length=required_field(
            payload, "text_character_length", int, context
        ),
        tokenizer_id=required_field(payload, "tokenizer_id", str, context),
        tokenizer_revision=required_field(payload, "tokenizer_revision", str, context),
    )


def decode_compiled_bundle(
    payload: Mapping[str, object], records: tuple[CanonicalQueryRecord, ...]
) -> CompiledQueryBundle:
    context = "compiled manifest"
    return CompiledQueryBundle(
        probe=required_field(payload, "probe", str, context),
        condition=choice(payload, "condition", QUERY_CONDITIONS, context),
        objective=choice(payload, "objective", QUERY_OBJECTIVES, context),
        selection_rule=optional_field(payload, "selection_rule", str, context),
        completion_tie_policy=choice(
            payload, "completion_tie_policy", COMPLETION_TIE_POLICIES, context
        ),
        rendering=choice(payload, "rendering", QUERY_RENDERINGS, context),
        study_role=choice(payload, "study_role", STUDY_ROLES, context),
        records=records,
        schema_version=required_field(payload, "schema_version", str, context),
    )


def decode_rendered_bundle(
    payload: Mapping[str, object], records: tuple[RenderedQueryRecord, ...]
) -> RenderedQueryBundle:
    context = "rendered manifest"
    return RenderedQueryBundle(
        probe=required_field(payload, "probe", str, context),
        condition=choice(payload, "condition", QUERY_CONDITIONS, context),
        objective=choice(payload, "objective", QUERY_OBJECTIVES, context),
        selection_rule=optional_field(payload, "selection_rule", str, context),
        rendering=choice(payload, "rendering", QUERY_RENDERINGS, context),
        study_role=choice(payload, "study_role", STUDY_ROLES, context),
        tokenizer_id=required_field(payload, "tokenizer_id", str, context),
        tokenizer_revision=required_field(payload, "tokenizer_revision", str, context),
        compiled_bundle_sha256=required_field(
            payload, "compiled_bundle_sha256", str, context
        ),
        rendering_settings=settings_field(payload, context),
        records=records,
        schema_version=required_field(payload, "schema_version", str, context),
    )


def iter_jsonl_objects(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line == "" or line.isspace():
                continue
            yield parse_json_object(line, path, line_number)


def read_jsonl_objects(path: Path) -> list[dict[str, object]]:
    return list(iter_jsonl_objects(path))


def count_jsonl_objects(path: Path) -> int:
    return sum(1 for _ in iter_jsonl_objects(path))


def write_compiled_query_bundle(bundle: CompiledQueryBundle, output_dir: Path) -> str:
    return _write_bundle(bundle, output_dir, "compiled")


def read_compiled_query_bundle(path: Path) -> CompiledQueryBundle:
    manifest, rows = _read_bundle(
        path, COMPILED_MANIFEST_NAME, COMPILED_RECORDS_NAME, "compiled"
    )
    records = tuple(decode_canonical_record(row) for row in rows)
    bundle = decode_compiled_bundle(manifest, records)
    validate_compiled_bundle(bundle)
    return bundle


def write_rendered_query_bundle(bundle: RenderedQueryBundle, output_dir: Path) -> str:
    compiled = read_compiled_query_bundle(output_dir)
    validate_rendered_binding(bundle, compiled)
    return _write_bundle(bundle, output_dir, "rendered")


def read_rendered_query_bundle(path: Path) -> RenderedQueryBundle:
    manifest, rows = _read_bundle(
        path, RENDERED_MANIFEST_NAME, RENDERED_RECORDS_NAME, "rendered"
    )
    records = tuple(decode_rendered_record(row) for row in rows)
    bundle = decode_rendered_bundle(manifest, records)
    validate_rendered_bundle(bundle)
    root = path if path.is_dir() else path.parent
    validate_rendered_binding(bundle, read_compiled_query_bundle(root))
    return bundle


def _write_bundle(bundle: QueryBundle, root: Path, kind: ArtifactKind) -> str:
    if kind == "compiled" and isinstance(bundle, CompiledQueryBundle):
        validate_compiled_bundle(bundle)
        artifact = build_compiled_artifact(bundle)
        names = (COMPILED_RECORDS_NAME, COMPILED_MANIFEST_NAME)
    elif kind == "rendered" and isinstance(bundle, RenderedQueryBundle):
        validate_rendered_bundle(bundle)
        artifact = build_rendered_artifact(bundle)
        names = (RENDERED_RECORDS_NAME, RENDERED_MANIFEST_NAME)
    else:
        raise TypeError("Bundle type does not match requested artifact kind")
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write(root / names[0], artifact.records_bytes)
    _atomic_write(root / names[1], artifact.manifest_bytes)
    return artifact.bundle_sha256


def _read_bundle(
    path: Path, manifest_name: str, records_name: str, kind: BundleKind
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest_path = path / manifest_name if path.is_dir() else path
    manifest = _read_object(manifest_path)
    validate_manifest(manifest, kind)
    require_equal(manifest.get("schema_version"), SCHEMA_VERSION, "schema version")
    require_equal(manifest.get("bundle_type"), kind, "bundle type")
    require_equal(manifest.get("records_file"), records_name, "records file name")
    claimed_bundle_hash = manifest.get("bundle_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    require_equal(claimed_bundle_hash, sha256_json(unsigned), "bundle hash")
    records_path = manifest_path.parent / records_name
    require_equal(
        manifest.get("records_sha256"), sha256_file(records_path), "file hash"
    )
    rows, record_hashes = _read_records(records_path)
    require_equal(manifest.get("record_count"), len(rows), "record count")
    require_equal(manifest.get("record_hashes"), record_hashes, "record hashes")
    return manifest, rows


def _read_records(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    hashes: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            envelope = parse_json_object(line, path, line_number)
            if envelope.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported record schema version at {path}:{line_number}"
                )
            claimed = envelope.get("record_sha256")
            payload = {
                key: value
                for key, value in envelope.items()
                if key not in _ENVELOPE_KEYS
            }
            require_equal(claimed, sha256_json(payload), "record hash")
            rows.append(payload)
            hashes.append(str(claimed))
    return rows, hashes


def _read_object(path: Path) -> dict[str, object]:
    return parse_json_object(path.read_text(encoding="utf-8"), path, 1)


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_bytes(content)
    temporary.replace(path)


def validate_compiled_bundle(bundle: CompiledQueryBundle) -> None:
    if type(bundle) is not CompiledQueryBundle:
        raise ValueError("Expected a compiled query bundle")
    contract = _validate_bundle_axes(bundle)
    _validate_ids(bundle, contract.expected_count)
    counts = Counter(record.task_core for record in bundle.records)
    if dict(counts) != dict(contract.task_counts):
        raise ValueError("Compiled bundle task counts do not match probe contract")
    for record in bundle.records:
        _validate_canonical_record(bundle, contract, record)
    checkpoints = {
        (record.base_completion_model_id, record.base_completion_model_revision)
        for record in bundle.records
    }
    if len(checkpoints) != 1:
        raise ValueError("Compiled bundle requires a uniform base checkpoint")


def validate_rendered_bundle(bundle: RenderedQueryBundle) -> None:
    if type(bundle) is not RenderedQueryBundle:
        raise ValueError("Expected a rendered query bundle")
    contract = _validate_bundle_axes(bundle)
    _validate_ids(bundle, contract.expected_count)
    _nonempty(bundle.tokenizer_id, "tokenizer ID")
    _nonempty(bundle.tokenizer_revision, "tokenizer revision")
    _validate_sha256(bundle.compiled_bundle_sha256, "compiled bundle hash")
    limit = validate_rendering_settings(bundle.rendering_settings)
    for record in bundle.records:
        for axis in _RENDERED_AXES:
            if getattr(record, axis) != getattr(bundle, axis):
                raise ValueError(f"Bundle requires a homogeneous {axis}")
        _validate_sha256(record.canonical_record_sha256, "canonical record hash")
        validate_rendered_record(bundle, record, limit)


def validate_rendered_binding(
    rendered: RenderedQueryBundle,
    compiled: CompiledQueryBundle,
) -> None:
    validate_compiled_bundle(compiled)
    validate_rendered_bundle(rendered)
    expected_hash = compiled_bundle_sha256(compiled)
    if rendered.compiled_bundle_sha256 != expected_hash:
        raise ValueError("Rendered bundle does not bind the compiled artifact hash")
    for axis in _BINDING_AXES:
        if getattr(rendered, axis) != getattr(compiled, axis):
            raise ValueError(f"Rendered and compiled bundles differ on {axis}")
    rendered_ids = tuple(record.query_id for record in rendered.records)
    compiled_ids = tuple(record.query_id for record in compiled.records)
    if rendered_ids != compiled_ids:
        raise ValueError("Rendered and compiled query IDs or order differ")
    for output, source in zip(rendered.records, compiled.records, strict=True):
        _validate_record_binding(output, source)


def validate_rendered_record(
    bundle: RenderedQueryBundle,
    record: RenderedQueryRecord,
    limit: int,
) -> None:
    ids, labels = _validate_sequences(record, limit)
    _validate_character_lengths(record)
    _validate_offsets(record)
    expected = _expected_labels(bundle.objective, record, ids)
    if labels != expected:
        label = (
            "completion-target"
            if bundle.objective == "completion_target"
            else "full-sequence"
        )
        raise ValueError(f"Rendered {label} labels do not match boundary evidence")
    _validate_span(record, expected)


def _validate_bundle_axes(bundle: QueryBundle) -> contracts.ProbeContract:
    if bundle.schema_version != SCHEMA_VERSION or bundle.rendering != "canonical_plain":
        raise ValueError("Bundle has invalid schema or rendering")
    if bundle.condition not in tuple(_OBJECTIVES):
        raise ValueError("Bundle has an invalid condition")
    if bundle.objective not in tuple(_OBJECTIVES[bundle.condition]):
        raise ValueError("Bundle has an invalid condition and objective pair")
    contract = contracts.get_probe_contract(bundle.probe)
    if (
        isinstance(bundle, CompiledQueryBundle)
        and bundle.completion_tie_policy != contract.completion_tie_policy
    ):
        raise ValueError("Compiled bundle tie policy does not match probe contract")
    primary = contract.primary_rule
    rule = bundle.selection_rule
    if bundle.condition == "model_completion":
        _nonempty(rule, "selection rule")
        declared = (primary, *contract.allowed_sensitivities)
        if rule not in declared:
            raise ValueError("Bundle selection rule is not declared by the probe")
    elif rule is not None:
        raise ValueError("Selection rule is only valid for Model-completion")
    if bundle.objective == "joint_sequence" and rule not in (None, primary):
        raise ValueError("Joint-sequence cannot combine with a selection sensitivity")
    sensitivity = bundle.objective == "joint_sequence" or rule not in (None, primary)
    expected_role = "sensitivity" if sensitivity else "primary"
    if bundle.study_role != expected_role:
        raise ValueError("Bundle has an invalid derived study role")
    return contract


def _validate_ids(bundle: QueryBundle, expected_count: int) -> None:
    ids = [record.query_id for record in bundle.records]
    if any(type(query_id) is not str or not query_id for query_id in ids):
        raise ValueError("Bundle has an invalid query ID")
    if len(ids) != len(set(ids)):
        raise ValueError("Bundle contains duplicate query_id values")
    if len(bundle.records) != expected_count:
        raise ValueError("Bundle record count does not match probe contract")
    if ids != sorted(ids):
        raise ValueError("Bundle query IDs must be sorted")


def _validate_canonical_record(
    bundle: CompiledQueryBundle,
    contract: contracts.ProbeContract,
    record: CanonicalQueryRecord,
) -> None:
    axes = ("probe", "condition", "selection_rule")
    for axis in axes:
        if getattr(record, axis) != getattr(bundle, axis):
            raise ValueError(f"Bundle requires a homogeneous {axis}")
    if record.task_core not in contract.task_counts:
        raise ValueError("Compiled record has an undeclared task core")
    if record.base_completion_tie_policy != bundle.completion_tie_policy:
        raise ValueError("Canonical base completion tie policy is inconsistent")
    _nonempty(record.prompt, "canonical prompt")
    validate_lineage(record.lineage)
    expected_rule = (
        bundle.selection_rule
        if bundle.condition == "model_completion"
        else contract.primary_rule
    )
    _validate_base(record, expected_rule)
    if bundle.condition == "model_completion":
        _target(record, "base_completion")
        _no_reference(record)
    elif bundle.condition == "reference_answer":
        _target(record, "benchmark_reference")
        _validate_reference(record, contract)
    else:
        if record.target is not None or record.target_source is not None:
            raise ValueError("Prompt-only record cannot have a target")
        _no_reference(record)


def _validate_base(record: CanonicalQueryRecord, expected_rule: str | None) -> None:
    index = record.base_completion_index
    if type(index) is not int or index < 0:
        raise ValueError("Canonical base completion index must be nonnegative")
    if type(record.base_completion_correct) is not bool:
        raise ValueError("Canonical base completion correctness must be boolean")
    for value, label in (
        (record.base_completion_model_id, "base model ID"),
        (record.base_completion_model_revision, "base model revision"),
        (record.base_completion_selection_rule, "base selection rule"),
    ):
        _nonempty(value, label)
    if record.base_completion_selection_rule != expected_rule:
        raise ValueError("Canonical base selection rule is inconsistent")


def _validate_reference(
    record: CanonicalQueryRecord, contract: contracts.ProbeContract
) -> None:
    if record.target != record.reference_continuation:
        raise ValueError("Canonical reference target differs from continuation")
    if type(record.reference_index) is not int or record.reference_index < 0:
        raise ValueError("Canonical reference index must be nonnegative")
    rule = contract.reference_rule(record.task_core)
    if record.reference_basis != rule.policy:
        raise ValueError("Canonical reference basis does not match contract")
    _nonempty(record.reference_continuation, "reference continuation")
    _nonempty(record.reference_source_revision, "reference source revision")
    _validate_sha256(record.reference_source_sha256, "reference source hash")
    anchors = [
        item for item in record.lineage if item.artifact_id == "reference_source"
    ]
    valid_anchor = len(anchors) == 1 and (
        anchors[0].uri == f"reference://{record.reference_basis}"
        and anchors[0].sha256 == record.reference_source_sha256
        and anchors[0].revision == record.reference_source_revision
    )
    if not valid_anchor:
        raise ValueError("Canonical reference source does not match lineage")
    _validate_reference_snapshot(record, contract)


def _validate_reference_snapshot(
    record: CanonicalQueryRecord, contract: contracts.ProbeContract
) -> None:
    kind = record.reference_input_kind
    payload = record.reference_input_payload
    continuations = record.reference_candidate_continuations
    if type(kind) is not str or not isinstance(payload, Mapping):
        raise ValueError("Canonical reference input snapshot is missing")
    if type(continuations) is not tuple or not continuations:
        raise ValueError("Canonical reference candidates are missing")
    if any(type(value) is not str or not value for value in continuations):
        raise ValueError("Canonical reference candidates are invalid")
    source = decode_reference_input(kind, payload)
    evidence = QueryEvidence(
        query_id=record.query_id,
        probe=record.probe,
        task_core=record.task_core,
        native_id=record.query_id,
        prompt=record.prompt,
        candidates=tuple(
            CandidateEvidence(index, continuation, {})
            for index, continuation in enumerate(continuations)
        ),
        model_id=record.base_completion_model_id or "serialized-reference",
        model_revision=record.base_completion_model_revision or "serialized-reference",
        task_revision="serialized-reference",
        lineage=record.lineage,
        reference_index=record.reference_index,
    )
    selected = contracts.resolve_reference(evidence, source, contract=contract)
    if (
        selected.candidate_index != record.reference_index
        or selected.continuation != record.reference_continuation
        or selected.basis != record.reference_basis
        or selected.source.revision != record.reference_source_revision
        or selected.source.artifact_sha256 != record.reference_source_sha256
    ):
        raise ValueError("Canonical reference resolution does not match source facts")


def _no_reference(record: CanonicalQueryRecord) -> None:
    values = (
        record.reference_basis,
        record.reference_index,
        record.reference_continuation,
        record.reference_source_revision,
        record.reference_source_sha256,
        record.reference_input_kind,
        record.reference_input_payload,
        record.reference_candidate_continuations,
    )
    if any(value is not None for value in values):
        raise ValueError("Canonical record has unexpected reference metadata")
    if any(item.artifact_id == "reference_source" for item in record.lineage):
        raise ValueError("Non-reference record has reference source lineage")


def _target(record: CanonicalQueryRecord, source: str) -> None:
    _nonempty(record.target, "canonical target")
    if record.target_source != source:
        raise ValueError("Canonical target source is inconsistent")


def _validate_record_binding(
    rendered: RenderedQueryRecord,
    compiled: CanonicalQueryRecord,
) -> None:
    if rendered.canonical_record_sha256 != sha256_json(compiled):
        raise ValueError("Rendered record does not bind its canonical record hash")
    text = compiled.prompt + (compiled.target or "")
    if rendered.prompt_character_length != len(compiled.prompt):
        raise ValueError(
            "Rendered prompt character length differs from canonical prompt"
        )
    if rendered.text_character_length != len(text):
        raise ValueError("Rendered text character length differs from canonical text")
    _validate_bound_offset_overlaps(rendered, text)


def _validate_bound_offset_overlaps(record: RenderedQueryRecord, text: str) -> None:
    previous = 0
    evidence = zip(record.offset_mapping, record.special_tokens_mask, strict=True)
    for (start, end), special in evidence:
        if special:
            continue
        if start < previous and len(text[start:end].encode("utf-8")) == 1:
            raise ValueError("Rendered offset overlap lacks multibyte source text")
        previous = max(previous, end)
    for proof in record.offset_overlap_evidence:
        if proof.grouped_decode != text[proof.source_start : proof.source_end]:
            raise ValueError("Rendered overlap evidence differs from canonical text")


def _validate_sequences(
    record: RenderedQueryRecord, limit: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ids, labels = record.input_ids, record.labels
    if not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("Rendered input IDs must be nonempty and nonnegative")
    if type(record.length) is not int or record.length != len(ids):
        raise ValueError("Rendered declared length does not match input IDs")
    if len(ids) > limit or len(labels) != len(ids):
        raise ValueError("Rendered inputs exceed limit or labels have wrong length")
    if any(type(label) is not int for label in labels):
        raise ValueError("Rendered labels must be integers")
    if labels[0] != -100:
        raise ValueError("Rendered label position zero must be ignored")
    return ids, labels


def _validate_character_lengths(record: RenderedQueryRecord) -> None:
    prompt = record.prompt_character_length
    text = record.text_character_length
    if type(prompt) is not int or type(text) is not int or not 0 < prompt <= text:
        raise ValueError("Rendered character lengths are invalid")
    target_bearing = record.condition in ("model_completion", "reference_answer")
    if target_bearing != (text > prompt):
        raise ValueError("Rendered target length disagrees with condition")


def _validate_offsets(record: RenderedQueryRecord) -> None:
    offsets, mask = record.offset_mapping, record.special_tokens_mask
    if len(offsets) != record.length or len(mask) != record.length:
        raise ValueError("Rendered boundary evidence lengths do not match tokens")
    if any(type(value) is not int or value not in (0, 1) for value in mask):
        raise ValueError("Rendered special-token mask must contain zero or one")
    nonspecial = [index for index, value in enumerate(mask) if not value]
    if not nonspecial:
        raise ValueError("Rendered boundary evidence has no text tokens")
    first, last = nonspecial[0], nonspecial[-1]
    if any(mask[index] for index in range(first, last + 1)):
        raise ValueError("Rendered boundary evidence has an internal special token")
    _validate_offset_values(record)
    _validate_overlap_evidence(record, nonspecial)


def _validate_overlap_evidence(
    record: RenderedQueryRecord, nonspecial: list[int]
) -> None:
    groups = _structural_overlap_groups(record, nonspecial)
    proofs = record.offset_overlap_evidence
    if len(groups) != len(proofs):
        raise ValueError("Rendered overlap evidence does not match offset groups")
    for proof, expected in zip(proofs, groups, strict=True):
        if type(proof) is not OffsetOverlapEvidence:
            raise ValueError("Rendered overlap evidence does not match offset groups")
        token_start, token_end, _, _ = expected
        coordinates = (
            proof.token_start,
            proof.token_end,
            proof.source_start,
            proof.source_end,
        )
        if coordinates != expected:
            raise ValueError("Rendered overlap evidence does not match offset groups")
        if (
            proof.token_ids != record.input_ids[token_start:token_end]
            or type(proof.grouped_decode) is not str
            or not proof.grouped_decode
            or len(proof.individual_decodes) != token_end - token_start
            or any(
                type(decoded) is not str or "�" not in decoded
                for decoded in proof.individual_decodes
            )
        ):
            raise ValueError("Rendered overlap evidence is invalid")


def _structural_overlap_groups(
    record: RenderedQueryRecord, nonspecial: list[int]
) -> tuple[tuple[int, int, int, int], ...]:
    group_start = nonspecial[0]
    previous = record.offset_mapping[group_start][1]
    has_overlap = False
    groups: list[tuple[int, int, int, int]] = []
    for index in nonspecial[1:]:
        start, end = record.offset_mapping[index]
        if start < previous:
            has_overlap = True
        else:
            if has_overlap:
                groups.append(
                    (
                        group_start,
                        index,
                        record.offset_mapping[group_start][0],
                        previous,
                    )
                )
            group_start = index
            has_overlap = False
        previous = max(previous, end)
    if has_overlap:
        groups.append(
            (
                group_start,
                nonspecial[-1] + 1,
                record.offset_mapping[group_start][0],
                previous,
            )
        )
    return tuple(groups)


def _validate_offset_values(record: RenderedQueryRecord) -> None:
    previous = 0
    for index, pair in enumerate(record.offset_mapping):
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("Rendered offset must be an integer pair")
        start, end = pair
        if type(start) is not int or type(end) is not int:
            raise ValueError("Rendered offset must be an integer pair")
        if record.special_tokens_mask[index]:
            if pair != (0, 0):
                raise ValueError("Rendered special token has ambiguous offset")
            continue
        previous = _validate_text_offset(record, start, end, previous)
    if previous != record.text_character_length:
        raise ValueError("Rendered offsets do not cover canonical text")


def _validate_text_offset(
    record: RenderedQueryRecord, start: int, end: int, previous: int
) -> int:
    if start < 0 or end < 0:
        raise ValueError("Rendered offsets must be nonnegative")
    if (start, end) == (0, 0) or end <= start:
        raise ValueError("Rendered non-special token has an invalid zero offset")
    if start < previous:
        if previous == 0 or end != previous or start != end - 1:
            raise ValueError("Rendered offsets overlap ambiguously")
        return previous
    if start > previous or end > record.text_character_length:
        raise ValueError("Rendered offsets contain a gap or exceed text")
    boundary = record.prompt_character_length
    if start < boundary < end:
        raise ValueError("Rendered token crosses prompt-target boundary")
    return end


def _expected_labels(
    objective: str,
    record: RenderedQueryRecord,
    ids: tuple[int, ...],
) -> tuple[int, ...]:
    if objective != "completion_target":
        return (-100, *ids[1:])
    labels = [-100] * len(ids)
    evidence = zip(record.offset_mapping, record.special_tokens_mask, strict=True)
    for index, ((start, _), special) in enumerate(evidence):
        if index > 0 and not special and start >= record.prompt_character_length:
            labels[index] = ids[index]
    return tuple(labels)


def _validate_span(record: RenderedQueryRecord, expected: tuple[int, ...]) -> None:
    effective = [index for index in range(1, len(expected)) if expected[index] != -100]
    if not effective:
        raise ValueError("Rendered boundary evidence yields zero effective labels")
    start, end = record.loss_token_start, record.loss_token_end
    if type(start) is not int or type(end) is not int:
        raise ValueError("Rendered loss span must contain integers")
    if (start, end) != (effective[0], effective[-1] + 1):
        raise ValueError("Rendered loss span does not match boundary evidence")
    if effective != list(range(start, end)):
        raise ValueError("Rendered boundary-derived labels are not contiguous")


def _validate_sha256(value: object, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be lowercase 64-hex SHA-256")


def _nonempty(value: object, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
