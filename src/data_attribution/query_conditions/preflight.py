"""Semantic equivalence checks for paired typed rendered query bundles.

Operational use requires Query Task 6 to persist and reread both bundles through
compiled-binding IO from stable non-symlink roots. This layer does not attest
filesystem provenance.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from data_attribution.query_conditions.compiler import get_probe_contract
from data_attribution.artifact_integrity import (
    canonical_json_bytes,
    jsonable,
    parse_json_object,
    sha256_json,
)
from data_attribution.query_conditions.io import (
    build_rendered_artifact,
    validate_rendered_bundle,
    validate_rendering_settings,
)
from data_attribution.query_conditions.types import (
    QUERY_CONDITIONS,
    QUERY_OBJECTIVES,
    QUERY_RENDERINGS,
    STUDY_ROLES,
    QueryCondition,
    QueryObjective,
    QueryRendering,
    RenderedQueryBundle,
    RenderedQueryRecord,
    RenderingSetting,
    StudyRole,
)

PREFLIGHT_SCHEMA_VERSION = "1"
CHECKED_RECORD_FIELDS = (
    "canonical_record_sha256",
    "input_ids",
    "labels",
    "length",
    "loss_token_span",
    "offset_mapping",
    "offset_overlap_evidence",
    "special_tokens_mask",
    "prompt_character_length",
    "text_character_length",
)
type FrozenSettings = tuple[tuple[str, RenderingSetting], ...]
_OBJECTIVES = {
    "model_completion": {"completion_target", "joint_sequence"},
    "reference_answer": {"completion_target", "joint_sequence"},
    "prompt_only": {"prompt_next_token"},
}

_BUNDLE_FIELDS = (
    "probe",
    "condition",
    "objective",
    "selection_rule",
    "rendering",
    "study_role",
    "compiled_bundle_sha256",
    "rendering_settings",
)


@dataclass(frozen=True)
class TokenizerEquivalenceReport:
    probe: str
    condition: QueryCondition
    objective: QueryObjective
    selection_rule: str | None
    rendering: QueryRendering
    study_role: StudyRole
    compiled_bundle_sha256: str
    rendering_settings: FrozenSettings
    record_count: int
    ordered_query_ids_sha256: str
    base_tokenizer_id: str
    base_tokenizer_revision: str
    instruct_tokenizer_id: str
    instruct_tokenizer_revision: str
    base_rendered_bundle_sha256: str
    instruct_rendered_bundle_sha256: str
    shared_rendered_bundle_sha256: str
    shared_artifact_source: Literal["base"]
    equivalent_payload_sha256: str
    checked_fields: tuple[str, ...]
    report_sha256: str
    schema_version: str = PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "rendering_settings", tuple(self.rendering_settings))
        object.__setattr__(self, "checked_fields", tuple(self.checked_fields))


def assert_olmo_tokenizer_equivalence(
    base: RenderedQueryBundle, instruct: RenderedQueryBundle
) -> TokenizerEquivalenceReport:
    """Compare typed bundle contents without attesting filesystem provenance."""
    query_ids, neutral = assert_rendered_payload_equivalence(base, instruct)
    report = _build_report(base, instruct, query_ids, neutral)
    report = replace(report, report_sha256=report_sha256(report))
    validate_tokenizer_equivalence_report(report)
    return report


def assert_rendered_payload_equivalence(
    left: RenderedQueryBundle, right: RenderedQueryBundle
) -> tuple[tuple[str, ...], str]:
    """Prove rendering equality without assigning either bundle a model role."""
    base, instruct = left, right
    validate_rendered_bundle(base)
    validate_rendered_bundle(instruct)
    _assert_bundle_match(base, instruct)
    query_ids = _assert_query_order(base, instruct)
    _assert_record_match(base, instruct)
    neutral = tokenizer_neutral_payload_sha256(base)
    if neutral != tokenizer_neutral_payload_sha256(instruct):
        raise ValueError("Tokenizer-neutral rendered payloads differ")
    return query_ids, neutral


def assert_historical_base_payload_equivalence(
    accepted_base: RenderedQueryBundle, historical_base: RenderedQueryBundle
) -> tuple[tuple[str, ...], str]:
    """Compare a historical Base reconstruction with only identity fields omitted."""
    validate_rendered_bundle(accepted_base)
    validate_rendered_bundle(historical_base)
    permitted = {"compiled_bundle_sha256"}
    differences = [
        field
        for field in _BUNDLE_FIELDS
        if field not in permitted
        and getattr(accepted_base, field) != getattr(historical_base, field)
    ]
    if differences:
        raise ValueError(
            "Historical Base bundle mismatch: fields " + ", ".join(differences)
        )
    if len(accepted_base.records) != len(historical_base.records):
        raise ValueError("Historical Base bundle mismatch: field record_count")
    query_ids = _assert_query_order(accepted_base, historical_base)
    for left, right in zip(accepted_base.records, historical_base.records, strict=True):
        for field in CHECKED_RECORD_FIELDS:
            if field == "canonical_record_sha256":
                continue
            if _record_value(left, field) != _record_value(right, field):
                raise ValueError(
                    f"Historical Base mismatch at query {left.query_id!r}: field {field}"
                )
    neutral = historical_base_identity_neutral_payload_sha256(accepted_base)
    if neutral != historical_base_identity_neutral_payload_sha256(historical_base):
        raise ValueError("Historical Base identity-neutral rendered payloads differ")
    return query_ids, neutral


def verify_tokenizer_equivalence_report(
    report: TokenizerEquivalenceReport,
    base: RenderedQueryBundle,
    instruct: RenderedQueryBundle,
) -> None:
    validate_tokenizer_equivalence_report(report)
    if report != assert_olmo_tokenizer_equivalence(base, instruct):
        raise ValueError("Tokenizer preflight report does not bind supplied bundles")


def tokenizer_neutral_payload_sha256(bundle: RenderedQueryBundle) -> str:
    payload = jsonable(bundle)
    if type(payload) is not dict:
        raise TypeError("Rendered bundle must serialize to an object")
    payload.pop("tokenizer_id")
    payload.pop("tokenizer_revision")
    records = payload.get("records")
    if type(records) is not list or any(type(item) is not dict for item in records):
        raise TypeError("Rendered records must serialize to objects")
    for item in records:
        item.pop("tokenizer_id")
        item.pop("tokenizer_revision")
    return sha256_json(payload)


def historical_base_identity_neutral_payload_sha256(
    bundle: RenderedQueryBundle,
) -> str:
    """Hash a rendering after removing the attested Base identity normalization."""
    payload = jsonable(bundle)
    if type(payload) is not dict:
        raise TypeError("Rendered bundle must serialize to an object")
    payload.pop("compiled_bundle_sha256")
    payload.pop("tokenizer_id")
    payload.pop("tokenizer_revision")
    records = payload.get("records")
    if type(records) is not list or any(type(item) is not dict for item in records):
        raise TypeError("Rendered records must serialize to objects")
    for item in records:
        item.pop("canonical_record_sha256")
        item.pop("tokenizer_id")
        item.pop("tokenizer_revision")
    return sha256_json(payload)


def _assert_bundle_match(
    base: RenderedQueryBundle, instruct: RenderedQueryBundle
) -> None:
    differences = [
        field
        for field in _BUNDLE_FIELDS
        if getattr(base, field) != getattr(instruct, field)
    ]
    if differences:
        raise ValueError(
            f"Tokenizer preflight bundle mismatch: fields {', '.join(differences)}"
        )
    if len(base.records) != len(instruct.records):
        raise ValueError("Tokenizer preflight bundle mismatch: field record_count")


def _assert_query_order(
    base: RenderedQueryBundle, instruct: RenderedQueryBundle
) -> tuple[str, ...]:
    base_ids = tuple(record.query_id for record in base.records)
    instruct_ids = tuple(record.query_id for record in instruct.records)
    if base_ids != instruct_ids:
        index = next(
            i
            for i, pair in enumerate(zip(base_ids, instruct_ids))
            if pair[0] != pair[1]
        )
        raise ValueError(
            f"Tokenizer preflight query ID/order mismatch at position {index}: "
            f"{base_ids[index]!r} != {instruct_ids[index]!r}"
        )
    return base_ids


def _assert_record_match(
    base: RenderedQueryBundle, instruct: RenderedQueryBundle
) -> None:
    for left, right in zip(base.records, instruct.records, strict=True):
        for field in CHECKED_RECORD_FIELDS:
            if _record_value(left, field) != _record_value(right, field):
                raise ValueError(
                    f"Tokenizer preflight mismatch at query {left.query_id!r}: "
                    f"field {field}"
                )


def _record_value(record: RenderedQueryRecord, field: str) -> object:
    if field == "loss_token_span":
        return record.loss_token_start, record.loss_token_end
    return getattr(record, field)


def _build_report(
    base: RenderedQueryBundle,
    instruct: RenderedQueryBundle,
    query_ids: tuple[str, ...],
    neutral: str,
) -> TokenizerEquivalenceReport:
    base_hash, instruct_hash = _artifact_hashes(base, instruct)
    return TokenizerEquivalenceReport(
        probe=base.probe,
        condition=base.condition,
        objective=base.objective,
        selection_rule=base.selection_rule,
        rendering=base.rendering,
        study_role=base.study_role,
        compiled_bundle_sha256=base.compiled_bundle_sha256,
        rendering_settings=tuple(sorted(base.rendering_settings.items())),
        record_count=len(base.records),
        ordered_query_ids_sha256=sha256_json(query_ids),
        base_tokenizer_id=base.tokenizer_id,
        base_tokenizer_revision=base.tokenizer_revision,
        instruct_tokenizer_id=instruct.tokenizer_id,
        instruct_tokenizer_revision=instruct.tokenizer_revision,
        base_rendered_bundle_sha256=base_hash,
        instruct_rendered_bundle_sha256=instruct_hash,
        shared_rendered_bundle_sha256=base_hash,
        shared_artifact_source="base",
        equivalent_payload_sha256=neutral,
        checked_fields=CHECKED_RECORD_FIELDS,
        report_sha256="",
    )


def _artifact_hashes(
    base: RenderedQueryBundle, instruct: RenderedQueryBundle
) -> tuple[str, str]:
    return (
        build_rendered_artifact(base).bundle_sha256,
        build_rendered_artifact(instruct).bundle_sha256,
    )


def report_sha256(report: TokenizerEquivalenceReport) -> str:
    payload = jsonable(report)
    if type(payload) is not dict:
        raise TypeError("Preflight report must serialize to an object")
    payload.pop("report_sha256")
    return sha256_json(payload)


def validate_tokenizer_equivalence_report(
    report: TokenizerEquivalenceReport,
) -> None:
    if type(report) is not TokenizerEquivalenceReport:
        raise ValueError("Expected a tokenizer equivalence report")
    _validate_report_axes(report)
    _validate_report_evidence(report)
    if report.report_sha256 != report_sha256(report):
        raise ValueError("Invalid tokenizer preflight report hash")


def _validate_report_axes(report: TokenizerEquivalenceReport) -> None:
    if report.schema_version != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("Unsupported tokenizer preflight schema version")
    if type(report.probe) is not str or not report.probe.strip():
        raise ValueError("Invalid tokenizer preflight probe")
    if report.condition not in QUERY_CONDITIONS:
        raise ValueError("Invalid tokenizer preflight condition")
    if report.objective not in QUERY_OBJECTIVES:
        raise ValueError("Invalid tokenizer preflight objective")
    if (
        report.rendering not in QUERY_RENDERINGS
        or report.rendering != "canonical_plain"
    ):
        raise ValueError("Invalid tokenizer preflight rendering")
    if report.study_role not in STUDY_ROLES:
        raise ValueError("Invalid tokenizer preflight study role")
    if report.selection_rule is not None and (
        type(report.selection_rule) is not str or not report.selection_rule.strip()
    ):
        raise ValueError("Invalid tokenizer preflight selection rule")
    _validate_axis_contract(report)


def _validate_axis_contract(report: TokenizerEquivalenceReport) -> None:
    contract = get_probe_contract(report.probe)
    if report.objective not in _OBJECTIVES[report.condition]:
        raise ValueError("Invalid tokenizer preflight condition and objective pair")
    rule = report.selection_rule
    if report.condition == "model_completion":
        declared = (contract.primary_rule, *contract.allowed_sensitivities)
        if rule not in declared:
            raise ValueError("Invalid tokenizer preflight selection rule contract")
    elif rule is not None:
        raise ValueError("Tokenizer preflight selection requires Model-completion")
    if report.objective == "joint_sequence" and rule not in (
        None,
        contract.primary_rule,
    ):
        raise ValueError("Joint-sequence requires the primary selection rule")
    sensitivity = report.objective == "joint_sequence" or rule not in (
        None,
        contract.primary_rule,
    )
    if report.study_role != ("sensitivity" if sensitivity else "primary"):
        raise ValueError("Invalid tokenizer preflight derived study role")


def _validate_report_evidence(report: TokenizerEquivalenceReport) -> None:
    expected_count = get_probe_contract(report.probe).expected_count
    if type(report.record_count) is not int or report.record_count != expected_count:
        raise ValueError("Invalid tokenizer preflight record_count")
    if report.checked_fields != CHECKED_RECORD_FIELDS:
        raise ValueError("Invalid tokenizer preflight checked fields")
    if report.shared_artifact_source != "base":
        raise ValueError("Invalid tokenizer preflight shared artifact source")
    if report.shared_rendered_bundle_sha256 != report.base_rendered_bundle_sha256:
        raise ValueError("Shared rendered artifact must be the base artifact")
    for value in _report_text_values(report):
        if type(value) is not str or not value.strip():
            raise ValueError("Invalid tokenizer preflight text evidence")
    for value in _report_hash_values(report):
        _validate_hash(value)
    settings = dict(report.rendering_settings)
    if tuple(sorted(settings.items())) != report.rendering_settings:
        raise ValueError("Invalid tokenizer preflight rendering settings")
    validate_rendering_settings(settings)


def _report_text_values(report: TokenizerEquivalenceReport) -> tuple[object, ...]:
    return (
        report.base_tokenizer_id,
        report.base_tokenizer_revision,
        report.instruct_tokenizer_id,
        report.instruct_tokenizer_revision,
    )


def _report_hash_values(report: TokenizerEquivalenceReport) -> tuple[object, ...]:
    return (
        report.compiled_bundle_sha256,
        report.ordered_query_ids_sha256,
        report.base_rendered_bundle_sha256,
        report.instruct_rendered_bundle_sha256,
        report.shared_rendered_bundle_sha256,
        report.equivalent_payload_sha256,
        report.report_sha256,
    )


def _validate_hash(value: object) -> None:
    if type(value) is not str or len(value) != 64:
        raise ValueError("Invalid tokenizer preflight SHA-256")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError("Invalid tokenizer preflight SHA-256")


def write_olmo_tokenizer_preflight(
    base: RenderedQueryBundle,
    instruct: RenderedQueryBundle,
    path: Path,
) -> TokenizerEquivalenceReport:
    report = assert_olmo_tokenizer_equivalence(base, instruct)
    _write_report(report, path)
    if load_verified_tokenizer_preflight(path) != report:
        raise ValueError("Written tokenizer preflight failed verification")
    return report


def load_verified_tokenizer_preflight(path: Path) -> TokenizerEquivalenceReport:
    _validate_report_input(path)
    payload = _read_payload(path)
    report = _decode_report(payload)
    validate_tokenizer_equivalence_report(report)
    return report


def decode_tokenizer_preflight(
    payload: dict[str, object],
) -> TokenizerEquivalenceReport:
    return _decode_report(payload)


def _decode_report(payload: dict[str, object]) -> TokenizerEquivalenceReport:
    return TokenizerEquivalenceReport(
        probe=_text(payload, "probe"),
        condition=_choice(payload, "condition", QUERY_CONDITIONS),
        objective=_choice(payload, "objective", QUERY_OBJECTIVES),
        selection_rule=_optional_text(payload, "selection_rule"),
        rendering=_choice(payload, "rendering", QUERY_RENDERINGS),
        study_role=_choice(payload, "study_role", STUDY_ROLES),
        compiled_bundle_sha256=_text(payload, "compiled_bundle_sha256"),
        rendering_settings=_settings(payload),
        record_count=_integer(payload, "record_count"),
        ordered_query_ids_sha256=_text(payload, "ordered_query_ids_sha256"),
        base_tokenizer_id=_text(payload, "base_tokenizer_id"),
        base_tokenizer_revision=_text(payload, "base_tokenizer_revision"),
        instruct_tokenizer_id=_text(payload, "instruct_tokenizer_id"),
        instruct_tokenizer_revision=_text(payload, "instruct_tokenizer_revision"),
        base_rendered_bundle_sha256=_text(payload, "base_rendered_bundle_sha256"),
        instruct_rendered_bundle_sha256=_text(
            payload, "instruct_rendered_bundle_sha256"
        ),
        shared_rendered_bundle_sha256=_text(payload, "shared_rendered_bundle_sha256"),
        shared_artifact_source=_choice(payload, "shared_artifact_source", ("base",)),
        equivalent_payload_sha256=_text(payload, "equivalent_payload_sha256"),
        checked_fields=_strings(payload, "checked_fields"),
        report_sha256=_text(payload, "report_sha256"),
        schema_version=_text(payload, "schema_version"),
    )


def _read_payload(path: Path) -> dict[str, object]:
    payload = parse_json_object(path.read_text(encoding="utf-8"), path, 1)
    _require_keys(payload)
    return payload


def _validate_report_input(path: Path) -> None:
    if _has_symlink_component(path):
        raise ValueError(f"Tokenizer preflight path contains a symlink: {path}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Tokenizer preflight path is not a regular file: {path}")


def _write_report(report: TokenizerEquivalenceReport, path: Path) -> None:
    validate_tokenizer_equivalence_report(report)
    if _has_symlink_component(path):
        raise ValueError(f"Tokenizer preflight path contains a symlink: {path}")
    if path.exists():
        raise ValueError(f"Refusing to overwrite tokenizer preflight: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if _has_symlink_component(path):
        raise ValueError(f"Tokenizer preflight path contains a symlink: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    temporary.write_bytes(canonical_json_bytes(report) + b"\n")
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(
            f"Refusing to overwrite tokenizer preflight: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _require_keys(payload: dict[str, object]) -> None:
    expected = {field.name for field in fields(TokenizerEquivalenceReport)}
    if set(payload) != expected:
        raise ValueError("Invalid tokenizer preflight report fields")


def _text(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if type(value) is not str:
        raise ValueError(f"Invalid tokenizer preflight field: {key}")
    return value


def _optional_text(payload: dict[str, object], key: str) -> str | None:
    value = payload[key]
    if value is not None and type(value) is not str:
        raise ValueError(f"Invalid tokenizer preflight field: {key}")
    return cast(str | None, value)


def _integer(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if type(value) is not int:
        raise ValueError(f"Invalid tokenizer preflight field: {key}")
    return value


def _strings(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload[key]
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"Invalid tokenizer preflight field: {key}")
    return tuple(value)


def _choice[Choice: str](
    payload: dict[str, object], key: str, choices: tuple[Choice, ...]
) -> Choice:
    value = _text(payload, key)
    if value not in choices:
        raise ValueError(f"Invalid tokenizer preflight field: {key}")
    return cast(Choice, value)


def _settings(
    payload: dict[str, object],
) -> tuple[tuple[str, RenderingSetting], ...]:
    value = payload["rendering_settings"]
    if type(value) is not list:
        raise ValueError("Invalid tokenizer preflight field: rendering_settings")
    result: list[tuple[str, RenderingSetting]] = []
    for pair in value:
        if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str:
            raise ValueError("Invalid tokenizer preflight field: rendering_settings")
        setting = pair[1]
        if type(setting) not in (str, int, bool, type(None)):
            raise ValueError("Invalid tokenizer preflight field: rendering_settings")
        result.append((pair[0], setting))
    return tuple(result)
