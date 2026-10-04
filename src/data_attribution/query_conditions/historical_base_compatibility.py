"""Explicit tokenizer compatibility evidence for historical and accepted Base."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from data_attribution.artifact_integrity import (
    canonical_json_bytes,
    parse_json_object,
    sha256_json,
)
from data_attribution.attribution.trackstar.preconditioner_promotion import (
    publish_bytes_absent,
)
from data_attribution.attribution.trackstar.lineage_model import (
    open_stable_regular_file,
)
from data_attribution.query_conditions.preflight import (
    CHECKED_RECORD_FIELDS,
    assert_historical_base_payload_equivalence,
)
from data_attribution.query_conditions.compiler import get_probe_contract
from data_attribution.query_conditions.io import (
    build_rendered_artifact,
    validate_rendering_settings,
    validate_rendered_binding,
)
from data_attribution.query_conditions.types import (
    CompiledQueryBundle,
    QUERY_CONDITIONS,
    QUERY_OBJECTIVES,
    QUERY_RENDERINGS,
    STUDY_ROLES,
    RenderedQueryBundle,
)

HISTORICAL_BASE_CHECKED_RECORD_FIELDS = tuple(
    field for field in CHECKED_RECORD_FIELDS if field != "canonical_record_sha256"
)


@dataclass(frozen=True)
class HistoricalBaseTokenizerCompatibilityReport:
    probe: str
    condition: str
    objective: str
    selection_rule: str | None
    rendering: str
    study_role: str
    accepted_compiled_bundle_sha256: str
    historical_compiled_bundle_sha256: str
    rendering_settings: tuple[tuple[str, object], ...]
    record_count: int
    ordered_query_ids_sha256: str
    accepted_base_tokenizer_id: str
    accepted_base_tokenizer_revision: str
    historical_base_tokenizer_id: str
    historical_base_tokenizer_revision: str
    accepted_base_rendered_bundle_sha256: str
    historical_base_rendered_bundle_sha256: str
    shared_rendered_bundle_sha256: str
    equivalent_payload_sha256: str
    checked_fields: tuple[str, ...]
    comparison_kind: str
    compatibility_sha256: str
    schema_version: str = "1"
    kind: str = "historical_base_tokenizer_compatibility"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rendering_settings",
            tuple(tuple(item) for item in self.rendering_settings),
        )
        object.__setattr__(self, "checked_fields", tuple(self.checked_fields))


def historical_base_compatibility_sha256(
    value: HistoricalBaseTokenizerCompatibilityReport,
) -> str:
    return sha256_json(replace(value, compatibility_sha256=""))


def build_historical_base_tokenizer_compatibility(
    accepted_compiled: CompiledQueryBundle,
    accepted_base: RenderedQueryBundle,
    historical_compiled: CompiledQueryBundle,
    historical_base: RenderedQueryBundle,
) -> HistoricalBaseTokenizerCompatibilityReport:
    """Compare two Base identities without introducing an Instruct identity."""
    validate_rendered_binding(accepted_base, accepted_compiled)
    validate_rendered_binding(historical_base, historical_compiled)
    if accepted_base.compiled_bundle_sha256 == historical_base.compiled_bundle_sha256:
        raise ValueError(
            "Historical Base rendering must bind an independent compilation"
        )
    _validate_historical_compilation_normalization(
        accepted_compiled, historical_compiled
    )
    query_ids, neutral = assert_historical_base_payload_equivalence(
        accepted_base, historical_base
    )
    accepted_hash = build_rendered_artifact(accepted_base).bundle_sha256
    historical_hash = build_rendered_artifact(historical_base).bundle_sha256
    unsigned = HistoricalBaseTokenizerCompatibilityReport(
        probe=accepted_base.probe,
        condition=accepted_base.condition,
        objective=accepted_base.objective,
        selection_rule=accepted_base.selection_rule,
        rendering=accepted_base.rendering,
        study_role=accepted_base.study_role,
        accepted_compiled_bundle_sha256=accepted_base.compiled_bundle_sha256,
        historical_compiled_bundle_sha256=historical_base.compiled_bundle_sha256,
        rendering_settings=tuple(sorted(accepted_base.rendering_settings.items())),
        record_count=len(accepted_base.records),
        ordered_query_ids_sha256=sha256_json(query_ids),
        accepted_base_tokenizer_id=accepted_base.tokenizer_id,
        accepted_base_tokenizer_revision=accepted_base.tokenizer_revision,
        historical_base_tokenizer_id=historical_base.tokenizer_id,
        historical_base_tokenizer_revision=historical_base.tokenizer_revision,
        accepted_base_rendered_bundle_sha256=accepted_hash,
        historical_base_rendered_bundle_sha256=historical_hash,
        shared_rendered_bundle_sha256=accepted_hash,
        equivalent_payload_sha256=neutral,
        checked_fields=HISTORICAL_BASE_CHECKED_RECORD_FIELDS,
        comparison_kind="historical_base_identity",
        compatibility_sha256="",
    )
    return replace(
        unsigned, compatibility_sha256=historical_base_compatibility_sha256(unsigned)
    )


def write_historical_base_tokenizer_compatibility(
    value: HistoricalBaseTokenizerCompatibilityReport, path: Path
) -> None:
    validate_historical_base_tokenizer_compatibility(value)
    publish_bytes_absent(path, canonical_json_bytes(value) + b"\n")


def decode_historical_base_tokenizer_compatibility(
    payload: dict[str, object],
) -> HistoricalBaseTokenizerCompatibilityReport:
    expected = set(HistoricalBaseTokenizerCompatibilityReport.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError("Invalid historical Base compatibility fields")
    settings = _settings(payload)
    checked_fields = _strings(payload, "checked_fields")
    value = HistoricalBaseTokenizerCompatibilityReport(
        probe=_text(payload, "probe"),
        condition=_text(payload, "condition"),
        objective=_text(payload, "objective"),
        selection_rule=_optional_text(payload, "selection_rule"),
        rendering=_text(payload, "rendering"),
        study_role=_text(payload, "study_role"),
        accepted_compiled_bundle_sha256=_text(
            payload, "accepted_compiled_bundle_sha256"
        ),
        historical_compiled_bundle_sha256=_text(
            payload, "historical_compiled_bundle_sha256"
        ),
        rendering_settings=settings,
        record_count=_integer(payload, "record_count"),
        ordered_query_ids_sha256=_text(payload, "ordered_query_ids_sha256"),
        accepted_base_tokenizer_id=_text(payload, "accepted_base_tokenizer_id"),
        accepted_base_tokenizer_revision=_text(
            payload, "accepted_base_tokenizer_revision"
        ),
        historical_base_tokenizer_id=_text(payload, "historical_base_tokenizer_id"),
        historical_base_tokenizer_revision=_text(
            payload, "historical_base_tokenizer_revision"
        ),
        accepted_base_rendered_bundle_sha256=_text(
            payload, "accepted_base_rendered_bundle_sha256"
        ),
        historical_base_rendered_bundle_sha256=_text(
            payload, "historical_base_rendered_bundle_sha256"
        ),
        shared_rendered_bundle_sha256=_text(payload, "shared_rendered_bundle_sha256"),
        equivalent_payload_sha256=_text(payload, "equivalent_payload_sha256"),
        checked_fields=checked_fields,
        comparison_kind=_text(payload, "comparison_kind"),
        compatibility_sha256=_text(payload, "compatibility_sha256"),
        schema_version=_text(payload, "schema_version"),
        kind=_text(payload, "kind"),
    )
    validate_historical_base_tokenizer_compatibility(value)
    return value


def load_historical_base_tokenizer_compatibility(
    path: Path,
) -> HistoricalBaseTokenizerCompatibilityReport:
    try:
        with open_stable_regular_file(path) as stream:
            raw = stream.read()
        value = decode_historical_base_tokenizer_compatibility(
            parse_json_object(raw.decode("utf-8"), path, 1)
        )
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Historical Base compatibility is not UTF-8: {path}"
        ) from error
    return value


def verify_historical_base_tokenizer_compatibility(
    report: HistoricalBaseTokenizerCompatibilityReport,
    accepted_compiled: CompiledQueryBundle,
    accepted_base: RenderedQueryBundle,
    historical_compiled: CompiledQueryBundle,
    historical_base: RenderedQueryBundle,
) -> None:
    """Bind a signed report to both independently rendered Base bundles."""
    validate_historical_base_tokenizer_compatibility(report)
    if report != build_historical_base_tokenizer_compatibility(
        accepted_compiled, accepted_base, historical_compiled, historical_base
    ):
        raise ValueError("Historical Base compatibility report does not bind bundles")


def validate_historical_base_tokenizer_compatibility(
    value: HistoricalBaseTokenizerCompatibilityReport,
) -> None:
    if (
        type(value) is not HistoricalBaseTokenizerCompatibilityReport
        or value.schema_version != "1"
        or value.kind != "historical_base_tokenizer_compatibility"
        or value.comparison_kind != "historical_base_identity"
        or value.checked_fields != HISTORICAL_BASE_CHECKED_RECORD_FIELDS
        or value.shared_rendered_bundle_sha256
        != value.accepted_base_rendered_bundle_sha256
        or value.compatibility_sha256 != historical_base_compatibility_sha256(value)
    ):
        raise ValueError("Invalid historical Base compatibility report")
    if type(value.record_count) is not int or value.record_count < 1:
        raise ValueError("Historical Base compatibility record count is invalid")
    contract = get_probe_contract(value.probe)
    if (
        value.condition not in QUERY_CONDITIONS
        or value.objective not in QUERY_OBJECTIVES
        or value.rendering not in QUERY_RENDERINGS
        or value.rendering != "canonical_plain"
        or value.study_role not in STUDY_ROLES
        or value.record_count != contract.expected_count
    ):
        raise ValueError("Invalid historical Base compatibility axes")
    if value.condition != "model_completion" or value.objective != "completion_target":
        raise ValueError("Historical Base compatibility requires model completion")
    if value.selection_rule != contract.primary_rule or value.study_role != "primary":
        raise ValueError(
            "Historical Base compatibility differs from the primary contract"
        )
    if any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or not item[0]
        for item in value.rendering_settings
    ):
        raise ValueError("Invalid historical Base compatibility rendering settings")
    settings = dict(value.rendering_settings)
    if tuple(sorted(settings.items())) != value.rendering_settings:
        raise ValueError("Invalid historical Base compatibility rendering settings")
    validate_rendering_settings(settings)
    for item in (
        value.probe,
        value.condition,
        value.objective,
        value.rendering,
        value.study_role,
        value.accepted_compiled_bundle_sha256,
        value.historical_compiled_bundle_sha256,
        value.accepted_base_tokenizer_id,
        value.accepted_base_tokenizer_revision,
        value.historical_base_tokenizer_id,
        value.historical_base_tokenizer_revision,
        value.accepted_base_rendered_bundle_sha256,
        value.historical_base_rendered_bundle_sha256,
        value.shared_rendered_bundle_sha256,
        value.equivalent_payload_sha256,
        value.compatibility_sha256,
    ):
        if type(item) is not str or not item:
            raise ValueError("Invalid historical Base compatibility value")
    for item in (
        value.accepted_compiled_bundle_sha256,
        value.historical_compiled_bundle_sha256,
        value.ordered_query_ids_sha256,
        value.accepted_base_rendered_bundle_sha256,
        value.historical_base_rendered_bundle_sha256,
        value.shared_rendered_bundle_sha256,
        value.equivalent_payload_sha256,
        value.compatibility_sha256,
    ):
        if len(item) != 64 or any(char not in "0123456789abcdef" for char in item):
            raise ValueError("Invalid historical Base compatibility SHA-256")


def _text(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if type(value) is not str:
        raise ValueError(f"Invalid historical Base compatibility field: {key}")
    return value


def _optional_text(payload: dict[str, object], key: str) -> str | None:
    value = payload[key]
    if value is not None and type(value) is not str:
        raise ValueError(f"Invalid historical Base compatibility field: {key}")
    return value


def _integer(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if type(value) is not int:
        raise ValueError(f"Invalid historical Base compatibility field: {key}")
    return value


def _strings(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload[key]
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"Invalid historical Base compatibility field: {key}")
    return tuple(value)


def _settings(payload: dict[str, object]) -> tuple[tuple[str, object], ...]:
    value = payload["rendering_settings"]
    if type(value) is not list:
        raise ValueError(
            "Invalid historical Base compatibility field: rendering_settings"
        )
    result: list[tuple[str, object]] = []
    for item in value:
        if (
            type(item) is not list
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) not in (str, int, bool, type(None))
        ):
            raise ValueError(
                "Invalid historical Base compatibility field: rendering_settings"
            )
        result.append((item[0], item[1]))
    return tuple(result)


def _validate_historical_compilation_normalization(
    accepted: CompiledQueryBundle, historical: CompiledQueryBundle
) -> None:
    if len(accepted.records) != len(historical.records):
        raise ValueError("Historical Base compilation has a different record count")
    normalized_records = tuple(
        replace(
            historical_record,
            base_completion_model_id=accepted_record.base_completion_model_id,
            base_completion_model_revision=accepted_record.base_completion_model_revision,
        )
        for accepted_record, historical_record in zip(
            accepted.records, historical.records, strict=True
        )
    )
    normalized = replace(historical, records=normalized_records)
    if normalized != accepted:
        raise ValueError(
            "Historical Base compilation differs beyond the attested model identity"
        )
    for accepted_record, normalized_record in zip(
        accepted.records, normalized.records, strict=True
    ):
        if sha256_json(normalized_record) != sha256_json(accepted_record):
            raise ValueError("Historical Base canonical record normalization differs")
