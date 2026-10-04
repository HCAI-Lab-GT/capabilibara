"""Render canonical query bundles with explicit causal labels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from data_attribution.query_conditions.io import (
    compiled_bundle_sha256,
    validate_compiled_bundle,
    validate_rendered_binding,
)
from data_attribution.artifact_integrity import sha256_json
from data_attribution.query_conditions.types import (
    CanonicalQueryRecord,
    CompiledQueryBundle,
    OffsetOverlapEvidence,
    RenderedQueryBundle,
    RenderedQueryRecord,
)

DEFAULT_MAX_LENGTH = 4096
type RenderSettings = tuple[str, str, int]


@dataclass(frozen=True)
class TokenizedText:
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    special_mask: tuple[int, ...]
    overlap_evidence: tuple[OffsetOverlapEvidence, ...] = ()


def render_query_bundle(
    bundle: CompiledQueryBundle,
    tokenizer: object,
    *,
    tokenizer_id: str,
    tokenizer_revision: str,
    max_length: int | None = None,
) -> RenderedQueryBundle:
    validate_compiled_bundle(bundle)
    limit = _validated_limit(max_length)
    _validate_tokenizer_identity(tokenizer_id, tokenizer_revision)
    records = _render_records(
        bundle, tokenizer, tokenizer_id, tokenizer_revision, limit
    )
    rendered = RenderedQueryBundle(
        probe=bundle.probe,
        condition=bundle.condition,
        objective=bundle.objective,
        selection_rule=bundle.selection_rule,
        rendering=bundle.rendering,
        study_role=bundle.study_role,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        compiled_bundle_sha256=compiled_bundle_sha256(bundle),
        rendering_settings=_rendering_settings(limit),
        records=records,
    )
    validate_rendered_binding(rendered, bundle)
    return rendered


def _render_records(
    bundle: CompiledQueryBundle,
    tokenizer: object,
    tokenizer_id: str,
    tokenizer_revision: str,
    limit: int,
) -> tuple[RenderedQueryRecord, ...]:
    return tuple(
        _render_record(
            record, bundle, tokenizer, (tokenizer_id, tokenizer_revision, limit)
        )
        for record in bundle.records
    )


def _render_record(
    record: CanonicalQueryRecord,
    bundle: CompiledQueryBundle,
    tokenizer: object,
    settings: RenderSettings,
) -> RenderedQueryRecord:
    text = record.prompt + (record.target or "")
    tokens = tokenize_exact(tokenizer, text, settings[2])
    labels, start, end = build_labels(tokens, bundle.objective, len(record.prompt))
    return RenderedQueryRecord(
        query_id=record.query_id,
        probe=record.probe,
        condition=record.condition,
        objective=bundle.objective,
        selection_rule=record.selection_rule,
        rendering=bundle.rendering,
        study_role=bundle.study_role,
        canonical_record_sha256=sha256_json(record),
        input_ids=tokens.input_ids,
        labels=labels,
        offset_mapping=tokens.offsets,
        offset_overlap_evidence=tokens.overlap_evidence,
        special_tokens_mask=tokens.special_mask,
        length=len(tokens.input_ids),
        loss_token_start=start,
        loss_token_end=end,
        prompt_character_length=len(record.prompt),
        text_character_length=len(text),
        tokenizer_id=settings[0],
        tokenizer_revision=settings[1],
    )


def _validated_limit(max_length: int | None) -> int:
    value = DEFAULT_MAX_LENGTH if max_length is None else max_length
    if type(value) is not int or value < 1:
        raise ValueError("Maximum length must be a positive integer")
    if value > DEFAULT_MAX_LENGTH:
        raise ValueError("Maximum length must be at most 4096")
    return value


def _validate_tokenizer_identity(tokenizer_id: str, revision: str) -> None:
    if any(
        type(value) is not str or not value.strip()
        for value in (tokenizer_id, revision)
    ):
        raise ValueError("tokenizer ID and revision must be nonempty")


def _rendering_settings(limit: int) -> dict[str, str | int | bool | None]:
    return {
        "add_special_tokens": True,
        "padding": False,
        "truncation": False,
        "return_offsets_mapping": True,
        "return_special_tokens_mask": True,
        "return_attention_mask": False,
        "clean_up_tokenization_spaces": False,
        "max_length": limit,
    }


def tokenize_exact(tokenizer: object, text: str, max_length: int) -> TokenizedText:
    caller = cast(Callable[..., object], tokenizer)
    output = caller(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_attention_mask=False,
    )
    tokens = _decode_output(output)
    if len(tokens.input_ids) > max_length:
        raise ValueError("Rendered query exceeds maximum length")
    overlap_evidence = _validate_offsets(tokenizer, tokens, text)
    _validate_round_trip(tokenizer, tokens.input_ids, text)
    return TokenizedText(
        tokens.input_ids,
        tokens.offsets,
        tokens.special_mask,
        overlap_evidence,
    )


def build_labels(
    tokens: TokenizedText, objective: str, prompt_length: int
) -> tuple[tuple[int, ...], int, int]:
    if objective in ("prompt_next_token", "joint_sequence"):
        labels = (-100, *tokens.input_ids[1:])
        return _validated_labels(tokens, labels)
    eligible = _completion_indices(tokens, prompt_length)
    labels = [-100] * len(tokens.input_ids)
    for index in eligible:
        if index > 0:
            labels[index] = tokens.input_ids[index]
    return _validated_labels(tokens, tuple(labels))


def _decode_output(output: object) -> TokenizedText:
    if not isinstance(output, Mapping):
        raise ValueError("Tokenizer output must be a mapping")
    required = ("input_ids", "offset_mapping", "special_tokens_mask")
    if missing := [key for key in required if key not in output]:
        raise ValueError(f"Tokenizer output is missing fields: {missing}")
    input_ids = _integer_list(output["input_ids"], "input IDs")
    mask = _integer_list(output["special_tokens_mask"], "special-token mask")
    if any(value not in (0, 1) for value in mask):
        raise ValueError("Tokenizer special-token mask must contain zero or one")
    offsets = _offset_list(output["offset_mapping"])
    if not input_ids or len(input_ids) != len(mask) or len(input_ids) != len(offsets):
        raise ValueError("Tokenizer outputs must have nonzero equal lengths")
    return TokenizedText(tuple(input_ids), tuple(offsets), tuple(mask))


def _integer_list(value: object, label: str) -> list[int]:
    if type(value) is not list or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise ValueError(f"Tokenizer {label} must be a list of nonnegative integers")
    return cast(list[int], value)


def _offset_list(value: object) -> list[tuple[int, int]]:
    if type(value) is not list:
        raise ValueError("Tokenizer offsets must be a list")
    result: list[tuple[int, int]] = []
    for pair in value:
        if type(pair) not in (tuple, list) or len(pair) != 2:
            raise ValueError("Tokenizer offset must be an integer pair")
        start, end = pair
        if type(start) is not int or type(end) is not int:
            raise ValueError("Tokenizer offset must be an integer pair")
        if start < 0 or end < 0:
            raise ValueError("Tokenizer offsets must be nonnegative")
        result.append((start, end))
    return result


def _validate_offsets(
    tokenizer: object, tokens: TokenizedText, text: str
) -> tuple[OffsetOverlapEvidence, ...]:
    nonspecial = [index for index, value in enumerate(tokens.special_mask) if not value]
    if not nonspecial:
        raise ValueError("Tokenizer output has no text tokens")
    first, last = nonspecial[0], nonspecial[-1]
    if any(tokens.special_mask[index] for index in range(first, last + 1)):
        raise ValueError("Tokenizer output has an unsupported internal special token")
    previous = 0
    group: list[int] = []
    overlap_evidence: list[OffsetOverlapEvidence] = []
    for index in nonspecial:
        start, end = tokens.offsets[index]
        if _is_suffix_overlap(start, end, previous, group, len(text)):
            group.append(index)
            continue
        if proof := _byte_fallback_evidence(tokenizer, tokens, text, group):
            overlap_evidence.append(proof)
        group = [index]
        previous = end
    if proof := _byte_fallback_evidence(tokenizer, tokens, text, group):
        overlap_evidence.append(proof)
    if previous != len(text):
        raise ValueError("Tokenizer offsets contain a gap")
    for index, special in enumerate(tokens.special_mask):
        if special and tokens.offsets[index] != (0, 0):
            raise ValueError("Tokenizer special token has ambiguous nonzero offset")
    return tuple(overlap_evidence)


def _is_suffix_overlap(
    start: int,
    end: int,
    previous: int,
    group: list[int],
    text_length: int,
) -> bool:
    if (start, end) == (0, 0):
        raise ValueError("Tokenizer non-special token has zero offset")
    if start > previous:
        raise ValueError("Tokenizer offsets contain a gap")
    if end <= start or end > text_length:
        raise ValueError("Tokenizer offset is outside canonical text")
    if start >= previous:
        return False
    if not group or end != previous or start != end - 1:
        raise ValueError("Tokenizer offsets overlap ambiguously")
    return True


def _byte_fallback_evidence(
    tokenizer: object,
    tokens: TokenizedText,
    text: str,
    group: list[int],
) -> OffsetOverlapEvidence | None:
    if len(group) < 2:
        return None
    start = tokens.offsets[group[0]][0]
    end = tokens.offsets[group[0]][1]
    repeated = text[end - 1 : end]
    if len(repeated.encode("utf-8")) == 1:
        raise ValueError("Tokenizer offset overlap lacks byte-fallback evidence")
    ids = [tokens.input_ids[index] for index in group]
    grouped_decode = _decode(tokenizer, ids)
    if grouped_decode != text[start:end]:
        raise ValueError("Tokenizer offset overlap lacks byte-fallback evidence")
    individual_decodes = tuple(_decode(tokenizer, [token_id]) for token_id in ids)
    if any("�" not in decoded for decoded in individual_decodes):
        raise ValueError("Tokenizer offset overlap lacks byte-fallback evidence")
    return OffsetOverlapEvidence(
        token_start=group[0],
        token_end=group[-1] + 1,
        token_ids=tuple(ids),
        source_start=start,
        source_end=end,
        grouped_decode=grouped_decode,
        individual_decodes=individual_decodes,
    )


def _validate_round_trip(tokenizer: object, ids: tuple[int, ...], text: str) -> None:
    if _decode(tokenizer, list(ids)) != text:
        raise ValueError("Tokenizer did not exactly round trip canonical text")


def _decode(tokenizer: object, ids: list[int]) -> str:
    decoder = getattr(tokenizer, "decode", None)
    if not callable(decoder):
        raise ValueError("Tokenizer must provide decode for exact round trip")
    decoded = decoder(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if type(decoded) is not str:
        raise ValueError("Tokenizer decode must return text")
    return decoded


def _completion_indices(tokens: TokenizedText, boundary: int) -> list[int]:
    indices: list[int] = []
    for index, ((start, end), special) in enumerate(
        zip(tokens.offsets, tokens.special_mask, strict=True)
    ):
        if special:
            continue
        if start < boundary < end:
            raise ValueError("Tokenizer token crosses prompt and target boundary")
        if start >= boundary:
            indices.append(index)
    return indices


def _validated_labels(
    tokens: TokenizedText, labels: tuple[int, ...]
) -> tuple[tuple[int, ...], int, int]:
    if len(labels) != len(tokens.input_ids):
        raise ValueError("Rendered input IDs and labels have unequal lengths")
    effective = [index for index in range(1, len(labels)) if labels[index] != -100]
    if not effective:
        raise ValueError("Rendered query has zero effective labeled tokens")
    start, end = effective[0], effective[-1] + 1
    if effective != list(range(start, end)):
        raise ValueError("Rendered loss-token span is not contiguous")
    return labels, start, end
