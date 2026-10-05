from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import data_attribution.query_conditions.types as query_types

CompileQueryRequest = query_types.CompileQueryRequest
type ReferenceMap = dict[
    str, tuple[query_types.ReferenceSelection, query_types.ReferenceInput]
]


def _freeze_task_counts(
    values: dict[str, dict[str, int]],
) -> Mapping[str, Mapping[str, int]]:
    return MappingProxyType(
        {probe: MappingProxyType(tasks) for probe, tasks in values.items()}
    )


PROBE_TASK_COUNTS = _freeze_task_counts(
    {
        "bbh_causal_judgement": {"bbh_causal_judgement": 187},
        "bbh_disambiguation_qa": {"bbh_disambiguation_qa": 250},
        "bbq": {"bbq": 58_492},
        "ethics": {
            "ethics_commonsense": 3_885,
            "ethics_deontology": 3_596,
            "ethics_justice": 2_704,
            "ethics_utilitarianism": 4_807,
            "ethics_virtue": 4_975,
        },
        "mmlu_moral": {
            "mmlu_moral_scenarios": 895,
            "mmlu_moral_disputes": 346,
            "mmlu_philosophy": 311,
            "mmlu_business_ethics": 100,
        },
        "negotiationtom": {
            "negotiationtom_desire": 14_280,
            "negotiationtom_belief": 14_280,
        },
        "pub": {
            "pub_1": 2_500,
            "pub_2": 2_506,
            "pub_3": 2_506,
            "pub_4": 2_000,
            "pub_5": 2_000,
            "pub_6": 2_000,
            "pub_7": 1_770,
            "pub_8": 1_770,
            "pub_9": 1_770,
            "pub_10": 2_100,
            "pub_11": 1_800,
            "pub_12": 2_563,
            "pub_13": 1_000,
            "pub_14": 457,
        },
        "simpletom": {
            "simpletom_mental-state-qa": 1_147,
            "simpletom_behavior-qa": 1_147,
            "simpletom_judgment-qa": 1_147,
        },
        "tombench": {"tombench": 2_860},
        "morables": {"morables": 709},
        "moralexceptqa_rbqa": {"moralexceptqa_rbqa": 148},
    }
)
PROBE_COUNTS: Mapping[str, int] = MappingProxyType(
    {probe: sum(tasks.values()) for probe, tasks in PROBE_TASK_COUNTS.items()}
)
TOTAL_QUERY_COUNT = 143_008
REFERENCE_POLICY_NAMES = frozenset(
    {
        "parsed_answer",
        "ethics_declared_ordering",
        "human_consensus",
        "explicit_item_index",
    }
)


def _validate_completion_tie_policy(
    policy: object,
) -> query_types.CompletionTiePolicy:
    if type(policy) is not str or policy not in query_types.COMPLETION_TIE_POLICIES:
        raise ValueError(f"Unknown completion tie policy: {policy!r}")
    return cast(query_types.CompletionTiePolicy, policy)


@dataclass(frozen=True)
class ReferenceRule:
    policy: str
    source_keys: tuple[str, ...] = ()
    allow_exclusion: bool = False

    def __post_init__(self) -> None:
        if type(self.source_keys) is not tuple:
            raise ValueError("Reference rule has invalid source keys type")
        if type(self.policy) is not str or self.policy not in REFERENCE_POLICY_NAMES:
            raise ValueError(f"Unknown reference policy: {self.policy!r}")
        if any(type(key) is not str or not key for key in self.source_keys):
            raise ValueError("Reference rule has an invalid source key")
        if len(set(self.source_keys)) != len(self.source_keys):
            raise ValueError("Reference rule has duplicate source keys")
        if type(self.allow_exclusion) is not bool:
            raise ValueError("Reference rule has invalid exclusion policy")


@dataclass(frozen=True)
class ProbeContract:
    probe: str
    expected_count: int
    task_counts: Mapping[str, int]
    primary_rule: str
    allowed_sensitivities: tuple[str, ...]
    reference_rules: Mapping[str, ReferenceRule]
    completion_tie_policy: query_types.CompletionTiePolicy = "reject"

    def __post_init__(self) -> None:
        if type(self.allowed_sensitivities) is not tuple:
            raise ValueError("Probe contract has invalid sensitivities type")
        object.__setattr__(
            self, "task_counts", MappingProxyType(dict(self.task_counts))
        )
        object.__setattr__(
            self, "reference_rules", MappingProxyType(dict(self.reference_rules))
        )
        validate_probe_contract(self)

    def reference_rule(self, task_core: str) -> ReferenceRule:
        rule = self.reference_rules.get(task_core)
        if rule is None:
            raise ValueError(
                f"Probe {self.probe!r} has unknown task core {task_core!r}"
            )
        return rule


def validate_probe_contract(contract: ProbeContract) -> None:
    if type(contract.probe) is not str or not contract.probe:
        raise ValueError("Probe contract requires a probe")
    if type(contract.expected_count) is not int or contract.expected_count < 1:
        raise ValueError("Probe contract has invalid expected_count")
    if not contract.task_counts or any(
        type(core) is not str or not core or type(count) is not int or count < 1
        for core, count in contract.task_counts.items()
    ):
        raise ValueError("Probe contract has invalid task counts")
    if sum(contract.task_counts.values()) != contract.expected_count:
        raise ValueError("Probe contract expected_count does not match task counts")
    _validate_rules(contract)


def _validate_rules(contract: ProbeContract) -> None:
    rules = (contract.primary_rule, *contract.allowed_sensitivities)
    if any(type(rule) is not str or not rule for rule in rules):
        raise ValueError("Probe contract has a missing rule")
    if len(set(rules)) != len(rules):
        raise ValueError("Probe contract has duplicate selection rules")
    if set(contract.reference_rules) != set(contract.task_counts):
        raise ValueError("Probe contract reference policies do not cover task cores")
    if any(
        not isinstance(rule, ReferenceRule)
        for rule in contract.reference_rules.values()
    ):
        raise ValueError("Probe contract has an invalid reference rule")
    _validate_completion_tie_policy(contract.completion_tie_policy)


def _reference_rules(
    tasks: Mapping[str, int], policy: str, *source_keys: str
) -> dict[str, ReferenceRule]:
    rule = ReferenceRule(policy, source_keys)
    return dict.fromkeys(tasks, rule)


def _contract(
    probe: str,
    policy: str,
    source_keys: tuple[str, ...] = (),
    *,
    sensitivities: tuple[str, ...] = (),
    overrides: Mapping[str, ReferenceRule] | None = None,
    completion_tie_policy: query_types.CompletionTiePolicy = "reject",
) -> ProbeContract:
    tasks = PROBE_TASK_COUNTS[probe]
    references = _reference_rules(tasks, policy, *source_keys)
    references.update(overrides or {})
    return ProbeContract(
        probe,
        PROBE_COUNTS[probe],
        tasks,
        "per_character",
        sensitivities,
        references,
        completion_tie_policy,
    )


PROBE_CONTRACTS: Mapping[str, ProbeContract] = MappingProxyType(
    {
        "bbh_causal_judgement": _contract(
            "bbh_causal_judgement",
            "parsed_answer",
            completion_tie_policy="lowest_candidate_index",
        ),
        "bbh_disambiguation_qa": _contract(
            "bbh_disambiguation_qa",
            "parsed_answer",
            completion_tie_policy="lowest_candidate_index",
        ),
        "bbq": _contract(
            "bbq",
            "explicit_item_index",
            ("label",),
            completion_tie_policy="lowest_candidate_index",
        ),
        "ethics": _contract(
            "ethics",
            "explicit_item_index",
            ("label",),
            overrides={
                "ethics_utilitarianism": ReferenceRule("ethics_declared_ordering")
            },
            completion_tie_policy="lowest_candidate_index",
        ),
        "mmlu_moral": _contract(
            "mmlu_moral",
            "explicit_item_index",
            ("answer",),
            completion_tie_policy="lowest_candidate_index",
        ),
        "negotiationtom": _contract(
            "negotiationtom",
            "explicit_item_index",
            ("answer_index", "answer"),
            completion_tie_policy="lowest_candidate_index",
        ),
        "pub": _contract(
            "pub",
            "explicit_item_index",
            ("correct answer",),
            completion_tie_policy="lowest_candidate_index",
        ),
        "simpletom": _contract(
            "simpletom",
            "explicit_item_index",
            ("answerKey",),
            sensitivities=("unconditioned",),
            completion_tie_policy="reject",
        ),
        "tombench": _contract(
            "tombench",
            "parsed_answer",
            completion_tie_policy="lowest_candidate_index",
        ),
        "morables": _contract(
            "morables",
            "explicit_item_index",
            ("correct_moral_label",),
            completion_tie_policy="lowest_candidate_index",
        ),
        "moralexceptqa_rbqa": _contract(
            "moralexceptqa_rbqa",
            "human_consensus",
            completion_tie_policy="lowest_candidate_index",
        ),
    }
)


def get_probe_contract(probe: str) -> ProbeContract:
    return resolve_probe_contract(probe)


def resolve_probe_contract(
    probe: str, override: ProbeContract | None = None
) -> ProbeContract:
    if type(probe) is not str or not probe:
        raise ValueError(f"Unknown probe: {probe!r}")
    registered = PROBE_CONTRACTS.get(probe)
    if override is None:
        if registered is None:
            raise ValueError(f"Unknown probe: {probe!r}")
        return registered
    if type(override) is not ProbeContract or override.probe != probe:
        raise ValueError("Contract override does not match the evidence probe")
    if registered is not None and override != registered:
        raise ValueError("Contract override does not match registered probe contract")
    return registered if registered is not None else override


if (
    sum(contract.expected_count for contract in PROBE_CONTRACTS.values())
    != TOTAL_QUERY_COUNT
):
    raise ValueError("Probe contracts do not total 143,008 rows")


_ANSWER_IS_PHRASE = re.compile(r"(?i)\banswer\s+is\b")
_CANONICAL_WRAPPED_ANSWER = re.compile(r"(?i)(?:the\s+)?answer\s+is\s+(.+?)(?:\.)?")
_SOURCE_LETTER = re.compile(r"\(?\s*([A-Z])\s*\)?[\s\.\):：、-]*")
_OPTION_LINE = re.compile(r"^ ([A-Z])\. (.+)$", re.MULTILINE)


def parse_source_answer(answer: str, choices: tuple[str, ...]) -> int:
    if type(answer) is not str or not answer.strip():
        raise ValueError("Parsed answer is empty")
    stripped = answer.strip()
    if (index := _positional_index(stripped, len(choices))) is not None:
        return index
    if (index := _choice_index(stripped, choices)) is not None:
        return index
    unwrapped = _unwrap_answer(stripped)
    if (index := _positional_index(unwrapped, len(choices))) is not None:
        return index
    if (index := _choice_index(unwrapped, choices)) is not None:
        return index
    raise ValueError("Parsed answer does not match one source choice unambiguously")


def _choice_index(answer: str, choices: tuple[str, ...]) -> int | None:
    normalized = _normalize(answer)
    matches = [
        index
        for index, choice in enumerate(choices)
        if _normalize(choice) == normalized
    ]
    if len(matches) > 1:
        raise ValueError("Parsed answer does not match one source choice unambiguously")
    return matches[0] if matches else None


def _positional_index(answer: str, candidate_count: int) -> int | None:
    match = _SOURCE_LETTER.fullmatch(answer.upper())
    return _letter_index(match.group(1), candidate_count) if match else None


def require_prompt_choices(prompt: str, values: tuple[str, ...], label: str) -> None:
    option_block = _option_block(prompt, label)
    matches = tuple(_OPTION_LINE.finditer(option_block))
    expected_labels = tuple(chr(ord("A") + index) for index in range(len(values)))
    found_labels = tuple(match.group(1) for match in matches)
    found_values = tuple(match.group(2) for match in matches)
    if found_labels != expected_labels or found_values != values:
        raise ValueError(
            f"{label} source order does not match the labeled option block"
        )
    if any(
        option_block[first.end() : second.start()] != "\n"
        for first, second in zip(matches, matches[1:], strict=False)
    ):
        raise ValueError(f"{label} has an ambiguous labeled option block")


def _unwrap_answer(answer: str) -> str:
    stripped = answer.strip()
    phrases = tuple(_ANSWER_IS_PHRASE.finditer(stripped))
    if len(phrases) > 1:
        raise ValueError("Parsed answer contains multiple wrapped answers")
    if not phrases:
        return stripped
    wrapper = _CANONICAL_WRAPPED_ANSWER.fullmatch(stripped)
    if wrapper is None:
        raise ValueError("Parsed answer is not in canonical wrapped form")
    return wrapper.group(1).strip()


def _option_block(prompt: str, label: str) -> str:
    if type(prompt) is not str or not prompt.endswith("\nAnswer:"):
        raise ValueError(f"{label} prompt has an unsupported labeled option block")
    block = prompt[: -len("\nAnswer:")]
    matches = tuple(_OPTION_LINE.finditer(block))
    if not matches or matches[-1].end() != len(block):
        raise ValueError(f"{label} prompt has an unsupported labeled option block")
    return block


def _letter_index(letter: str, candidate_count: int) -> int:
    index = ord(letter) - ord("A")
    if index not in range(candidate_count):
        raise ValueError("Parsed source letter is outside candidate range")
    return index


def _normalize(value: str) -> str:
    return " ".join(value.strip().rstrip(".").split()).casefold()


type ReferenceResolver = Callable[
    [query_types.QueryEvidence, query_types.ReferenceInputBase, ReferenceRule], int
]
type ReferencePolicy = tuple[type[query_types.ReferenceInputBase], ReferenceResolver]


def resolve_reference(
    evidence: query_types.QueryEvidence,
    source: query_types.ReferenceInput,
    *,
    contract: ProbeContract | None = None,
) -> query_types.ReferenceSelection:
    active = _contract_for(evidence, contract)
    rule = active.reference_rule(evidence.task_core)
    policy = REFERENCE_POLICIES[rule.policy]
    candidates = validate_candidates(evidence.candidates)
    _validate_input_identity(evidence, source)
    if type(source) is not policy[0]:
        raise ValueError(f"Reference policy {rule.policy!r} has wrong input type")
    index = policy[1](evidence, source, rule)
    if type(evidence.reference_index) is not int or index != evidence.reference_index:
        raise ValueError("Reference source facts disagree with Task 2 reference index")
    return query_types.ReferenceSelection(
        evidence.query_id,
        index,
        candidates[index].continuation,
        rule.policy,
        source.source,
    )


def _parsed_index(
    evidence: query_types.QueryEvidence,
    source: query_types.ReferenceInputBase,
    _rule: ReferenceRule,
) -> int:
    parsed = cast(query_types.ParsedAnswerReferenceInput, source)
    choices = parsed.ordered_source_choices
    if len(choices) != len(evidence.candidates):
        raise ValueError("Parsed answer choices do not match candidate count")
    require_prompt_choices(evidence.prompt, choices, "Parsed answer choices")
    primary = parse_source_answer(parsed.source_answer, choices)
    if parsed.corroborating_answer is not None:
        corroborating = parse_source_answer(parsed.corroborating_answer, choices)
        if corroborating != primary:
            raise ValueError("Parsed source answers disagree")
    return primary


def _ethics_index(
    evidence: query_types.QueryEvidence,
    source: query_types.ReferenceInputBase,
    _rule: ReferenceRule,
) -> int:
    ethics = cast(query_types.EthicsOrderingReferenceInput, source)
    if ethics.roles != ("baseline", "less_pleasant"):
        raise ValueError("ETHICS utilitarianism requires declared source roles")
    if len(ethics.ordered_source_values) != 2 or len(evidence.candidates) != 2:
        raise ValueError(
            "ETHICS utilitarianism requires two source values and candidates"
        )
    require_prompt_choices(
        evidence.prompt, ethics.ordered_source_values, "ETHICS values"
    )
    return 0


def _consensus_index(
    evidence: query_types.QueryEvidence,
    source: query_types.ReferenceInputBase,
    _rule: ReferenceRule,
) -> int:
    consensus = cast(query_types.HumanConsensusReferenceInput, source)
    if consensus.orientation != ("No", "Yes") or len(evidence.candidates) != 2:
        raise ValueError("MoralExcept human-consensus candidate orientation is invalid")
    require_prompt_choices(
        evidence.prompt, consensus.orientation, "MoralExcept orientation"
    )
    if type(consensus.threshold) not in (int, float) or consensus.threshold != 0.5:
        raise ValueError("MoralExcept human-consensus threshold must be 0.5")
    if consensus.comparator != ">":
        raise ValueError("MoralExcept human-consensus comparator must be >")
    response = consensus.human_response
    if (
        type(response) not in (int, float)
        or not math.isfinite(response)
        or not 0 <= response <= 1
    ):
        raise ValueError("MoralExcept human response must be finite and in [0, 1]")
    return int(response > 0.5)


def _explicit_index(
    evidence: query_types.QueryEvidence,
    source: query_types.ReferenceInputBase,
    rule: ReferenceRule,
) -> int:
    explicit = cast(query_types.ExplicitIndexReferenceInput, source)
    if (
        type(explicit.source_key) is not str
        or explicit.source_key not in rule.source_keys
    ):
        raise ValueError("Explicit reference source key is not declared")
    if type(explicit.source_index) is not int or explicit.source_index not in range(
        len(evidence.candidates)
    ):
        raise ValueError("Explicit reference source index is invalid")
    return explicit.source_index


def _contract_for(
    evidence: query_types.QueryEvidence, contract: ProbeContract | None
) -> ProbeContract:
    return resolve_probe_contract(evidence.probe, contract)


def _validate_input_identity(
    evidence: query_types.QueryEvidence, source: query_types.ReferenceInputBase
) -> None:
    if source.query_id != evidence.query_id:
        raise ValueError("Reference input query ID does not match evidence")


REFERENCE_POLICIES: Mapping[str, ReferencePolicy] = MappingProxyType(
    {
        "parsed_answer": (query_types.ParsedAnswerReferenceInput, _parsed_index),
        "ethics_declared_ordering": (
            query_types.EthicsOrderingReferenceInput,
            _ethics_index,
        ),
        "human_consensus": (
            query_types.HumanConsensusReferenceInput,
            _consensus_index,
        ),
        "explicit_item_index": (
            query_types.ExplicitIndexReferenceInput,
            _explicit_index,
        ),
    }
)
if set(REFERENCE_POLICIES) != set(REFERENCE_POLICY_NAMES):
    raise ValueError("Reference resolver registry does not match contract policies")


def resolve_reference_inputs(
    evidence: Iterable[query_types.QueryEvidence],
    inputs: Iterable[query_types.ReferenceInput],
    *,
    contract: ProbeContract | None = None,
) -> query_types.ReferenceResolution:
    rows, sources = tuple(evidence), tuple(inputs)
    if not rows:
        raise ValueError("Reference resolution requires query evidence")
    active = resolve_probe_contract(rows[0].probe, contract)
    _validate_batch_identity(active, rows, sources)
    by_id = {source.query_id: source for source in sources}
    records: list[query_types.ReferenceSelection] = []
    exclusions: list[query_types.ReferenceExclusion] = []
    ordered_rows = sorted(rows, key=lambda item: item.query_id)
    for row in ordered_rows:
        result = _resolve_one(active, row, by_id[row.query_id])
        if isinstance(result, query_types.ReferenceExclusion):
            exclusions.append(result)
        else:
            records.append(result)
    ordered_inputs = tuple(by_id[row.query_id] for row in ordered_rows)
    return query_types.ReferenceResolution(
        ordered_inputs, tuple(records), tuple(exclusions)
    )


def validate_reference_coverage(
    contract: ProbeContract,
    evidence: Iterable[query_types.QueryEvidence],
    resolution: query_types.ReferenceResolution,
) -> None:
    rows = tuple(evidence)
    active = resolve_probe_contract(contract.probe, contract)
    _validate_evidence_roster(active, rows)
    if type(resolution) is not query_types.ReferenceResolution:
        raise ValueError("Reference coverage requires an immutable resolution")
    expected = resolve_reference_inputs(rows, resolution.inputs, contract=active)
    if resolution != expected:
        raise ValueError("Reference coverage differs from resolved reference outputs")
    if len(resolution.records) + len(resolution.exclusions) != active.expected_count:
        raise ValueError("Reference coverage does not match declared contract count")


def _resolve_one(
    contract: ProbeContract,
    evidence: query_types.QueryEvidence,
    source: query_types.ReferenceInput,
) -> query_types.ReferenceSelection | query_types.ReferenceExclusion:
    if not isinstance(source, query_types.UnavailableReferenceInput):
        return resolve_reference(evidence, source, contract=contract)
    rule = contract.reference_rule(evidence.task_core)
    if not rule.allow_exclusion:
        raise ValueError(f"Reference rule {rule.policy!r} does not allow exclusions")
    return query_types.ReferenceExclusion(
        evidence.query_id, contract.probe, source.reason, rule.policy, source.source
    )


def _validate_batch_identity(
    contract: ProbeContract,
    rows: tuple[query_types.QueryEvidence, ...],
    sources: tuple[query_types.ReferenceInput, ...],
) -> None:
    row_ids = _unique_ids([row.query_id for row in rows], "query evidence")
    source_ids = _unique_ids([source.query_id for source in sources], "reference input")
    if any(row.probe != contract.probe for row in rows):
        raise ValueError("Reference evidence probe does not match contract")
    if missing := sorted(row_ids - source_ids):
        raise ValueError(f"Missing reference input for query IDs: {missing}")
    if extra := sorted(source_ids - row_ids):
        raise ValueError(f"Reference inputs have unknown query IDs: {extra}")


def _unique_ids(values: list[str], label: str) -> set[str]:
    if any(type(value) is not str or not value for value in values):
        raise ValueError(f"Invalid {label} query ID")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} query ID")
    return set(values)


def _validate_evidence_roster(
    contract: ProbeContract, rows: tuple[query_types.QueryEvidence, ...]
) -> None:
    query_ids = [row.query_id for row in rows]
    if any(type(query_id) is not str or not query_id for query_id in query_ids):
        raise ValueError("Reference evidence has an invalid query ID")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query evidence query ID")
    if any(row.probe != contract.probe for row in rows):
        raise ValueError("Reference evidence has the wrong probe")
    counts = Counter(row.task_core for row in rows)
    if len(rows) != contract.expected_count or dict(counts) != dict(
        contract.task_counts
    ):
        raise ValueError("Reference evidence roster does not match contract")


@dataclass(frozen=True)
class SelectionStrategy:
    name: str
    score_field: str
    minimize: bool = False


@dataclass(frozen=True)
class CompletionSelection:
    query_id: str
    candidate_index: int
    continuation: str
    selection_rule: str
    tie_policy: query_types.CompletionTiePolicy
    is_correct: bool


SELECTION_STRATEGIES: Mapping[str, SelectionStrategy] = MappingProxyType(
    {
        name: SelectionStrategy(name, name, minimize=name == "per_byte")
        for name in (
            "raw",
            "per_character",
            "per_token",
            "per_byte",
            "unconditioned",
        )
    }
)


def get_selection_strategy(rule: str) -> SelectionStrategy:
    if type(rule) is not str or rule not in SELECTION_STRATEGIES:
        raise ValueError(f"Unknown selection strategy: {rule!r}")
    return SELECTION_STRATEGIES[rule]


def validate_selection_contract(contract: ProbeContract) -> None:
    for rule in (contract.primary_rule, *contract.allowed_sensitivities):
        get_selection_strategy(rule)


def validate_candidates(
    candidates: Iterable[query_types.CandidateEvidence],
) -> tuple[query_types.CandidateEvidence, ...]:
    values = tuple(candidates)
    if not values:
        raise ValueError("Evidence has no candidates")
    indices = [candidate.candidate_index for candidate in values]
    if any(type(index) is not int for index in indices) or indices != list(
        range(len(values))
    ):
        raise ValueError("Evidence candidate indices must be ordered and contiguous")
    for candidate in values:
        if type(candidate.continuation) is not str or not candidate.continuation:
            raise ValueError("Evidence candidate continuation must be nonempty")
        if not isinstance(candidate.scores, Mapping):
            raise ValueError("Evidence candidate scores must be a mapping")
    return values


def select_candidate_index(
    candidates: Iterable[query_types.CandidateEvidence],
    rule: str,
    *,
    tie_policy: query_types.CompletionTiePolicy = "reject",
) -> int:
    _validate_completion_tie_policy(tie_policy)
    values = validate_candidates(candidates)
    strategy = get_selection_strategy(rule)
    scores = [_score(candidate, strategy.score_field) for candidate in values]
    winning = min(scores) if strategy.minimize else max(scores)
    if scores.count(winning) != 1:
        if tie_policy == "reject":
            raise ValueError(f"Exact score tie for {rule}")
        return min(
            candidate.candidate_index
            for candidate, score in zip(values, scores, strict=True)
            if score == winning
        )
    return values[scores.index(winning)].candidate_index


def select_base_completion(
    evidence: query_types.QueryEvidence, selection_rule: str | None = None
) -> CompletionSelection:
    if type(evidence.query_id) is not str or not evidence.query_id:
        raise ValueError("Evidence requires a query ID")
    contract = get_probe_contract(evidence.probe)
    contract.reference_rule(evidence.task_core)
    rule = contract.primary_rule if selection_rule is None else selection_rule
    get_selection_strategy(rule)
    if rule != contract.primary_rule and rule not in contract.allowed_sensitivities:
        raise ValueError(
            f"Selection rule {rule!r} is not declared for {contract.probe}"
        )
    candidates = validate_candidates(evidence.candidates)
    winner = select_candidate_index(
        candidates, rule, tie_policy=contract.completion_tie_policy
    )
    correctness = _validated_correctness(evidence, rule, winner, len(candidates))
    selected = candidates[winner]
    return CompletionSelection(
        evidence.query_id,
        winner,
        selected.continuation,
        rule,
        contract.completion_tie_policy,
        correctness,
    )


def validate_probe_evidence(
    probe: str, evidence: Iterable[query_types.QueryEvidence]
) -> tuple[query_types.QueryEvidence, ...]:
    contract = get_probe_contract(probe)
    rows = tuple(evidence)
    for row in rows:
        if row.probe != probe:
            raise ValueError(f"Evidence probe does not match {probe!r}")
        if type(row.query_id) is not str or not row.query_id:
            raise ValueError("Evidence requires a query ID")
    _reject_duplicate_query_ids(rows)
    invalid = sorted({row.task_core for row in rows} - set(contract.task_counts))
    if invalid:
        raise ValueError(f"Evidence has unknown task core: {invalid}")
    counts = Counter(row.task_core for row in rows)
    if (
        dict(counts) != dict(contract.task_counts)
        or len(rows) != contract.expected_count
    ):
        raise ValueError(f"Evidence count does not match contract for {probe}")
    return tuple(sorted(rows, key=lambda row: row.query_id))


def _score(candidate: query_types.CandidateEvidence, field: str) -> float:
    if field not in candidate.scores:
        raise ValueError(f"Missing {field} score")
    value = candidate.scores[field]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Selection score {field!r} must be finite")
    return float(value)


def _validated_correctness(
    evidence: query_types.QueryEvidence,
    rule: str,
    winner: int,
    candidate_count: int,
) -> bool:
    reference = evidence.reference_index
    if type(reference) is not int or reference not in range(candidate_count):
        raise ValueError("Evidence has an invalid reference index")
    if rule not in evidence.correctness:
        raise ValueError(f"Missing correctness for {rule}")
    value = evidence.correctness[rule]
    if type(value) is not bool:
        raise ValueError(f"Correctness for {rule} must be boolean")
    if value != (winner == reference):
        raise ValueError(f"Correctness for {rule} is inconsistent with winner")
    return value


def _reject_duplicate_query_ids(
    rows: tuple[query_types.QueryEvidence, ...],
) -> None:
    counts = Counter(row.query_id for row in rows)
    duplicates = sorted(query_id for query_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate query ID: {duplicates}")


for _probe_contract in PROBE_CONTRACTS.values():
    validate_selection_contract(_probe_contract)
_OBJECTIVES = {
    "model_completion": {"completion_target", "joint_sequence"},
    "reference_answer": {"completion_target", "joint_sequence"},
    "prompt_only": {"prompt_next_token"},
}


def compile_query_bundle(
    request: CompileQueryRequest,
) -> query_types.CompiledQueryBundle:
    from data_attribution.query_conditions.io import (
        validate_compiled_bundle,
    )

    _validate_request(request)
    contract = get_probe_contract(request.probe)
    rows = _validated_rows(request)
    _validate_selection_axes(request, contract.primary_rule)
    study_role = _study_role(request, contract.primary_rule)
    references = _reference_records(request, contract)
    records = tuple(
        _compile_record(request, row, contract.primary_rule, references) for row in rows
    )
    selection_rule = records[0].selection_rule
    bundle = query_types.CompiledQueryBundle(
        probe=request.probe,
        condition=request.condition,
        objective=request.objective,
        selection_rule=selection_rule,
        completion_tie_policy=contract.completion_tie_policy,
        rendering="canonical_plain",
        study_role=study_role,
        records=records,
    )
    validate_compiled_bundle(bundle)
    return bundle


def _validate_request(request: CompileQueryRequest) -> None:
    if type(request) is not CompileQueryRequest:
        raise ValueError("Compilation requires an immutable request")
    if type(request.probe) is not str or not request.probe:
        raise ValueError("Compilation requires a probe")
    if request.condition not in query_types.QUERY_CONDITIONS:
        raise ValueError("Compilation has an invalid condition")
    if request.objective not in query_types.QUERY_OBJECTIVES:
        raise ValueError("Compilation has an invalid objective")
    if request.objective not in _OBJECTIVES[request.condition]:
        raise ValueError("Invalid condition and objective combination")


def _validated_rows(
    request: CompileQueryRequest,
) -> tuple[query_types.QueryEvidence, ...]:
    if not request.evidence:
        raise ValueError("Compilation requires query evidence")
    for row in request.evidence:
        if type(row) is not query_types.QueryEvidence or row.probe != request.probe:
            raise ValueError("Compilation evidence probe does not match request")
        _validate_evidence_identity(row)
    rows = validate_probe_evidence(request.probe, request.evidence)
    checkpoints = {(row.model_id, row.model_revision) for row in rows}
    if len(checkpoints) != 1:
        raise ValueError("Compilation evidence has mixed base model checkpoints")
    return rows


def _validate_evidence_identity(row: query_types.QueryEvidence) -> None:
    values = (row.query_id, row.model_id, row.model_revision)
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError("Compilation evidence has invalid identity metadata")
    if type(row.prompt) is not str or not row.prompt:
        raise ValueError("Compilation evidence requires a nonempty prompt")


def _validate_selection_axes(request: CompileQueryRequest, primary_rule: str) -> None:
    if request.selection_rule is not None and (
        type(request.selection_rule) is not str or not request.selection_rule.strip()
    ):
        raise ValueError("Explicit selection rule must be a nonempty string")
    if request.condition != "model_completion" and request.selection_rule is not None:
        raise ValueError("Selection rules are accepted only for Model-completion")
    selected_rule = (
        primary_rule if request.selection_rule is None else request.selection_rule
    )
    if request.objective == "joint_sequence" and selected_rule != primary_rule:
        raise ValueError("Joint-sequence permits only one sensitivity axis")
    if request.condition != "reference_answer" and request.reference_inputs:
        raise ValueError("Reference inputs are accepted only for reference-answer")


def _study_role(
    request: CompileQueryRequest, primary_rule: str
) -> query_types.StudyRole:
    selected_rule = (
        primary_rule if request.selection_rule is None else request.selection_rule
    )
    if request.objective == "joint_sequence" or selected_rule != primary_rule:
        return "sensitivity"
    return "primary"


def _reference_records(
    request: CompileQueryRequest, contract: ProbeContract
) -> ReferenceMap:
    if request.condition != "reference_answer":
        return {}
    resolution = resolve_reference_inputs(
        request.evidence, request.reference_inputs, contract=contract
    )
    validate_reference_coverage(contract, request.evidence, resolution)
    if resolution.exclusions:
        raise ValueError("Current reference-answer compilation rejects exclusions")
    inputs = {item.query_id: item for item in resolution.inputs}
    return {
        record.query_id: (record, inputs[record.query_id])
        for record in resolution.records
    }


def _compile_record(
    request: CompileQueryRequest,
    evidence: query_types.QueryEvidence,
    primary_rule: str,
    references: ReferenceMap,
) -> query_types.CanonicalQueryRecord:
    is_model = request.condition == "model_completion"
    rule = primary_rule if request.selection_rule is None else request.selection_rule
    base = select_base_completion(evidence, rule)
    reference_entry = references.get(evidence.query_id)
    reference = reference_entry[0] if reference_entry is not None else None
    reference_input = reference_entry[1] if reference_entry is not None else None
    target, source = _target(request.condition, base, reference)
    public_rule = base.selection_rule if is_model else None
    return query_types.CanonicalQueryRecord(
        query_id=evidence.query_id,
        probe=evidence.probe,
        task_core=evidence.task_core,
        prompt=evidence.prompt,
        target=target,
        target_source=source,
        condition=request.condition,
        selection_rule=public_rule,
        lineage=evidence.lineage + _reference_lineage(reference),
        base_completion_tie_policy=base.tie_policy,
        **_reference_fields(reference, reference_input, evidence),
        base_completion_index=base.candidate_index,
        base_completion_correct=base.is_correct,
        base_completion_model_id=evidence.model_id,
        base_completion_model_revision=evidence.model_revision,
        base_completion_selection_rule=base.selection_rule,
    )


def _target(
    condition: query_types.QueryCondition,
    base: CompletionSelection,
    reference: query_types.ReferenceSelection | None,
) -> tuple[str | None, str | None]:
    if condition == "model_completion":
        return base.continuation, "base_completion"
    if condition == "prompt_only":
        return None, None
    if reference is None:
        raise ValueError("Reference-answer compilation is missing a resolved reference")
    return reference.continuation, "benchmark_reference"


def _reference_fields(
    reference: query_types.ReferenceSelection | None,
    reference_input: query_types.ReferenceInput | None,
    evidence: query_types.QueryEvidence,
) -> dict[str, Any]:
    if reference is None or reference_input is None:
        return {}
    kind, payload = query_types.reference_input_snapshot(reference_input)
    return {
        "reference_basis": reference.basis,
        "reference_index": reference.candidate_index,
        "reference_continuation": reference.continuation,
        "reference_source_revision": reference.source.revision,
        "reference_source_sha256": reference.source.artifact_sha256,
        "reference_input_kind": kind,
        "reference_input_payload": payload,
        "reference_candidate_continuations": tuple(
            candidate.continuation for candidate in evidence.candidates
        ),
    }


def _reference_lineage(
    reference: query_types.ReferenceSelection | None,
) -> tuple[query_types.ArtifactLineage, ...]:
    if reference is None:
        return ()
    return (
        query_types.ArtifactLineage(
            "reference_source",
            f"reference://{reference.basis}",
            reference.source.artifact_sha256,
            reference.source.revision,
        ),
    )
