"""OlmoBaseEval probe lookup helpers.

OlmoBaseEval is a separate expansion track from the current SocialTDA paper
result set. See `src/data_attribution/evaluation/README.md` before treating
`base_olmobaseeval` artifacts as paper-facing outputs.
"""

from __future__ import annotations

from data_attribution.evaluation.olmobaseeval_registry import (
    CLOZE_EVALUATION_ALIASES as CLOZE_EVALUATION_ALIASES,
    CONDITION as CONDITION,
    NATIVE_GENQA_EVALUATION_ALIASES as NATIVE_GENQA_EVALUATION_ALIASES,
    PROBES as PROBES,
    QUERY_PREFIX as QUERY_PREFIX,
    OlmoBaseEvalProbe as OlmoBaseEvalProbe,
    SocialTDAMode as SocialTDAMode,
)
from data_attribution.evaluation.olmobaseeval_subjects import (
    BASIC_SKILLS as BASIC_SKILLS,
    MMLU_HUMANITIES as MMLU_HUMANITIES,
    MMLU_OTHER as MMLU_OTHER,
    MMLU_SOCIAL_SCIENCES as MMLU_SOCIAL_SCIENCES,
    MMLU_STEM as MMLU_STEM,
)

PROBE_BY_ID = {probe.probe_id: probe for probe in PROBES}
PAPER_COVERED_PROBE_IDS = frozenset(
    {"arc_mc", "mmlu_stem", "mmlu_social_science", "socialiqa"}
)


def _key_map() -> dict[str, OlmoBaseEvalProbe]:
    mapping: dict[str, OlmoBaseEvalProbe] = {}
    for probe in PROBES:
        for key in probe.task_aliases + probe.task_names + probe.task_cores:
            mapping.setdefault(key, probe)
    return mapping


TASK_KEY_TO_PROBE = _key_map()


def probe_for_task(
    *, task_core: str, task_name: str = "", task_alias: str = ""
) -> OlmoBaseEvalProbe | None:
    for key in (task_alias, task_name, task_core):
        probe = TASK_KEY_TO_PROBE.get(key)
        if probe is not None:
            return probe
    return None


def valid_splits() -> set[str]:
    return set(PROBE_BY_ID)


def socialtda_probe_ids(*, exclude_paper_covered: bool = False) -> set[str]:
    probe_ids = {probe.probe_id for probe in PROBES if probe.socialtda_mode == "cloze"}
    if exclude_paper_covered:
        return probe_ids - PAPER_COVERED_PROBE_IDS
    return probe_ids


def evaluation_only_probe_ids() -> set[str]:
    return {
        probe.probe_id for probe in PROBES if probe.socialtda_mode == "evaluation_only"
    }
