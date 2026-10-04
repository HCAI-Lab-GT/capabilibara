from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from data_attribution.evaluation.olmobaseeval_subjects import (
    MMLU_HUMANITIES,
    MMLU_OTHER,
    MMLU_SOCIAL_SCIENCES,
    MMLU_STEM,
)

SocialTDAMode = Literal["cloze", "evaluation_only"]

CONDITION = "base_olmobaseeval"
QUERY_PREFIX = "olmes_base_"

CLOZE_EVALUATION_ALIASES = (
    "olmo3:base:stem_qa_mc",
    "olmo3:base:nonstem_qa_mc",
    "hellaswag:rc::xlarge",
    "winogrande:rc::xlarge",
    "lambada",
    "basic_skills:rc::olmes",
)

NATIVE_GENQA_EVALUATION_ALIASES = (
    "drop::xlarge",
    "jeopardy::xlarge",
    "naturalqs::xlarge",
    "squad::xlarge",
    "coqa::xlarge",
)


@dataclass(frozen=True)
class OlmoBaseEvalProbe:
    probe_id: str
    label: str
    family: str
    socialtda_mode: SocialTDAMode
    task_aliases: tuple[str, ...] = ()
    task_names: tuple[str, ...] = ()
    task_cores: tuple[str, ...] = ()

    @property
    def eligibility_reason(self) -> str:
        if self.socialtda_mode == "cloze":
            return "cloze_compatible"
        return "native_genqa_f1_evaluation_only"


def _mmlu_probe(
    probe_id: str,
    label: str,
    family: str,
    subjects: tuple[str, ...],
) -> OlmoBaseEvalProbe:
    return OlmoBaseEvalProbe(
        probe_id,
        label,
        family,
        "cloze",
        task_cores=tuple(f"mmlu_{s}" for s in subjects),
    )


def _core_probe(
    probe_id: str, label: str, family: str, task_cores: tuple[str, ...]
) -> OlmoBaseEvalProbe:
    return OlmoBaseEvalProbe(probe_id, label, family, "cloze", task_cores=task_cores)


def _alias_probe(
    probe_id: str,
    label: str,
    family: str,
    task_alias: str,
    *,
    task_name: str = "",
    task_core: str = "",
) -> OlmoBaseEvalProbe:
    task_names = (task_name,) if task_name else ()
    task_cores = (task_core,) if task_core else ()
    return OlmoBaseEvalProbe(
        probe_id,
        label,
        family,
        "cloze",
        task_aliases=(task_alias,),
        task_names=task_names,
        task_cores=task_cores,
    )


def _genqa_probe(
    probe_id: str, label: str, task_alias: str, task_name: str, task_core: str
) -> OlmoBaseEvalProbe:
    return OlmoBaseEvalProbe(
        probe_id,
        label,
        "genqa",
        "evaluation_only",
        task_aliases=(task_alias,),
        task_names=(task_name,),
        task_cores=(task_core,),
    )


PROBES = (
    _core_probe("arc_mc", "ARC MC", "mc_stem", ("arc_easy", "arc_challenge")),
    _mmlu_probe("mmlu_stem", "MMLU STEM", "mc_stem", MMLU_STEM),
    _core_probe("medmcqa", "MedMCQA MC", "mc_stem", ("medmcqa",)),
    _core_probe("medqa", "MedQA MC", "mc_stem", ("medqa_en",)),
    _core_probe("sciq", "SciQ MC", "mc_stem", ("sciq",)),
    _alias_probe(
        "openbookqa",
        "OpenBookQA MC",
        "mc_stem",
        "openbookqa:mc::olmes",
        task_name="openbookqa:mc",
        task_core="openbookqa",
    ),
    _mmlu_probe("mmlu_humanities", "MMLU Humanities", "mc_nonstem", MMLU_HUMANITIES),
    _mmlu_probe(
        "mmlu_social_science",
        "MMLU Social Science",
        "mc_nonstem",
        MMLU_SOCIAL_SCIENCES,
    ),
    _mmlu_probe("mmlu_other", "MMLU Other", "mc_nonstem", MMLU_OTHER),
    _core_probe("csqa", "CSQA MC", "mc_nonstem", ("csqa",)),
    _core_probe("piqa", "PiQA MC", "mc_nonstem", ("piqa",)),
    _core_probe("socialiqa", "SocialIQA MC", "mc_nonstem", ("socialiqa",)),
    _alias_probe(
        "coqa_gen2mc",
        "CoQA Gen2MC MC",
        "mc_nonstem",
        "coqa:mc::gen2mc:xlarge",
        task_name="coqa:mc::xlarge",
    ),
    _alias_probe(
        "drop_gen2mc",
        "DROP Gen2MC MC",
        "mc_nonstem",
        "drop:mc::gen2mc:xlarge",
        task_name="drop:mc::xlarge",
    ),
    _alias_probe(
        "jeopardy_gen2mc",
        "Jeopardy Gen2MC MC",
        "mc_nonstem",
        "jeopardy:mc::gen2mc:xlarge",
        task_name="jeopardy:mc::xlarge",
    ),
    _alias_probe(
        "naturalqs_gen2mc",
        "NaturalQs Gen2MC MC",
        "mc_nonstem",
        "naturalqs:mc::gen2mc:xlarge",
        task_name="naturalqs_open:mc::xlarge",
    ),
    _alias_probe(
        "squad_gen2mc",
        "SQuAD Gen2MC MC",
        "mc_nonstem",
        "squad:mc::gen2mc:xlarge",
        task_name="squad:mc::xlarge",
    ),
    _alias_probe("hellaswag_rc", "HellaSwag RC", "gen_rc", "hellaswag:rc::xlarge"),
    _alias_probe("winogrande_rc", "WinoGrande RC", "gen_rc", "winogrande:rc::xlarge"),
    _alias_probe(
        "lambada",
        "Lambada",
        "gen_rc",
        "lambada",
        task_name="lambada",
        task_core="lambada",
    ),
    _alias_probe(
        "basic_skills_rc",
        "Basic Skills RC",
        "gen_rc",
        "basic_skills:rc::olmes",
        task_core="basic_skills",
    ),
    _genqa_probe("drop_genqa", "DROP GenQA", "drop::xlarge", "drop", "drop"),
    _genqa_probe(
        "jeopardy_genqa", "Jeopardy GenQA", "jeopardy::xlarge", "jeopardy", "jeopardy"
    ),
    _genqa_probe(
        "naturalqs_genqa",
        "NaturalQs GenQA",
        "naturalqs::xlarge",
        "naturalqs_open",
        "naturalqs_open",
    ),
    _genqa_probe("squad_genqa", "SQuAD GenQA", "squad::xlarge", "squad", "squad"),
    _genqa_probe("coqa_genqa", "CoQA GenQA", "coqa::xlarge", "coqa", "coqa"),
)
