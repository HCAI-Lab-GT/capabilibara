from data_attribution.evaluation import olmobaseeval


def test_probe_registry_contains_target_probes() -> None:
    expected = {
        "arc_mc",
        "mmlu_stem",
        "medmcqa",
        "medqa",
        "sciq",
        "openbookqa",
        "mmlu_humanities",
        "mmlu_social_science",
        "mmlu_other",
        "csqa",
        "piqa",
        "socialiqa",
        "coqa_gen2mc",
        "drop_gen2mc",
        "jeopardy_gen2mc",
        "naturalqs_gen2mc",
        "squad_gen2mc",
        "hellaswag_rc",
        "winogrande_rc",
        "lambada",
        "basic_skills_rc",
        "drop_genqa",
        "jeopardy_genqa",
        "naturalqs_genqa",
        "squad_genqa",
        "coqa_genqa",
    }

    assert set(olmobaseeval.PROBE_BY_ID) == expected


def test_native_genqa_probes_are_evaluation_only() -> None:
    for probe_id in {
        "drop_genqa",
        "jeopardy_genqa",
        "naturalqs_genqa",
        "squad_genqa",
        "coqa_genqa",
    }:
        probe = olmobaseeval.PROBE_BY_ID[probe_id]
        assert probe.socialtda_mode == "evaluation_only"
        assert probe.eligibility_reason == "native_genqa_f1_evaluation_only"


def test_paper_covered_probe_scope_is_excluded_from_new_attribution() -> None:
    assert olmobaseeval.PAPER_COVERED_PROBE_IDS == {
        "arc_mc",
        "mmlu_stem",
        "mmlu_social_science",
        "socialiqa",
    }
    assert "medqa" in olmobaseeval.socialtda_probe_ids(exclude_paper_covered=True)
    assert "arc_mc" not in olmobaseeval.socialtda_probe_ids(exclude_paper_covered=True)
    assert len(olmobaseeval.socialtda_probe_ids(exclude_paper_covered=True)) == 17


def test_probe_lookup_prefers_alias_before_task_core() -> None:
    gen2mc = olmobaseeval.probe_for_task(
        task_core="coqa",
        task_name="coqa:mc::xlarge",
        task_alias="coqa:mc::gen2mc:xlarge",
    )
    native = olmobaseeval.probe_for_task(
        task_core="coqa",
        task_name="coqa",
        task_alias="coqa::xlarge",
    )

    assert gen2mc is not None
    assert native is not None
    assert gen2mc.probe_id == "coqa_gen2mc"
    assert native.probe_id == "coqa_genqa"


def test_openbookqa_legacy_probe_resolves_without_reordering_existing_probes() -> None:
    expected_probe_ids = (
        "arc_mc",
        "mmlu_stem",
        "medmcqa",
        "medqa",
        "sciq",
        "openbookqa",
        "mmlu_humanities",
        "mmlu_social_science",
        "mmlu_other",
        "csqa",
        "piqa",
        "socialiqa",
        "coqa_gen2mc",
        "drop_gen2mc",
        "jeopardy_gen2mc",
        "naturalqs_gen2mc",
        "squad_gen2mc",
        "hellaswag_rc",
        "winogrande_rc",
        "lambada",
        "basic_skills_rc",
        "drop_genqa",
        "jeopardy_genqa",
        "naturalqs_genqa",
        "squad_genqa",
        "coqa_genqa",
    )

    assert tuple(olmobaseeval.PROBE_BY_ID) == expected_probe_ids
    probe_by_id = olmobaseeval.PROBE_BY_ID["openbookqa"]
    assert probe_by_id.family == "mc_stem"
    for task in (
        {"task_core": "", "task_alias": "openbookqa:mc::olmes"},
        {"task_core": "", "task_name": "openbookqa:mc"},
        {"task_core": "openbookqa"},
    ):
        probe = olmobaseeval.probe_for_task(**task)
        assert probe == probe_by_id
