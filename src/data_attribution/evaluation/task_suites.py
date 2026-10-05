TASK_TO_SUITE = {
    "gsm8k": "gsm8k",
    "socialiqa": "socialiqa",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
    "bbh_snarks": "bbh_snarks",
    "bbh_causal_judgement": "bbh_causal_judgement",
    "bbh_disambiguation_qa": "bbh_disambiguation_qa",
    "bbh_sports_understanding": "bbh_sports_understanding",
    "bbq": "bbq",
    "pub": "pub",
    "simpletom_mental-state-qa": "simpletom",
    "simpletom_behavior-qa": "simpletom",
    "simpletom_judgment-qa": "simpletom",
}

TASK_TO_SUITE.update(
    dict.fromkeys("ethics_commonsense ethics_deontology ethics_justice " "ethics_utilitarianism ethics_virtue".split(), "ethics")
)

TASK_TO_SUITE.update(
    dict.fromkeys("negotiationtom_desire negotiationtom_belief negotiationtom_intention".split(), "negotiationtom")
)

TASK_TO_SUITE.update({f"pub_{index}": "pub" for index in range(1, 15)})

TASK_TO_SUITE.update(
    dict.fromkeys("mmlu_abstract_algebra mmlu_astronomy mmlu_college_biology mmlu_college_chemistry " "mmlu_college_computer_science mmlu_college_mathematics mmlu_college_physics " "mmlu_computer_security mmlu_conceptual_physics mmlu_electrical_engineering " "mmlu_elementary_mathematics mmlu_high_school_biology mmlu_high_school_chemistry " "mmlu_high_school_computer_science mmlu_high_school_mathematics " "mmlu_high_school_physics mmlu_high_school_statistics mmlu_machine_learning".split(), "mmlu_stem")
)
TASK_TO_SUITE.update(
    dict.fromkeys("mmlu_econometrics mmlu_high_school_geography " "mmlu_high_school_government_and_politics mmlu_high_school_macroeconomics " "mmlu_high_school_microeconomics mmlu_high_school_psychology " "mmlu_human_sexuality mmlu_professional_psychology mmlu_public_relations " "mmlu_security_studies mmlu_sociology mmlu_us_foreign_policy".split(), "mmlu_social_science")
)
TASK_TO_SUITE.update(
    dict.fromkeys("mmlu_moral_scenarios mmlu_moral_disputes mmlu_philosophy " "mmlu_business_ethics".split(), "mmlu_moral")
)

TASK_TO_SUITE.update(
    {
        "tombench": "tombench",
        "stereoset": "stereoset",
        "moralexceptqa_rbqa": "moralexceptqa_rbqa",
        "morables": "morables",
    }
)

SUITE_LABEL = {
    "socialiqa": "SocialIQA",
    "gsm8k": "GSM8k",
    "mmlu_social_science": "MMLU Social Science",
    "mmlu_moral": "MMLU Moral / Humanities",
    "mmlu_stem": "MMLU STEM",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "bbh_snarks": "BBH Snarks",
    "bbh_causal_judgement": "BBH Causal Judgement",
    "bbh_disambiguation_qa": "BBH Disambiguation QA",
    "bbh_sports_understanding": "BBH Sports Understanding",
    "bbq": "BBQ",
    "ethics": "ETHICS",
    "negotiationtom": "NegotiationToM",
    "pub": "PUB",
    "simpletom": "SimpleToM",
    "tombench": "ToMBench",
    "stereoset": "StereoSet",
    "moralexceptqa_rbqa": "MoralExceptQA / RBQA",
    "morables": "MORABLES",
}
