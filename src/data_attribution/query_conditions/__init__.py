"""Public interfaces for controlled query-condition compilation.

Model-completion records use the audited ``per_character`` score as the
attribution-target completion contract. This target-selection rule is
intentionally distinct from the main evaluation-reporting path, whose primary
multiple-choice metric remains raw ``sum_logits``/``acc_raw`` except for its
documented SimpleToM fallback.
"""

from data_attribution.query_conditions.compiler import (
    CompileQueryRequest,
    compile_query_bundle,
)
from data_attribution.query_conditions.io import read_compiled_query_bundle
from data_attribution.query_conditions.olmes import (
    OlmesTaskSource,
    load_olmes_evidence,
    load_olmes_source_manifest,
)

__all__ = [
    "CompileQueryRequest",
    "OlmesTaskSource",
    "compile_query_bundle",
    "load_olmes_evidence",
    "load_olmes_source_manifest",
    "read_compiled_query_bundle",
]
