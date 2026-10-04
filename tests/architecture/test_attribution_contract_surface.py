import ast
import importlib
import tomllib
from pathlib import Path

import pytest

import data_attribution.query_conditions as query_conditions
import data_attribution.attribution.trackstar as trackstar

CONTROLLED_QUERY_CONSOLES = {
    "data-attribution-query-conditions": "data_attribution.query_conditions.cli:main"
}
LINEAGE_CONSOLES = {
    "data-attribution-trackstar-lineage": (
        "data_attribution.attribution.trackstar.lineage_cli:main"
    )
}
FORBIDDEN_RUN_CONSOLE_MARKERS = ("run-spec", "run-manifest")


def query_condition_module_names() -> set[str]:
    package = Path(query_conditions.__file__).parent
    return {path.stem for path in package.glob("*.py") if path.name != "__init__.py"}


def trackstar_module_names() -> set[str]:
    package = Path(trackstar.__file__).parent
    return {path.stem for path in package.glob("*.py") if path.name != "__init__.py"}


def _project_scripts(repository: Path) -> dict[str, str]:
    project = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))
    return project["project"]["scripts"]


def _assert_stage1_console_surface(scripts: dict[str, str], repository: Path) -> None:
    query_candidates = {
        name: target
        for name, target in scripts.items()
        if "query-conditions" in _normalized_console_text(name)
        or "query-conditions" in _normalized_console_text(target)
    }
    lineage_candidates = {
        name: target
        for name, target in scripts.items()
        if "lineage" in _normalized_console_text(name)
        or "lineage" in _normalized_console_text(target)
    }
    assert query_candidates == CONTROLLED_QUERY_CONSOLES
    assert lineage_candidates == LINEAGE_CONSOLES
    for name, target in scripts.items():
        normalized = (
            f"{_normalized_console_text(name)} {_normalized_console_text(target)}"
        )
        assert not any(marker in normalized for marker in FORBIDDEN_RUN_CONSOLE_MARKERS)
        _assert_project_console_target(target, repository)


def _normalized_console_text(value: str) -> str:
    return value.casefold().replace("_", "-")


def _assert_project_console_target(target: str, repository: Path) -> None:
    assert target.count(":") == 1
    module_name, symbol = target.split(":")
    assert module_name and symbol
    source_root = repository / "src"
    if not (source_root / module_name.split(".", 1)[0]).exists():
        return
    relative_module = Path(*module_name.split("."))
    candidates = (
        (source_root / relative_module).with_suffix(".py"),
        source_root / relative_module / "__init__.py",
    )
    modules = tuple(path for path in candidates if path.is_file())
    assert len(modules) == 1
    tree = ast.parse(modules[0].read_text(encoding="utf-8"))
    assert symbol in _module_symbol_names(tree)


def _module_symbol_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            names.update(
                alias.asname or alias.name for alias in node.names if alias.name != "*"
            )
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_olmes_source_adapter_has_no_fragment_modules() -> None:
    forbidden = {
        "olmes_requests",
        "olmes_rows",
        "olmes_scores",
        "olmes_validation",
        "reconstruction",
        "revision_evidence",
        "source_artifacts",
        "source_contract",
        "source_manifest",
        "source_manifest_schema",
    }
    assert not forbidden.intersection(query_condition_module_names())


def test_query_compiler_has_no_contract_or_selection_fragments() -> None:
    forbidden = {
        "contract_roster",
        "contracts",
        "reference_coverage",
        "reference_parsing",
        "reference_types",
        "references",
        "selection",
    }
    assert not forbidden.intersection(query_condition_module_names())


def test_query_artifacts_have_no_fragment_modules() -> None:
    forbidden = {
        "artifact_hashing",
        "bundle_validation",
        "decoding",
        "jsonl",
        "preflight_io",
        "preflight_types",
        "render_binding",
        "render_tokens",
        "rendered_validation",
        "schema",
    }
    assert not forbidden.intersection(query_condition_module_names())


@pytest.mark.xfail(
    reason=(
        "main still carries scripts/attribution/build_holdout_base_query_jsonls.py, "
        "which the rollout branch retired in favour of "
        "build_legacy_stitched_query_jsonls.py. It cannot be deleted yet: "
        "scripts/smoke/phaseC_holdout_eval.sbatch still calls it. Retire the pair "
        "together, then drop this marker - an xpass here means the surface is single-path."
    ),
    strict=False,
)
def test_controlled_query_has_one_public_path() -> None:
    repository = Path(__file__).parents[2]
    duplicate_wrapper = (
        repository / "scripts" / "attribution" / "build_holdout_base_query_jsonls.py"
    )
    assert not duplicate_wrapper.exists()


def test_model_lineage_has_no_fragment_modules() -> None:
    forbidden = {
        "lineage_attestation_types",
        "lineage_file_evidence",
        "lineage_hub",
        "lineage_hub_api",
        "lineage_io",
        "lineage_json",
        "lineage_path_validation",
        "lineage_types",
    }
    assert not forbidden.intersection(trackstar_module_names())


def test_preconditioner_lineage_has_no_fragment_modules() -> None:
    forbidden = {
        "lineage_preconditioner_config",
        "lineage_preconditioner_config_hash",
        "lineage_preconditioner_config_schema",
        "lineage_preconditioner_contract",
        "lineage_preconditioner_hash",
        "lineage_preconditioner_io",
        "lineage_preconditioner_record_validation",
        "lineage_preconditioner_settings",
        "lineage_preconditioner_tensors",
        "lineage_preconditioner_types",
        "lineage_preconditioner_validation",
        "lineage_root_reader",
    }
    assert not forbidden.intersection(trackstar_module_names())


def test_index_manifest_has_no_fragment_modules() -> None:
    forbidden = {
        "gradient_artifact",
        "index_artifact_config",
        "index_artifact_config_hash",
        "index_artifact_controlled_config",
        "index_artifact_files",
        "index_artifact_lineage",
        "index_artifact_schema",
        "index_artifact_tensor",
        "index_manifest_field_validation",
        "index_manifest_hashing",
        "index_manifest_identity",
        "index_manifest_io",
        "index_manifest_json",
        "index_manifest_record_validation",
        "index_manifest_roster",
        "index_manifest_source",
        "index_manifest_types",
        "index_manifest_validation",
    }
    assert not forbidden.intersection(trackstar_module_names())


def test_run_manifest_has_no_fragment_modules() -> None:
    forbidden = {
        "run_binding_construction",
        "run_index_reference",
        "run_manifest_hashing",
        "run_manifest_io",
        "run_manifest_json",
        "run_manifest_json_nested",
        "run_manifest_json_outputs",
        "run_manifest_types",
        "run_module_validation",
        "run_output_validation",
        "run_query_artifact",
        "run_query_artifact_ref",
        "run_query_json",
        "run_reference_alignment",
        "run_reference_loading",
        "run_reference_validation",
        "run_slices",
        "run_spec",
        "run_spec_request",
    }
    assert not forbidden.intersection(trackstar_module_names())


@pytest.mark.xfail(
    reason=(
        "main still carries scripts/attribution/build_holdout_base_query_jsonls.py, "
        "which the rollout branch retired in favour of "
        "build_legacy_stitched_query_jsonls.py. It cannot be deleted yet: "
        "scripts/smoke/phaseC_holdout_eval.sbatch still calls it. Retire the pair "
        "together, then drop this marker - an xpass here means the surface is single-path."
    ),
    strict=False,
)
def test_stage1_public_surface_has_no_duplicate_paths() -> None:
    repository = Path(__file__).parents[2]
    scripts = _project_scripts(repository)
    _assert_stage1_console_surface(scripts, repository)
    attribution_scripts = repository / "scripts" / "attribution"
    legacy_adapter = attribution_scripts / "build_legacy_stitched_query_jsonls.py"
    assert tuple(sorted(attribution_scripts.glob("build_*query_jsonls.py"))) == (
        legacy_adapter,
    )
    assert not (attribution_scripts / "build_holdout_base_query_jsonls.py").exists()
    trackstar_package = Path(trackstar.__file__).parent
    assert not (trackstar_package / "completed_run.py").exists()
    assert not (trackstar_package / "compatibility.py").exists()
    assert "hashing" not in query_condition_module_names()
    assert set(query_conditions.__all__) == {
        "CompileQueryRequest",
        "OlmesTaskSource",
        "compile_query_bundle",
        "load_olmes_evidence",
        "load_olmes_source_manifest",
        "read_compiled_query_bundle",
    }
    assert not hasattr(trackstar, "__all__")
    trackstar_initializer = ast.parse(
        Path(trackstar.__file__).read_text(encoding="utf-8")
    )
    assert all(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for node in trackstar_initializer.body
    )
    preconditioner = importlib.import_module(
        "data_attribution.attribution.trackstar.lineage_preconditioner"
    )
    assert {
        "build_root_inventory",
        "root_artifact_sha256",
        "state_sha256",
    } <= set(preconditioner.__all__)
    run_manifest = trackstar_package / "run_manifest.py"
    run_manifest_source = run_manifest.read_text(encoding="utf-8")
    assert "def _read_signed_reference" not in run_manifest_source
    assert "def _signed_reference_state" not in run_manifest_source


@pytest.mark.parametrize(
    ("name", "target"),
    [
        pytest.param(
            "data-attribution-trackstar-run-spec",
            "data_attribution.attribution.trackstar.run_spec:main",
            id="run-spec-target",
        ),
        pytest.param(
            "data-attribution-run-manifest",
            "data_attribution.cli.main:main",
            id="run-manifest-name",
        ),
        pytest.param(
            "data-attribution-trackstar-lineage-alternative",
            "data_attribution.attribution.trackstar.lineage:write_attestation",
            id="alternate-lineage-target",
        ),
        pytest.param(
            "data-attribution-query-conditions-alternative",
            "data_attribution.cli.main:main",
            id="alternate-controlled-query-name",
        ),
        pytest.param(
            "data-attribution-stale-module",
            "data_attribution.not_a_module:main",
            id="missing-project-module",
        ),
        pytest.param(
            "data-attribution-stale-symbol",
            "data_attribution.cli.main:not_a_symbol",
            id="missing-project-symbol",
        ),
        pytest.param(
            "data-attribution-malformed-target",
            "data_attribution.cli.main",
            id="missing-target-colon",
        ),
        pytest.param(
            "data-attribution-malformed-target",
            "data_attribution.cli.main:main:extra",
            id="extra-target-colon",
        ),
    ],
)
def test_stage1_public_surface_rejects_forbidden_console_mutations(
    name: str, target: str
) -> None:
    repository = Path(__file__).parents[2]
    scripts = dict(_project_scripts(repository))
    scripts[name] = target
    with pytest.raises(AssertionError):
        _assert_stage1_console_surface(scripts, repository)
