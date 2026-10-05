# Anonymous Software Guide

## Scope

This guide explains how to orient to the software package, find the relevant code, and run the main workflows. It is written for human reviewers and for AI-assisted code reading.

This guide avoids names, affiliations, contact paths, and private access instructions. Other repository documents may be internal notes. Do not treat every existing document as review-safe without a separate anonymity scan.

## Repository Map

| Path | Role |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, and console script entry points. |
| `src/data_attribution/` | Attribution, scoring, evaluation export, cache, and analysis code. |
| `src/data_attribution/attribution/trackstar/` | Query preparation, score aggregation, top-k extraction, and proponent-example workflows. |
| `src/data_attribution/analysis/socialtda_trackstar/` | Cross-benchmark attribution analysis and report-oriented summaries. |
| `src/dolma/` | Corpus manifest construction, sampling, sidecar processing, quality labels, and report generation. |
| `src/dolma/distribution_report/` | Figure and table generation for corpus and attribution summaries. |
| `src/data_attribution/rq4_lexical/` | Lexical profile aggregation and bootstrap utilities. |
| `src/unlearning/` | Unlearning training, data sampling, retain-pool logic, and evaluation helpers. |
| `scripts/` | Batch launchers, validation checks, analysis scripts, and data transformation utilities. |
| `configs/` | YAML and JSON configuration files for lexical and format analyses. |
| `tests/` | Unit and integration tests that define expected schemas and command behavior. |
| `artifacts/`, `results/`, `reports/`, `archive/` | Generated outputs, report material, run products, and historical records. |

## Install

The project targets Python 3.12 and uses `uv`.

For a normal clone:

```bash
bash scripts/bootstrap_local_deps.sh
uv sync --no-build-isolation
```

For a fresh git worktree:

```bash
bash scripts/bootstrap/setup_worktree.sh
```

After setup, prefer `uv run --no-sync` for routine checks:

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
uv run --no-sync pyright src
```

If a command fails because the environment was not synchronized, rerun the setup command before changing code.

## Workflow Map

Use [ANONYMOUS_SOFTWARE_WORKFLOWS.md](ANONYMOUS_SOFTWARE_WORKFLOWS.md) for command maps and workflow-specific code paths. That companion file covers corpus manifests, attribution, evaluation export, report generation, lexical profiles, and unlearning.

## How To Read The Code

For a human reviewer:

1. Start with `README_REVIEW.md`.
2. Read this guide.
3. Inspect `pyproject.toml` to see the installed command surface.
4. Read the source paths for the workflow you care about.
5. Use the matching tests to confirm expected inputs and outputs.

For AI-assisted reading:

1. Treat `pyproject.toml [project.scripts]` as the command index.
2. Treat `src/` as source of truth for behavior.
3. Treat `tests/` as executable schema documentation.
4. Treat generated files under `artifacts/`, `results/`, `reports/`, and `archive/` as outputs unless a workflow explicitly reads them as inputs.
5. If a CLI example disagrees with installed behavior, run the command with `--help` and trust the installed parser.

## Data And Artifact Boundaries

Some workflows need large external artifacts or credentials. The package still exposes the code paths without those credentials, but full reproduction may require access to external datasets or score matrices.

For anonymous review:

1. Do not include private credentials.
2. Do not include private workspace paths.
3. Do not include contact instructions.
4. Do not include lab, institution, or author identifiers.
5. Prefer review-bundle paths, local fixtures, or public artifact paths that have already been anonymized.

If a workflow needs data that is not available in the review bundle, document the missing input as an access limitation instead of editing code to hide the dependency.

## Minimal Verification

Run the lowest-cost checks first:

```bash
uv run --no-sync ruff check src tests
uv run --no-sync pyright src
uv run --no-sync pytest -q
```

For targeted checks, use the test file that matches the edited package surface. Examples:

```bash
uv run --no-sync pytest tests/dolma/pool_sample -q
uv run --no-sync pytest tests/data_attribution/analysis -q
uv run --no-sync pytest tests/unlearning -q
```

Large workflows may not be practical on a reviewer laptop. In those cases, use the unit tests, small fixtures, and command `--help` output to verify interface behavior.

## Glossary

| Term | Meaning in this repo |
|---|---|
| Bin | A topic-by-format cell used to group corpus documents. |
| Manifest | A structured table that records document identity, shard location, token counts, and labels. |
| Sidecar | Metadata stored next to source data, often used to avoid rewriting large source shards. |
| Query | A benchmark or evaluation item converted into a scoring input. |
| Score matrix | Per-query by per-document scores produced by an attribution run. |
| Top-k | The highest-scoring documents for a query or benchmark after aggregation. |
| Preconditioner | A reusable scoring artifact used by attribution workflows. |
| Working sample | A bounded corpus subset used for analysis and report generation. |
