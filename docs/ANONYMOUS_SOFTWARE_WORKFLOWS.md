# Anonymous Software Workflow Map

This file maps user-facing commands to the source paths that implement them. Read it after [ANONYMOUS_SOFTWARE_GUIDE.md](ANONYMOUS_SOFTWARE_GUIDE.md).

## Corpus And Manifest Workflows

Use these workflows to construct or inspect corpus metadata without treating generated outputs as source code:

| Command | Purpose |
|---|---|
| `data-attribution-corpus-manifest` | Build a corpus manifest from source shards and sidecar metadata. |
| `data-attribution-manifest-sample` | Draw or inspect manifest-backed samples. |
| `data-attribution-draw-working-sample` | Materialize a working sample from manifest inputs. |
| `data-attribution-materialize-sample` | Convert sampled manifests into materialized sample files. |
| `data-attribution-sidecar-schema-inventory` | Inventory sidecar schemas before manifest construction. |
| `data-attribution-quality-sidecars` | Build quality sidecar outputs. |
| `data-attribution-quality-validation` | Validate quality sidecar outputs. |

Important code paths:

1. `src/dolma/manifest_fields.py` defines topic, format, and bin conventions.
2. `src/dolma/pool_sample/` implements pool and stratified sampling.
3. `src/data_attribution/recipes/corpus_manifest.py` builds the unified manifest.
4. `src/data_attribution/recipes/sidecar_manifest/` handles sidecar-backed manifest artifacts.

## Attribution Workflows

Use these workflows for query preparation, score processing, and attribution summaries:

| Command | Purpose |
|---|---|
| `data-attribution-trackstar-query` | Write query JSONL files for attribution runs. |
| `data-attribution-trackstar-prepare` | Prepare query gradients or score inputs. |
| `data-attribution-trackstar-dot-score` | Compute dot-product score outputs. |
| `data-attribution-trackstar-aggregate` | Aggregate score shards. |
| `data-attribution-trackstar-bin-aggregate` | Aggregate scores by corpus bin. |
| `data-attribution-trackstar-bin-aggregate-split` | Aggregate scores by bin and correctness split. |
| `data-attribution-trackstar-bin-aggregate-perquery` | Preserve per-query bin aggregates. |
| `data-attribution-trackstar-extract` | Extract top-k attribution outputs from score matrices. |
| `data-attribution-trackstar-proponent-examples` | Join influential documents back to readable examples. |
| `data-attribution-socialtda-trackstar-analysis` | Build cross-benchmark attribution summaries. |

Important code paths:

1. `src/data_attribution/attribution/trackstar/queries.py` writes query inputs.
2. `src/data_attribution/attribution/trackstar/prepare.py` prepares scoring inputs.
3. `src/data_attribution/attribution/trackstar/dot_score.py` computes score outputs.
4. `src/data_attribution/attribution/trackstar/bin_aggregate*.py` build bin-level summaries.
5. `src/data_attribution/analysis/socialtda_trackstar/` turns score summaries into benchmark-level analyses.

## Evaluation Export Workflows

Use these workflows to run evaluation exports and convert model outputs into attribution-ready rows:

| Command | Purpose |
|---|---|
| `data-attribution-run-olmes-evaluation` | Run configured evaluation tasks. |
| `data-attribution-export-olmes-predictions` | Export predictions into a stitched row format. |
| `data-attribution-olmes-query-index` | Build query index files from exported rows. |
| `data-attribution-olmes-instruct-manifest` | Build an instruction-variant query manifest. |
| `data-attribution-olmes-instruct-cot-manifest` | Build a chain-of-thought query manifest. |

Important code paths:

1. `src/data_attribution/recipes/olmes_evaluation.py` runs evaluation tasks.
2. `src/data_attribution/recipes/olmes_predictions_export.py` exports prediction rows.
3. `src/data_attribution/recipes/olmes_query_manifest_rows.py` converts stitched rows into query rows.
4. `tests/data_attribution/recipes/` contains compatibility checks for these recipe surfaces.

## Figure And Report Workflows

Use these workflows to regenerate tables, figures, and HTML reports:

| Command | Purpose |
|---|---|
| `dolma-eda` | Build corpus EDA outputs. |
| `dolma-corpus-stats` | Build corpus statistics reports. |
| `data-attribution-bin-analysis` | Summarize bin-level corpus statistics. |
| `data-attribution-weborganizer-report` | Build topic-format report outputs. |
| `data-attribution-paper-figures` | Alias for the current paper-figure generation path. |

Important code paths:

1. `src/dolma/distribution_report/cli.py` is the report CLI entry point.
2. `src/dolma/distribution_report/runner.py` coordinates corpus and sampling reports.
3. `src/dolma/distribution_report/influence_runner.py` coordinates attribution figures.
4. `src/dolma/distribution_report/style.py` centralizes figure styling.
5. `docs/VISUALIZATION_INVENTORY.md` catalogs figure-producing modules, but it may include internal context.

## Lexical Profile Workflows

Use the lexical profile package for bin-level text characterization and group comparisons:

| Path | Purpose |
|---|---|
| `src/data_attribution/rq4_lexical/pipeline.py` | Coordinates lexical profile extraction. |
| `src/data_attribution/rq4_lexical/aggregate.py` | Aggregates profile outputs. |
| `src/data_attribution/rq4_lexical/bootstrap.py` | Computes bootstrap intervals. |
| `configs/rq4_*.yaml` | Defines lexical dimensions, format clusters, and benchmark groups. |
| `tests/test_rq4_lexical_counters.py` | Regression tests for lexical counters and artifacts. |

## Unlearning Workflows

Use the unlearning package for targeted retain/forget experiments:

| Path | Purpose |
|---|---|
| `src/unlearning/train.py` | Training entry point. |
| `src/unlearning/configs/` | Training, model, data, and trainer configuration. |
| `src/unlearning/data/` | Forget and retain data sampling. |
| `src/unlearning/trainer/` | Trainer implementations and cooldown logic. |
| `src/unlearning/eval_harness.py` | Evaluation harness integration. |
| `tests/unlearning/` | Unit tests for sampling, trainer utilities, and evaluation helpers. |
