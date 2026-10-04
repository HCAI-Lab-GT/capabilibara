# COLM 2026 source snapshot

The curated source files and four aggregate CSVs were copied from research
revision `5b7b9cc6eff3980087e41fb7b0c07a70638e4484`. The
[`MANIFEST.sha256`](MANIFEST.sha256) file records checksums for the 172 source
snapshot files. Four import-only changes move abstract collection types to
`collections.abc` so the published subset passes Ruff lint. The source
snapshot omits private paths, raw text, query records, document-level influence
scores, and operational outputs.

The aggregate CSVs each contain 576 unique topic-format bins. The working-set
manifest is in the existing [public sample dataset](https://huggingface.co/datasets/HCAI-Lab-GT/dolma3-6t-sample-10000-docs)
at dataset revision `561e73c7e0ad35c04f386bae1e3dd39dfb6755e7`.
[ARC-Challenge adapters](https://huggingface.co/buckets/HCAI-Lab-GT/unlearn-binlevel-arc-ngdiff-adapters)
are in a public Hugging Face bucket. Buckets are mutable, so the link identifies
the public collection rather than an immutable model revision.

For this snapshot, Ruff lint passed on `src/`, `tests/`, and the published
analysis and query-data scripts. The local unit suite reported 86 passed,
1 expected failure, and 1 unexpected pass. Static checks confirmed the CSV
schema and row counts. No fresh-clone dependency setup or production
experiment was run for this release.
