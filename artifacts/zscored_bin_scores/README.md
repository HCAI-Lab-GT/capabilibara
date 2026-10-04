# Aggregate influence by corpus bin

The `aggregated/` directory contains one CSV for each benchmark in the paper's
2-by-2 design: SocialIQA, MMLU Social Sciences, ARC-Challenge, and MMLU STEM.
Each file has 576 rows, one for every WebOrganizer topic-format bin. The files
contain aggregate statistics and no document identifiers, text, query records,
or document-level influence scores.

| Column | Meaning |
| --- | --- |
| `topic_label` | WebOrganizer topic. |
| `format_label` | WebOrganizer format. |
| `mean_score` | Mean influence across documents in the bin. |
| `zscore` | Bin mean standardized within this benchmark. |
| `doc_count` | Number of scored documents in the bin. |

Each benchmark is standardized independently. Compare `zscore` values across
benchmarks; the raw `mean_score` scales are different. The export logic is in
[`scripts/analysis/export_zscored_bin_scores.py`](../../scripts/analysis/export_zscored_bin_scores.py).
The source revision and file checksums are recorded in
[`MANIFEST.sha256`](../../MANIFEST.sha256) and the release history.

The four aggregate CSVs are licensed under
[CC BY 4.0](../../LICENSE-data.txt). Attribute the
[*Capability Provenance in Language Models* paper](https://arxiv.org/abs/2606.19625)
when reusing them, link the license, and indicate changes. The statistics were
derived using the [Dolma3 corpus](https://huggingface.co/datasets/allenai/dolma3_pool)
and [WebOrganizer taxonomy](https://proceedings.mlr.press/v267/wettig25a.html);
their upstream materials retain their own terms. No source documents are
redistributed in these CSVs.
