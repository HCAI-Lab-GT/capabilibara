<div align="center">
  <img src="public/static/images/capabilibara.svg" width="192" height="192" alt="Capabilibara mascot">
  <h1>Capability Provenance in Language Models</h1>
  <p><em>A Case Study in Social Reasoning</em></p>

  <!-- badges -->
  <p>
    <a href="https://arxiv.org/abs/2606.19625"><img src="https://img.shields.io/badge/%F0%9F%93%84%20paper-arXiv%202606.19625-1f2328.svg" alt="Paper"></a>
    <a href="https://eilab.gatech.edu/capabilibara/"><img src="https://img.shields.io/badge/venue-COLM%202026-762a83.svg" alt="COLM 2026"></a>
    <a href="https://eilab.gatech.edu/capabilibara/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20website-project%20page-2f6fa8.svg" alt="Project page"></a>
    <img src="https://img.shields.io/badge/code-source%20available-2f6fa8.svg" alt="Code source available">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-a93428.svg" alt="License: AGPL-3.0"></a>
    <img src="https://img.shields.io/badge/python-3.12-2f6fa8.svg" alt="Python 3.12">
    <img src="https://img.shields.io/github/last-commit/HCAI-Lab-GT/capabilibara?label=last%20commit" alt="Last commit">
    <img src="https://img.shields.io/github/stars/HCAI-Lab-GT/capabilibara?style=social" alt="Stars">
  </p>

  <p>
    <em>The software for <strong>Capability Provenance in Language Models</strong> (COLM 2026):</em><br>
    a training-data attribution pipeline that maps which regions of a pretraining corpus
    support different reasoning capabilities, and validates those regions with targeted unlearning.
  </p>
</div>

## Overview

This repository contains a curated source snapshot for **"Capability Provenance in Language Models: A Case Study in Social Reasoning"** (COLM 2026). The study runs gradient-based training-data attribution over a stratified sample of the Dolma3 corpus, aggregates document-level influence to corpus regions defined by the WebOrganizer 24×24 topic-by-format taxonomy, and validates the flagged regions with selective unlearning.

The full results, figures, and method write-up live in the **[paper](https://arxiv.org/abs/2606.19625)** and on the **[project site](https://eilab.gatech.edu/capabilibara/)**. This repository is the software.

The release includes source for the study's sampling, attribution, aggregation, and unlearning components and the four 576-bin aggregate result files. It excludes document-level influence scores, raw training text, and operational run outputs. The full production pipeline has not been rerun from this public snapshot.
The [source snapshot record](SOURCE_SNAPSHOT.md) gives the source revision, checksums, and checks run before publication.

## Results at a glance

The headline scale of the study (full analysis and figures in the paper and on the site):

| | |
|---|---|
| **576** topic-format bins | **5.68M** documents in the working set |
| **4** contrastive benchmarks | **paired unlearning** validation (Wilcoxon p ≈ 10⁻⁵) |

## Repository structure

```
src/
├── data_attribution/      # attribution, benchmark probes, scoring
├── dolma/                 # corpus manifests, enrichment, sampling
└── unlearning/            # unlearning data and trainer components
scripts/analysis/          # aggregate result export and selected analyses
artifacts/zscored_bin_scores/aggregated/  # four 576-row result files
```

The code snapshot records the component implementations. Internal run orchestration and private artifact paths are outside this release.

## Quick start

```bash
git clone https://github.com/HCAI-Lab-GT/capabilibara.git
cd capabilibara
bash scripts/bootstrap_local_deps.sh
uv sync --no-build-isolation
```

This is the recorded dependency setup for Python 3.12, with pinned Bergson and OLMES revisions. It has not been tested from a fresh public clone, and the complete production workflow requires large upstream datasets, model weights, and compute. See [third-party notices](THIRD_PARTY_NOTICES.md) for the patch licenses.

## Method

The central move is aggregation: every benchmark query is traced back to many documents, then summarized into comparable corpus regions. Dolma3 is de-duplicated, classified into WebOrganizer's 24 topic × 24 format taxonomy, and sampled into a 576-bin working set. Benchmark query gradients come from OLMo3-7B Instruct; document gradients and corpus-side curvature are computed on OLMo3-7B Base. Document-level influence is aggregated to a signed 576-bin influence matrix, contrasted across benchmarks, and the flagged regions are tested causally with selective unlearning. Figure 1 above shows the pipeline; the paper carries the depth.

## Data & models

| Component | Detail |
|---|---|
| Base model | `allenai/Olmo-3-1025-7B` (OLMo3-7B Base) |
| Instruction model | `allenai/Olmo-3-7B-Instruct` (query gradients) |
| Corpus | Dolma3, de-duplicated, ~1.26B unique documents |
| Working set | 5.68M documents, stratified across 576 WebOrganizer bins |
| Benchmarks | SocialIQA, MMLU Social Sciences, ARC-Challenge, MMLU STEM (headline 2×2); also GSM8K, ARC-Easy, BBH social tasks |
| Attribution | gradient-based TDA via TrackStar (Bergson) |
| Compute | ~37K H200-equivalent GPU-hours |

Public aggregate files and external artifacts are listed below.

## Release artifacts

- [Source and four 576-bin aggregate CSVs](artifacts/zscored_bin_scores/aggregated/) are included in this repository. Each CSV has one row per WebOrganizer topic-format bin for one benchmark.
- The [5.68M-document working-sample manifest](https://huggingface.co/datasets/HCAI-Lab-GT/dolma3-6t-sample-10000-docs/blob/main/working_sample_manifest.parquet), [sample contract](https://huggingface.co/datasets/HCAI-Lab-GT/dolma3-6t-sample-10000-docs/blob/main/sample_contract.json), and [bin summary](https://huggingface.co/datasets/HCAI-Lab-GT/dolma3-6t-sample-10000-docs/blob/main/bin_summary.csv) are public metadata files in the existing dataset.
- [ARC-Challenge NGDiff adapters](https://huggingface.co/buckets/HCAI-Lab-GT/unlearn-binlevel-arc-ngdiff-adapters) are in an existing public artifact bucket. Selected completed runs have an `adapter/` directory; the bucket also contains intermediate checkpoints.

This release does not include document-level attribution scores or raw training text. The linked upstream sample dataset is a separate public resource and includes corpus data in addition to the metadata files linked above.

## Limitations

- Attribution is an analytic lens, not an exact proof of causal necessity for individual documents; unlearning validates aggregate patterns without eliminating approximation error.
- The analysis runs on a 5.68M-document stratified working set drawn from the ~1.26B-document population; results do not characterize every document in the full corpus.
- Unlearning shows a corpus region is load-bearing; it does not explain the mechanism by which those documents shape behavior.
- The deep-dive is measured on one open-data ecosystem (OLMo3-7B / Dolma3). Generalization across model families is an open question.
- The released influence results are aggregate bin-level statistics; document-level attribution scores are not published.

## Citation

If you use this software, please cite the paper. GitHub's **"Cite this repository"** button (from `CITATION.cff`) surfaces the same BibTeX; the canonical text matches the project site.

```bibtex
@inproceedings{matlin2026capabilityprovenance,
  title         = {Capability Provenance in Language Models: A Case Study in Social Reasoning},
  author        = {Glenn Matlin and Chandreyi Chakraborty and Saehee Eom and Mika Okamoto and
                   Rayan Castilla and Louis Jaburi and Alvin Deng and Taywon Min and
                   Lucia Quirke and Stella Biderman and Mark Riedl},
  booktitle     = {Proceedings of the Conference on Language Modeling (COLM 2026)},
  year          = {2026},
  eprint        = {2606.19625},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.19625}
}
```

## License

- **Code** in this repository is licensed under [AGPL-3.0](LICENSE).
- **Four aggregate CSVs** under `artifacts/zscored_bin_scores/aggregated/` are licensed under [CC BY 4.0](LICENSE-data.txt); see their [schema and attribution notes](artifacts/zscored_bin_scores/README.md).
- **Website content** (`public/`) is licensed under [CC BY-SA 4.0](LICENSE-website.md).

## Acknowledgments

This work was supported by the Georgia Institute of Technology Experimental AI Lab (EILab), EleutherAI, and the MATS program. See the paper for the full acknowledgments.
