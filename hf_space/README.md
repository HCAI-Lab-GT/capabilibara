---
title: Capabilibara - Capability Provenance in Language Models
emoji: 🦫
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Capability provenance in language models (COLM 2026).
---

# Capabilibara: Capability Provenance in Language Models
*A Case Study in Social Reasoning (COLM 2026)*

[![Paper](https://img.shields.io/badge/%F0%9F%93%84%20paper-arXiv%202606.19625-1f2328.svg)](https://arxiv.org/abs/2606.19625)
[![Venue](https://img.shields.io/badge/venue-COLM%202026-762a83.svg)](https://eilab.gatech.edu/capabilibara/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-a93428.svg)](https://github.com/eilab-gt/capabilibara/blob/main/LICENSE)

Hosted by **HCAI-Lab** (Human-Centered AI Lab / EILab).

## Overview

This Space provides an interactive interface for exploring **Capability Provenance in Language Models: A Case Study in Social Reasoning** (COLM 2026).

The pipeline maps which regions of pretraining text (Dolma3 stratified into WebOrganizer's 24×24 topic-by-format taxonomy) support social vs. STEM reasoning, validated with targeted unlearning.

### Features
- **Matrix Explorer**: Browse 576 corpus bins across 24 topics and 24 formats.
- **Influence Breakdown**: Compare signed influence across SocialIQA, MMLU Social Sciences, ARC-Challenge, and MMLU STEM.
- **Paper & Citation**: Access the arXiv paper, bibtex, and repository details.

## Citation

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
