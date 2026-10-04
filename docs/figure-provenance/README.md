# Overview figure provenance

The overview and the topic/format profile figure use all benchmark queries,
including correct and incorrect answers. They use the same four CSVs under
`artifacts/zscored_bin_scores/aggregated/`. Correct-only query subsets belong
to the paper's correctness appendix and must not replace these values.

`overview-influence-all-queries.json` records the 24 selected bin values,
source CSV hashes, bin ordering, and normalization. `fig-overview.drawio` is
the editable overview source; `fig-overview.pdf` is its vector export. These
copies come from the manuscript's figure source at revision `fe372c2`.
The website hero PNG and WebP are rendered from that PDF at 2400 pixels wide.
Retained earlier assets under `2026-10-04-before-cohort-fix/` are historical
copies and must not be reused in active publication materials.

For a data update, regenerate the manuscript overview with its
`scripts/sync_overview_influence.py`, refresh these source copies, and regenerate
the website images and poster embedded figures. Verify all 24 website animation
values against the aggregate CSVs before publication. The paper and poster use
one decimal for absolute values of at least 10, and two decimals otherwise.
Color limits affect shading only; they do not change the printed values.

Run `node scripts/check_figure1_values.mjs` to compare all 24 animation values
against the CSVs in this repository. When the private sibling checkout exists,
the same command also checks its website mirror. `FIGURE1_CANONICAL_REPO` may
point at an explicitly selected source checkout. The local website preview was
visually checked on the influence-map stage after this correction.

The corrected paper PDF SHA-256 is
`b50d3f93126bdc038dedcf045310f5b9b0767f18fefa4f595aeaadc2cb45d3a0`.
The overview PDF SHA-256 is
`33e8ae83ee60876a5f53ea2cfcaa423e40e8130f47b53fa2a394972b99cde06a`.
