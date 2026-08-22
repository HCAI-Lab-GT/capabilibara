# Website update plan: methodology-first, numbers-out, multi-model-ready

Status: executed 2026-07-29 on branch `glenn/modest-raman-09a43d`
(commits 195261b..3957562). Supersedes the results-forward content decisions
in `website-redesign-design.md` (three-tier findings with numbers,
signature-bin feature, metric-led metrics band) while keeping its structural
wins (shared data module, pending-model slots, single-file stack).

Maintainer decisions during execution: the metrics band is COMMENTED OUT in
place (not cut) with a restore-when-final note; the model roster is the
seven-entry list OLMo3 Base 7B (primary), OLMo3 Instruct 7B, Marin 8B,
DCLM 7B, Comma 7B, plus planned OLMo3 32B and Marin 32B.

Source: Capabilibara@COLM meeting, 2026-07-28
(<https://fathom.video/calls/762069268>, esp. 29:00–33:30). Note: the FinTDA
call on 2026-07-23 (<https://fathom.video/calls/757067300>) contains no website
discussion — the directives below all come from the 7/28 call.

## Directives from the meeting (authoritative)

1. **Strip results and numbers.** "No one cares about the numbers. The numbers
   are pointless." Numbers change frequently (everything is being rerun; the
   published ARC-Challenge numbers were computed on a subset; more models are
   coming), and most visitors can't interpret them.
2. **The site's job is methodology storytelling.** Explain training-data
   attribution and, explicitly, "what is an influence function" — the site
   never explains it today.
3. **No bin-level claims.** The 576 bins don't hold enough documents per bin;
   reviewers dinged the Literature × Customer Support story. Claims live at
   topic (24) or format (24) level, or higher.
4. **The durable claim is lexical and model-agnostic.** Affective, informal,
   linguistically rich text supports social reasoning; math/STEM draws on
   different text. Frame as a property tested across models, not OLMo charts.
5. **Multi-model future.** OLMo3 base + instruct, Comma, DCLM, Marin incoming
   (4–5 variants). Site must accommodate without rework.
6. **De-emphasize WebOrganizer.** "WebOrganizer isn't even the feature." The
   story is: you break up the corpus, you label it.
7. **Readability.** "A 12th grader should be able to read it and get it." Fun
   little visuals that define the message are the vehicle; repurpose the
   existing animations for methodology rather than results.
8. **Scope.** Lower priority than the camera-ready paper: incremental edits to
   the existing single-file site, not a rebuild. Stack stays static HTML +
   vanilla JS.

## Section-by-section disposition

| Section | Verdict | What replaces it |
|---|---|---|
| Head metadata | Rewrite | Model-agnostic, method-first descriptions; add "influence functions" keyword; fix canonical/og:url to the live `/social-data-attribution/` path; add SciFM workshop to venue metadata (confirm exact workshop name first) |
| Hero | Keep copy, cut teaser | Delete the `hero-static-fig` block — the overview PNG headlines bin-level numbers on its face. Hero becomes title/authors/links; venue line gains "· SciFM Workshop". The animated pipeline in Method is the visual anchor |
| Contribution band | Rewrite | The durable claim, 12th-grade register: "Where does a language model's social reasoning come from? We trace skills back to training data. Affective, informal, conversational text supports social reasoning; math and STEM reasoning draw on different text. Deleting the flagged data weakens the matching skill more than deleting random data." |
| Metrics band | Rewrite (or cut) | Structural facts only: "1 open corpus · 24 topics × 24 formats · 4 contrastive benchmarks · 5 model runs underway". Drop "+1.60 pp (p ≈ 10^-5)" and "5.68M documents". Cutting the band entirely is an acceptable fallback |
| Abstract | Rewrite | Plain-language overview, no `576 × 4` matrix framing, no WebOrganizer by name ("we label every document by topic and format"), scoped as "measured first in OLMo3, with more open-data models underway" |
| Method | Rewrite/extend | Keep pipeline walkthrough + taxonomy + 2×2 rows. Insert the influence-function explainer (below). WebOrganizer demoted to a parenthetical/footnote. Replace "Literature × Customer Support" illustrative labels with neutral combos (e.g. "Science & Tech × Tutorial"). Rework `figure1.js` scenes 3–4 to illustrative values; replace the numeric fallback image with a schematic poster (via the `?fig1stage=2&fig1end` screenshot aid) |
| Data | Rewrite (slim) | Keep benchmark identities and the fully-open-corpus story. Drop `~37K GPU hours`, `27.8%`, `>100,000x`, exact doc counts → qualitative ("billions of unique documents; a stratified sample of several million; heavily skewed toward a few big topic-format groups"), scoped "for the OLMo3 deep-dive". Query counts (10,000 SocialIQA etc.) may stay — dataset facts — but trim if in doubt |
| Findings | Rewrite | 4 reframed findings at topic/format/lexical level (below); all `finding-evidence` numeric `<dl>` blocks deleted |
| Result-lab `#results` | Cut | Delete the section and its nav item. Signature-bin bar viz and unlearning bar viz die with it. The `cluster-viz` word-cloud (no numbers) migrates into the new lexical finding. Delete the four evidence-grid figure files (png+webp) so stale numbers can't leak via direct links |
| Models | Rewrite | Keep H2 "Does the pattern hold beyond OLMo3?" Table becomes a qualitative roster (Model / Corpus / Status / "pattern observed?" with Observed vs In progress), no numbers. Roster: OLMo3-7B Base, OLMo3-7B Instruct, Comma 7B, DCLM 7B, Marin (confirm exact names/corpora with the team before commit; never fabricate a corpus name) |
| Limitations | Keep, light edit | Drop "5.68M-document/1.26B" precision; update the multi-model bullet to the full incoming roster; keep the demographic-claims disclaimer verbatim |
| Resources | Keep | Only touch phrasing if needed — "bin-level influence matrix" is fine as an artifact description (data shape, not a claim) |
| BibTeX | Keep | Revisit only when the camera-ready BibTeX changes |
| Footer | Keep | Optionally add SciFM to the venue line |

## Numbers ledger: every removal and its qualitative replacement

| Number (location) | Action | Qualitative replacement |
|---|---|---|
| +1.60 pp, p ≈ 10^-5 (metrics band, finding 04, unlearning viz, models table, fig1 caption 4) | Remove everywhere | "Forgetting the flagged data damages social reasoning more than forgetting random comparable data" |
| z = +16.00 / −7.31 / −1.92 / −5.75 (finding 02, signature-bin viz, models table, fig1 scene 3) | Remove everywhere | Nothing — bin-level claims exit the site entirely |
| r = 0.06/0.22/0.09, r = 0.76–0.86 (finding 01, models table) | Remove | "Social reasoning's data profile looks unlike the other three tasks', which resemble each other" |
| max Δz 0.91 vs 0.63 + topic loadings (finding 03) | Remove numbers, keep topic names | "The gap is wider for the reasoning pair than the knowledge pair"; topic names are permitted altitude |
| Correctness-differential +13.11/+6.0/−2.19/−2.17 (finding 02 secondary) | Remove | Nothing — bin-level |
| 245 words/doc, dialogue z 2.38, 10/10 split (lexical living figure) | Remove | "The text that supports social reasoning comes in two flavors: conversational (dialogue, feelings, 'you' and 'I') and expository (guides, manuals) — both affectively and linguistically rich" |
| Unlearning per-benchmark p-values table | Remove | One sentence: "significant for social reasoning, weak or absent for the others" |
| ~37K GPU hours, 27.8%, >100,000x, 1.26B, 1.10B/87%, 5.68M, 10.5B tokens (data section, taxonomy inline-facts) | Remove or round to words | "billions of documents", "a stratified sample of millions", "a few large groups dominate" |
| 576 bins / 24 × 24 (throughout) | **Keep** | Structural method fact, not a result |
| `fig-overview-revised.png/.webp` (hero + method fallback) | Remove from page | Animated pipeline + new schematic poster fallback |
| 4 evidence-grid figures (`fig_rq1_format_concentration`, `fig_influence_signed_2x2`, `unlearning_paired`, `fig_lex_cluster_composition`) | Remove + delete files | None — results-bearing and being rerun |

## The "What is an influence function?" explainer

**Placement:** in `#method`, as a new `concept-row` with `id="influence"`
immediately after the section heading and before the pipeline walkthrough. Add
a nav item "Influence" between Method and Data.

**Content outline (12th-grade register, ~150 words):**

1. Hook: "The model got this question right. Which parts of its training data
   taught it that?"
2. Definition: an influence function estimates how the model's answer would
   change if a given training document had been left out — without retraining.
3. Mechanism, one metaphor deep: every document gives the model a small nudge
   in some direction during training; a benchmark question also defines a
   direction ("what would make me better at this?"). Documents whose nudge
   points the same way have positive influence; opposed nudges are negative.
4. The aggregation move: one document's score is noisy. Group documents by
   topic and format and average, and the noise becomes a readable map. ("You
   break up the corpus, you label it" — taxonomy tool unnamed.)
5. Honesty beat: these are estimates, not proofs — which is exactly why the
   pipeline ends with the unlearning check.

**Visuals to repurpose:**

- New small inline SVG (`data-concept="influence"`): three mini-panels —
  question arrow, document arrow aligned ("supports"), document arrow opposed
  ("works against"). Reuse existing `concept-row`/`taxonomy-viz` CSS patterns
  and the reveal machinery in `main.js` `initScrollAnimations`.
- `figure1.js` `sceneAttribution` (probe chips, beams, signed color wash) is
  the animated companion; rewrite its caption pedagogically and cross-reference
  it ("watch stage 2 below").
- `figure1.js` captions: rewrite all four as methodology teaching text;
  caption 1 drops "WebOrganizer" for "labeled by topic and format"; caption 3
  becomes "aggregation turns noisy per-document scores into a map (illustrative
  values)"; caption 4 becomes the qualitative unlearning logic.

## Findings reframe (topic / format / lexical altitude)

Four findings, same `finding-row` markup, numeric evidence `<dl>`s deleted
(keep `.finding-evidence` as a one-sentence "how we measured it" line, or drop):

- **01 — Different skills, different data.** Social reasoning draws on
  different parts of the training corpus than STEM reasoning and factual-recall
  tasks. Measured first in OLMo3; framed as a property being tested across
  models.
- **02 — What social-reasoning text looks like** (replaces the signature-bin
  finding). The durable lexical claim: affective, informal, linguistically rich
  text — dialogue, first- and second-person, feelings and intentions —
  supports social reasoning; math/STEM draws on structured, technical,
  expository text. The `cluster-viz` word-cloud moves here as the visual.
  Format-level, never a named bin.
- **03 — Reasoning separates more than knowledge.** Keep, numbers stripped: the
  social-vs-STEM data split is wider for reasoning tasks than for knowledge
  tasks in the same domains.
- **04 — The causal check.** Qualitative unlearning story: make the model
  forget the flagged data, compare with forgetting random comparable data; the
  matched skill drops more. "Correlation to cause" survives at corpus-region
  level, with no bin named and no effect sizes.

## Concrete `socialtda-data.js` edits

- **Remove:** `SELECTED_BINS_PANEL`, `SIGNATURE_BIN`, `UNLEARNING`, `LEXICAL`
  (numeric result exports; every consumer is cut or made illustrative). The
  cluster-viz words stay hard-coded in HTML.
- **Keep:** `PROJECT`, `TAXONOMY`, `BENCHMARKS` (the 2×2 design is
  methodology), `INFLUENCE_METRIC` (feeds explainer copy), `SOURCE` (update its
  comment: "structural/method facts only; quantitative results intentionally
  excluded from the site").
- **Trim:** `WORKING_SET` — drop or stop consuming exact `documents: 5678621`.
- **Reshape `MODELS`:** roster-only entries `{ key, name, corpus, status }`;
  delete `socialiqaCorrelation`, `signatureBin`, `socialiqaUnlearning`. Add
  `olmo3_instruct`, `marin` entries (status "pending"; confirm names/corpora
  before commit). Preserves the fill-the-object extension contract.
- **`main.js`:** delete `initClaimBars`, `fmtZ`/`fmtPp`/`fmtP`, and the
  `SIGNATURE_BIN`/`UNLEARNING` imports; drop `[data-bars]` handling from
  `initScrollAnimations` and add the new influence-viz selector; simplify
  `initModelsTable` to render the roster + qualitative status cells.
- **`figure1.js`:** remove result imports; scene 3 renders clearly-labeled
  illustrative values; scene 4 becomes a schematic targeted-vs-random
  comparison with no real deltas.

## Work order (small sequential commits, site working after each)

Ordering note: `initClaimBars` and the reveal code no-op safely when their HTML
is absent, but `initModelsTable` overwrites static HTML — so the models HTML
and `main.js` must change in the same commit.

1. **Head + hero + bands.** Meta/OG/JSON-LD rewrite, canonical URL fix, SciFM
   venue line, delete hero teaser figure, rewrite contribution + metrics bands.
   HTML-only.
2. **Findings reframe.** Rewrite the 4 findings, delete evidence `<dl>`s and
   finding-02 bin content, move `cluster-viz` markup from `#results` into
   finding 02.
3. **Cut `#results`.** Delete the section, its nav link, and both bar-viz
   articles; delete the four evidence figure files (8 files, png+webp); remove
   `initClaimBars` + `[data-bars]` from `main.js` in the same commit.
4. **Method + explainer.** Insert the influence-function concept-row and
   mini-SVG, add nav item, de-WebOrganizer the abstract/method/data copy,
   neutralize taxonomy labels, slim the Data section.
5. **Pipeline animation rework.** `figure1.js` scenes 3–4 illustrative +
   caption rewrite; generate the schematic poster and swap the `.fig1-fallback`
   image; remove `fig-overview-revised.*` from the page (and repo if unused).
6. **Models roster.** Reshape `MODELS`, simplify `initModelsTable`, rewrite the
   static fallback table and intro/note, update the limitations multi-model
   bullet. Gate on team confirmation of the model list.
7. **Data-module cleanup.** Delete orphaned exports, update header comments.
8. **Verification pass:**
   - Serve locally (`python3 -m http.server` from `public/`); load with JS on,
     JS off (`no-js` fallbacks), and `prefers-reduced-motion`; console clean.
   - Anchor audit: every `href="#..."` has a matching `id`; no `#results`
     references remain.
   - Leftover-claims grep over `public/`:
     `grep -rnE 'Customer Support|z = |r = |r ≤|p ≈|p_BH|pp\b|16\.00|7\.31|1\.60|0\.22|0\.86|0\.91|37K|5\.68M|1\.26B|signature bin|WebOrganizer'`
     — expected hits only: "bin-level influence matrix" in Resources and (if
     kept) WebOrganizer's single footnote mention.
   - Mobile check: burger nav, models table horizontal scroll, fig1 controls at
     375px.

## Out-of-scope flags for the team

- `press/bluf.md` and `press/tweet-thread.md` still carry the old numbers and
  the signature-bin story; sync after the site lands. (2026-08-22: the press
  folder moved out of this repo into the private research repo's `press/`,
  where the numbers were re-synced to the arXiv page.)
- The on-page abstract should be re-synced when the camera-ready lands.
- `og-card.png` tagline mentions Dolma3 (optional regeneration; no numbers on
  it).
- URL inconsistency fixed in step 1: canonical/OG pointed at
  `eilab.gatech.edu/capabilibara/` but the live site is
  `eilab.gatech.edu/social-data-attribution/`.
