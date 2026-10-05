#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
"""Paired-significance test for Finding 9 (influence-in-topic vs random-in-topic),
using the FAITHFUL 3-seed gamma matrix on all 24 WebOrganizer topics. Answers
R4 C2: "is the influence-vs-random difference statistically significant?"

Pairing: for each (topic, target_benchmark, seed) tuple, pair
  - expA's gamma on its target benchmark  (influence-targeted, in topic)
  - exp1's gamma on the same benchmark    (random, in same topic, same seed)
into a per-cell paired difference d = gamma_expA - gamma_exp1.

gamma = (baseline_acc - unlearned_acc), so POSITIVE gamma = damage. A POSITIVE
paired difference d > 0 means influence-targeted damages MORE on the intended
capability than random-in-topic does (supports Finding 9).

Per benchmark we run two tests:
  - Wilcoxon signed-rank (primary, non-parametric)        on pooled (topic x seed) cells
  - paired t-test on per-topic seed-mean differences      on per-topic means

Both with alternative='greater' (one-sided: expecting d > 0). Multiple-comparisons
across 4 benchmarks x 2 tests = 8 family-wise p-values, Benjamini-Hochberg adjusted.

Effect sizes:
  - median paired difference + 95% bootstrap CI of median (B=2000)
  - Cohen's d_z = mean(d) / std(d), the paired-t effect size

Output CSV columns:
  target, test, n_cells, n_topics, n_seeds, median_diff, ci95_low, ci95_high,
  statistic, p_uncorrected, p_bh, cohens_dz, n_neg, n_pos, n_zero
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HOME = Path.home()
TIDY = HOME / "scratch/n16_selectivity/results/faithful_gamma_tidy.csv"
OUT = HOME / "scratch/n16_selectivity/results/paired_significance.csv"
PRIMARY = ["socialiqa", "mmlu_social_science", "mmlu_stem", "arc_challenge"]


def benjamini_hochberg(pvals):
    """BH-adjust an array of p-values (Benjamini-Hochberg, 1995). Returns
    adjusted p in the original order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1.0)
    # Enforce monotonicity from the smallest p upward.
    ranked_monotone = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked_monotone)
    out[order] = np.minimum(ranked_monotone, 1.0)
    return out


def bootstrap_median_ci(x, b=2000, alpha=0.05, rng=None):
    """Bootstrap CI of the median. Returns (low, high)."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    samples = rng.choice(x, size=(b, len(x)), replace=True)
    medians = np.median(samples, axis=1)
    lo = float(np.quantile(medians, alpha / 2))
    hi = float(np.quantile(medians, 1 - alpha / 2))
    return (lo, hi)


def cohens_dz(d):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return np.nan
    return float(np.mean(d)) / sd


def build_paired_cells(df):
    """For each (topic, eval_benchmark=target, seed), join expA's gamma-on-target
    with exp1's gamma-on-the-same-benchmark. Returns DataFrame with columns:
      topic, target, seed, gamma_expA, gamma_exp1, d
    where target == eval_benchmark and d = gamma_expA - gamma_exp1.
    Positive d => influence-targeted damages more than random-in-topic (Finding 9)."""
    # expA: rows where target matches eval_benchmark — that's the "gamma on the
    # intended target" row. (expA rows on OTHER benchmarks are collateral, not
    # relevant for the Finding-9 paired contrast.)
    a = df[(df.condition == "expA") & (df.target == df.eval_benchmark)][
        ["topic", "target", "seed", "gamma"]
    ].rename(columns={"gamma": "gamma_expA"})

    # exp1: rows are per-(topic, seed) for each eval_benchmark. Rename
    # eval_benchmark -> target so the join key matches expA's target column.
    e = df[df.condition == "exp1"][["topic", "eval_benchmark", "seed", "gamma"]].rename(
        columns={"eval_benchmark": "target", "gamma": "gamma_exp1"}
    )

    paired = pd.merge(a, e, on=["topic", "target", "seed"], how="inner")
    paired["d"] = paired["gamma_expA"] - paired["gamma_exp1"]
    return paired


def analyze(paired):
    """Run Wilcoxon (per-cell pooled) and paired t-test (per-topic seed-mean)
    per benchmark. Collect p-values into a single family for BH adjustment."""
    rows = []
    p_uncorr = []
    rng = np.random.default_rng(20260601)

    for target in PRIMARY:
        sub = paired[paired.target == target].copy()
        n_cells = len(sub)
        n_seeds = sub.seed.nunique() if n_cells else 0
        n_topics = sub.topic.nunique() if n_cells else 0
        d_all = sub["d"].to_numpy()
        n_neg = int((d_all < 0).sum())
        n_pos = int((d_all > 0).sum())
        n_zero = int((d_all == 0).sum())
        med = float(np.median(d_all)) if n_cells else np.nan
        ci_lo, ci_hi = (
            bootstrap_median_ci(d_all, b=2000, rng=rng) if n_cells else (np.nan, np.nan)
        )

        # Wilcoxon signed-rank on pooled (topic x seed) cells, alternative='greater'
        # (testing d > 0 = influence-targeted damages MORE than random-in-topic).
        if n_cells >= 2 and not np.all(d_all == 0):
            try:
                w_stat, w_p = stats.wilcoxon(
                    d_all, alternative="greater", zero_method="wilcox"
                )
                w_stat = float(w_stat)
                w_p = float(w_p)
            except ValueError:
                w_stat, w_p = (np.nan, np.nan)
        else:
            w_stat, w_p = (np.nan, np.nan)
        dz_pooled = cohens_dz(d_all)

        rows.append(
            {
                "target": target,
                "test": "wilcoxon_pooled",
                "n_cells": n_cells,
                "n_topics": n_topics,
                "n_seeds": n_seeds,
                "median_diff": med,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "statistic": w_stat,
                "p_uncorrected": w_p,
                "p_bh": np.nan,  # filled after
                "cohens_dz": dz_pooled,
                "n_neg": n_neg,
                "n_pos": n_pos,
                "n_zero": n_zero,
            }
        )
        p_uncorr.append(w_p)

        # Paired t-test on per-topic seed-mean differences, alternative='greater'.
        per_topic = sub.groupby("topic")["d"].mean().to_numpy()
        n_topics_t = int(np.sum(~np.isnan(per_topic)))
        med_t = (
            float(np.median(per_topic[~np.isnan(per_topic)])) if n_topics_t else np.nan
        )
        ci_lo_t, ci_hi_t = (
            bootstrap_median_ci(per_topic, b=2000, rng=rng)
            if n_topics_t
            else (np.nan, np.nan)
        )
        if n_topics_t >= 2 and np.std(per_topic[~np.isnan(per_topic)], ddof=1) > 0:
            t_stat, t_p = stats.ttest_1samp(
                per_topic[~np.isnan(per_topic)], 0, alternative="greater"
            )
            t_stat = float(t_stat)
            t_p = float(t_p)
        else:
            t_stat, t_p = (np.nan, np.nan)
        dz_topic = cohens_dz(per_topic)
        n_neg_t = int(np.sum(per_topic < 0))
        n_pos_t = int(np.sum(per_topic > 0))
        n_zero_t = int(np.sum(per_topic == 0))

        rows.append(
            {
                "target": target,
                "test": "paired_t_topic_mean",
                "n_cells": n_topics_t,
                "n_topics": n_topics_t,
                "n_seeds": n_seeds,
                "median_diff": med_t,
                "ci95_low": ci_lo_t,
                "ci95_high": ci_hi_t,
                "statistic": t_stat,
                "p_uncorrected": t_p,
                "p_bh": np.nan,
                "cohens_dz": dz_topic,
                "n_neg": n_neg_t,
                "n_pos": n_pos_t,
                "n_zero": n_zero_t,
            }
        )
        p_uncorr.append(t_p)

    # BH-adjust across the family of 8 tests, ignoring NaNs.
    p_arr = np.array(p_uncorr, dtype=float)
    valid = ~np.isnan(p_arr)
    if valid.sum():
        adj = benjamini_hochberg(p_arr[valid])
        p_bh_full = np.full_like(p_arr, np.nan)
        p_bh_full[valid] = adj
    else:
        p_bh_full = p_arr
    for i, r in enumerate(rows):
        r["p_bh"] = p_bh_full[i]

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(TIDY)
    print(f"loaded {len(df)} rows from {TIDY}")
    print(f"conditions: {dict(df.condition.value_counts())}")
    print(f"seeds: {sorted(df.seed.unique())}")
    print(f"benchmarks: {sorted(df.eval_benchmark.unique())}")

    paired = build_paired_cells(df)
    print(f"\npaired (expA target=bench, exp1 same-bench) cells: {len(paired)}")
    print(paired.groupby("target").size().to_string())
    print("\nper-target topic coverage:")
    print(paired.groupby("target")["topic"].nunique().to_string())

    results = analyze(paired)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}\n")
    # Pretty-print summary.
    for target in PRIMARY:
        sub = results[results.target == target]
        if sub.empty:
            print(f"{target}: NO DATA")
            continue
        wil = sub[sub.test == "wilcoxon_pooled"].iloc[0]
        tt = sub[sub.test == "paired_t_topic_mean"].iloc[0]
        sig_w = (
            "***"
            if wil.p_bh < 0.001
            else "**"
            if wil.p_bh < 0.01
            else "*"
            if wil.p_bh < 0.05
            else "ns"
        )
        sig_t = (
            "***"
            if tt.p_bh < 0.001
            else "**"
            if tt.p_bh < 0.01
            else "*"
            if tt.p_bh < 0.05
            else "ns"
        )
        print(
            f"{target:>25s}  Wilcoxon  n_cells={int(wil.n_cells):3d} median_d={wil.median_diff:+.4f} "
            f"[{wil.ci95_low:+.4f},{wil.ci95_high:+.4f}]  p_BH={wil.p_bh:.4g} ({sig_w})  d_z={wil.cohens_dz:+.2f}"
        )
        print(
            f"{'':25s}  paired_t  n_topics={int(tt.n_cells):3d} median_d={tt.median_diff:+.4f} "
            f"[{tt.ci95_low:+.4f},{tt.ci95_high:+.4f}]  p_BH={tt.p_bh:.4g} ({sig_t})  d_z={tt.cohens_dz:+.2f}"
        )


if __name__ == "__main__":
    main()
