#!/usr/bin/env python3
"""Recompute selectivity from the standardized OLMES gamma matrix (gamma_olmes_tidy.csv),
with corrected config-matched baselines, plus the ToM extension (off-target + target).

Produces:
  - primary_selectivity_olmes.csv : corrected 2x2 + head-to-head (replaces Tables 22-24 numbers)
  - tom_offtarget_selectivity.csv : primary-target selectivity with ToM probes added as collateral
  - tom_target_selectivity.csv    : selectivity of targeting each ToM probe (n10b checkpoints)
Robust to partial data: computes whatever cells are present.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd

HOME = Path.home()
TIDY = HOME / "scratch/n16_selectivity/results/gamma_olmes_tidy.csv"
ZS = HOME / "dev/data-attribution/artifacts/zscored_bin_scores/aggregated"
OUTDIR = HOME / "scratch/n16_selectivity/results"

PRIMARY = ["socialiqa", "mmlu_social_science", "mmlu_stem", "arc_challenge"]
DISPLAY = {
    "socialiqa": "SocialIQA",
    "mmlu_social_science": "MMLU-SS",
    "mmlu_stem": "MMLU-STEM",
    "arc_challenge": "ARC-C",
}


def load_gamma():
    return pd.read_csv(TIDY)


def gmap(df, condition):
    """Return {key: {eval_benchmark: gamma}} for a condition."""
    out = {}
    sub = df[df.condition == condition]
    for key, g in sub.groupby("target_or_topic"):
        out[key] = dict(zip(g.eval_benchmark, g.gamma))
    return out


def selectivity(damage: dict, target: str, off_targets) -> float:
    t = abs(damage.get(target, 0.0))
    others = [abs(damage[b]) for b in off_targets if b in damage]
    if not others:
        return float("nan")
    m = float(np.mean(others))
    return t / m if m > 0 else (float("inf") if t > 0 else float("nan"))


def top1_topic(bench):
    z = pd.read_csv(ZS / f"zscored_{bench}.csv")
    return z.loc[z.zscore.idxmax(), "topic_label"]


def main():
    df = load_gamma()
    present = set(df.condition.unique())
    print(f"conditions present: {sorted(present)}")
    print(f"rows: {len(df)}, checkpoints: {df.target_or_topic.nunique()}")

    expA = gmap(df, "expA")  # keys: "<topic>__<targetbench>"
    n10b = gmap(df, "n10b_target")  # keys: "<probe>__<tag>_<topic>"

    # ---- ToM as off-target: when bin-targeting a primary benchmark, is ToM spared? ----
    tom_cols = sorted(set(df[df.eval_family == "tom"].eval_benchmark))
    print(f"\nToM probes available as collateral columns: {len(tom_cols)}")
    rows = []
    for target in PRIMARY:
        try:
            kstar = top1_topic(target)
        except Exception:
            continue
        bin_d = expA.get(f"{kstar}__{target}", {})
        if not bin_d:
            continue
        tom_collateral = {p: bin_d[p] for p in tom_cols if p in bin_d}
        if not tom_collateral:
            continue
        prim_off = [abs(bin_d[b]) for b in PRIMARY if b != target and b in bin_d]
        rows.append(
            {
                "target": DISPLAY[target],
                "top1_topic": kstar,
                "gamma_on_target": bin_d.get(target, float("nan")),
                "mean_abs_primary_collateral": float(np.mean(prim_off))
                if prim_off
                else float("nan"),
                "mean_abs_tom_collateral": float(
                    np.mean([abs(v) for v in tom_collateral.values()])
                ),
                "max_abs_tom_collateral": float(
                    max(abs(v) for v in tom_collateral.values())
                ),
                "n_tom_probes": len(tom_collateral),
                "tom_spared_vs_target": (
                    abs(bin_d.get(target, 0.0))
                    / (np.mean([abs(v) for v in tom_collateral.values()]) + 1e-9)
                ),
            }
        )
    off = pd.DataFrame(rows)
    off.to_csv(OUTDIR / "tom_offtarget_selectivity.csv", index=False)
    print(
        "\n=== ToM as OFF-TARGET: bin-targeting a primary benchmark — is held-out ToM spared? ==="
    )
    if not off.empty:
        print(off.to_string(index=False))

    # ---- ToM as target: collateral on primary when targeting each ToM probe (n10b top-1) ----
    rows = []
    for key, dmg in n10b.items():
        m = re.match(r"(.+?)__(top\d|null)_(.+)", key)
        if not m:
            continue
        probe, tag, topic = m.group(1), m.group(2), m.group(3)
        if tag != "top1":
            continue
        prim_coll = [abs(dmg[b]) for b in PRIMARY if b in dmg]
        rows.append(
            {
                "probe": probe,
                "top1_topic": topic,
                "gamma_on_primary_mean_abs": float(np.mean(prim_coll))
                if prim_coll
                else float("nan"),
                "gamma_on_primary_max_abs": float(max(prim_coll))
                if prim_coll
                else float("nan"),
                "n_primary": len(prim_coll),
            }
        )
    tgt = pd.DataFrame(rows)
    tgt.to_csv(OUTDIR / "tom_target_selectivity.csv", index=False)
    print(
        "\n=== ToM as TARGET: primary-benchmark collateral when targeting a ToM probe (top-1) ==="
    )
    if not tgt.empty:
        print(tgt.to_string(index=False))
    print(f"\nWrote outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
