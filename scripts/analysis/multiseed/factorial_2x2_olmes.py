"""2x2 factorial selectivity on the CORRECTED, config-matched OLMES gamma matrix.

Same metric + 2x2 contrast logic as factorial_2x2.py, but ingests gamma directly
from gamma_olmes_tidy.csv (base_acc and unlearned_acc both measured under :mc::olmes),
eliminating the old dual-CSV stitch + hardcoded BASELINE dict that mixed eval configs
(ARC baseline 0.892->0.8345, SocialIQA 0.8029->0.739, MMLU-SS 0.7508->0.7524).

Conditions (tidy `condition` / `target_or_topic`):
  expA  <topic>__<targetbench>   bin-targeted (influence-ranked, one topic)
  expC  <targetbench>            naive top-K (influence-ranked, corpus-wide)
  exp1  <topic>                  random-in-topic (target-agnostic)
  exp3  null_bin                 random-global (single condition)
"""

from pathlib import Path
import numpy as np
import pandas as pd

HOME = Path.home()
ROOT = HOME / "scratch" / "n16_selectivity"
TIDY = ROOT / "results" / "gamma_olmes_tidy.csv"
ZSCORED_DIR = (
    HOME
    / "dev"
    / "data-attribution"
    / "artifacts"
    / "zscored_bin_scores"
    / "aggregated"
)
OUT_DIR = ROOT / "results" / "robustness"

BENCHMARKS = ["socialiqa", "mmlu_social_science", "mmlu_stem", "arc_challenge"]
DISPLAY = {
    "socialiqa": "SocialIQA",
    "mmlu_social_science": "MMLU-SS",
    "mmlu_stem": "MMLU-STEM",
    "arc_challenge": "ARC-C",
}


def load_gamma_tidy():
    df = pd.read_csv(TIDY)
    return df[df.eval_family == "primary"].copy()


def cell_map(sub):
    """rows of one checkpoint -> {eval_benchmark: gamma}"""
    return dict(zip(sub.eval_benchmark, sub.gamma))


def build_expA(df):
    """gamma[target][topic] = {bench: gamma}"""
    out = {}
    sub = df[df.condition == "expA"]
    for key, g in sub.groupby("target_or_topic"):
        if "__" not in str(key):
            continue
        topic, target = str(key).rsplit("__", 1)
        out.setdefault(target, {})[topic] = cell_map(g)
    return out


def build_expC(df):
    out = {}
    for key, g in df[df.condition == "expC"].groupby("target_or_topic"):
        out[str(key)] = cell_map(g)
    return out


def build_exp1(df):
    out = {}
    for key, g in df[df.condition == "exp1"].groupby("target_or_topic"):
        out[str(key)] = cell_map(g)
    return out


def build_exp3(df):
    sub = df[df.condition == "exp3"]
    if sub.empty:
        return {}
    return cell_map(sub)


def selectivity_M1(damage, target):
    t = abs(damage.get(target, 0.0))
    others = [abs(damage[b]) for b in BENCHMARKS if b != target and b in damage]
    if not others:
        return float("nan")
    m = float(np.mean(others))
    return t / m if m > 0 else (float("inf") if t > 0 else float("nan"))


def top1_topic(z_all, target):
    z = z_all[target]
    return z.loc[z["zscore"].idxmax(), "topic_label"]


def main():
    df = load_gamma_tidy()
    z_all = {b: pd.read_csv(ZSCORED_DIR / f"zscored_{b}.csv") for b in BENCHMARKS}

    expA = build_expA(df)
    exp1 = build_exp1(df)
    expC = build_expC(df)
    exp3 = build_exp3(df)

    print("Data coverage check (CORRECTED OLMES gamma):")
    print(
        f"  expA targets: {list(expA.keys())}  (topics per target: {[len(v) for v in expA.values()]})"
    )
    print(f"  exp1 topics: {len(exp1)}  sample keys: {list(exp1.keys())[:3]}")
    print(f"  expC targets: {list(expC.keys())}")
    print(
        f"  exp3 gamma: { {b: round(exp3.get(b, float('nan')), 4) for b in BENCHMARKS} }"
    )

    rows = []
    print("\n" + "=" * 80)
    print("2x2 FACTORIAL SELECTIVITY (at top-1 topic k* per target) -- CORRECTED OLMES")
    print("=" * 80)
    for target in BENCHMARKS:
        kstar = top1_topic(z_all, target)
        selA = selectivity_M1(expA.get(target, {}).get(kstar, {}), target)
        sel1 = selectivity_M1(exp1.get(kstar, {}), target)
        selC = selectivity_M1(expC.get(target, {}), target)
        sel3 = selectivity_M1(exp3, target)
        influence_effect = (
            selA / sel1 if (sel1 and not np.isnan(sel1) and sel1 != 0) else float("nan")
        )
        topic_effect = (
            selA / selC if (selC and not np.isnan(selC) and selC != 0) else float("nan")
        )
        combined = (
            selA / sel3 if (sel3 and not np.isnan(sel3) and sel3 != 0) else float("nan")
        )
        rows.append(
            {
                "target": target,
                "display": DISPLAY[target],
                "top1_topic": kstar,
                "selA_influence_topic": selA,
                "selC_influence_global": selC,
                "sel1_random_topic": sel1,
                "sel3_random_global": sel3,
                "influence_effect_A_over_1": influence_effect,
                "topic_effect_A_over_C": topic_effect,
                "combined_A_over_3": combined,
            }
        )
        print(f"\n  Target: {DISPLAY[target]}  (top-1 topic k* = {kstar})")
        print(f"    influence-ranked | A: sel={selA:7.3f}     | C: sel={selC:7.3f}")
        print(f"    random           | 1: sel={sel1:7.3f}     | 3: sel={sel3:7.3f}")
        print(
            f"    -> influence (A/1) = {influence_effect:6.2f}   topic (A/C) = {topic_effect:6.2f}   combined (A/3) = {combined:6.2f}"
        )

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "factorial_2x2_olmes.csv", index=False)
    n = len(out)
    print("\n" + "=" * 80)
    print(
        f"  Influence ranking helps (A>1):  {(out['influence_effect_A_over_1'] > 1.0).sum()}/{n}"
    )
    print(
        f"  Topic filtering helps  (A>C):   {(out['topic_effect_A_over_C'] > 1.0).sum()}/{n}"
    )
    print(
        f"  Combined beats random  (A>3):   {(out['combined_A_over_3'] > 1.0).sum()}/{n}"
    )
    print(
        f"  Median influence (A/1): {out['influence_effect_A_over_1'].median():.2f}   "
        f"topic (A/C): {out['topic_effect_A_over_C'].median():.2f}   "
        f"combined (A/3): {out['combined_A_over_3'].median():.2f}"
    )
    print(f"\nWrote {OUT_DIR / 'factorial_2x2_olmes.csv'}")


if __name__ == "__main__":
    main()
