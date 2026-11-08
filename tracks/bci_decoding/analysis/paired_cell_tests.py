"""Paired per-cell tests between every evaluated configuration.

Reads Benchopt result parquets whose objective recorded per-cell balanced
accuracy (``objective_cell/<stream>_<context>`` columns) and compares
configurations on the *same* cells:

- vs chance (1/3): one-sample, per configuration;
- vs the best configuration: paired, per configuration;
- all pairs among the best configuration of each model family.

Each comparison reports the mean per-cell difference, an exact paired
sign-flip permutation test (all 2^n sign patterns; primary test), the
Wilcoxon signed-rank test and a paired t-test, plus Holm-adjusted
permutation p-values within each family of comparisons.

Usage::

    python tracks/bci_decoding/analysis/paired_cell_tests.py \
        tracks/bci_decoding/outputs/percell_*.parquet --out-dir <dir>
"""

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load(paths):
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    cell_cols = sorted(c for c in df.columns
                       if c.startswith("objective_cell/"))
    if not cell_cols:
        raise SystemExit("No per-cell columns found: rerun inference with "
                         "the objective that records cell/<id> scores.")
    df = df.drop_duplicates("solver_name", keep="last")
    scores = df.set_index("solver_name")[cell_cols]
    scores.columns = [c.removeprefix("objective_cell/") for c in cell_cols]
    if scores.isna().any().any():
        raise SystemExit("Some runs are missing cells; cannot pair them.")
    return scores, df.set_index("solver_name")[
        "objective_balanced_accuracy_cell_mean"]


def label(solver_name):
    """``EEGPT[n_shot=0,strategy=probe]`` -> ``EEGPT probe all``."""
    m = re.match(r"([^\[]+)(?:\[(.*)\])?", solver_name)
    name, params = m.group(1), dict(
        kv.split("=", 1) for kv in (m.group(2) or "").split(",") if kv)
    if name == "Scratch":
        return f"{params['model']} scratch all"
    if name == "TangentSpace-Ridge":
        return "TangentSpace-Ridge scratch all"
    if name.startswith("Ensemble"):
        return f"{name} softvote all"
    if "strategy" not in params:
        return f"{name} baseline"
    shot = params.get("n_shot", "0")
    return f"{name} {params['strategy']} {'all' if shot == '0' else shot}"


def family(lbl):
    return lbl.split(" ")[0]


def sign_flip_p(diff):
    """Exact two-sided paired permutation p-value for mean(diff) == 0."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    signs = np.array(list(itertools.product([1, -1], repeat=n)))
    null = (signs * diff).mean(axis=1)
    observed = abs(diff.mean())
    return float(np.mean(np.abs(null) >= observed - 1e-12))


def compare(a, b):
    diff = np.asarray(a) - np.asarray(b)
    if np.allclose(diff, 0):
        return dict(mean_diff=0.0, p_perm=1.0, p_wilcoxon=1.0, p_ttest=1.0,
                    n_pos=0, n_neg=0)
    nonzero = diff[np.abs(diff) > 1e-12]
    p_w = (stats.wilcoxon(nonzero, method="exact").pvalue
           if len(nonzero) else 1.0)
    return dict(
        mean_diff=float(diff.mean()),
        p_perm=sign_flip_p(diff),
        p_wilcoxon=float(p_w),
        p_ttest=float(stats.ttest_1samp(diff, 0.0).pvalue),
        n_pos=int((diff > 1e-12).sum()),
        n_neg=int((diff < -1e-12).sum()),
    )


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("parquets", nargs="+")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scores, means = load(args.parquets)
    scores.index = [label(s) for s in scores.index]
    means.index = [label(s) for s in means.index]
    cells = list(scores.columns)
    print(f"{len(scores)} configurations x {len(cells)} cells: {cells}")
    scores.to_csv(out / "per_cell_scores.csv")

    chance = 1 / 3
    rows = []
    for name, row in scores.iterrows():
        rows.append(dict(config=name, mean=row.mean(),
                         **compare(row.values, np.full(len(row), chance))))
    vs_chance = pd.DataFrame(rows)
    vs_chance["p_perm_holm"] = holm(vs_chance["p_perm"])
    vs_chance = vs_chance.sort_values("mean", ascending=False)
    vs_chance.to_csv(out / "vs_chance.csv", index=False)

    best = vs_chance.iloc[0]["config"]
    rows = []
    for name, row in scores.iterrows():
        if name == best:
            continue
        rows.append(dict(config=name, mean=row.mean(),
                         **compare(scores.loc[best].values, row.values)))
    vs_best = pd.DataFrame(rows)
    vs_best["p_perm_holm"] = holm(vs_best["p_perm"])
    vs_best = vs_best.sort_values("mean", ascending=False)
    vs_best.insert(0, "best", best)
    vs_best.to_csv(out / "vs_best.csv", index=False)

    tops = (vs_chance.assign(family=vs_chance["config"].map(family))
            .sort_values("mean", ascending=False)
            .drop_duplicates("family")["config"].tolist())
    rows = []
    for a, b in itertools.combinations(tops, 2):
        rows.append(dict(a=a, b=b, mean_a=scores.loc[a].mean(),
                         mean_b=scores.loc[b].mean(),
                         **compare(scores.loc[a].values,
                                   scores.loc[b].values)))
    pairwise = pd.DataFrame(rows)
    pairwise["p_perm_holm"] = holm(pairwise["p_perm"])
    pairwise.to_csv(out / "pairwise_best_per_family.csv", index=False)

    matrix = pd.DataFrame(np.nan, index=tops, columns=tops)
    for _, r in pairwise.iterrows():
        matrix.loc[r["a"], r["b"]] = r["p_perm"]
        matrix.loc[r["b"], r["a"]] = r["p_perm"]
    matrix.to_csv(out / "pairwise_p_perm_matrix.csv")

    pd.set_option("display.width", 200)
    print("\nvs chance (top 15):")
    print(vs_chance.head(15).round(4).to_string(index=False))
    print(f"\nvs best ({best}), top 15:")
    print(vs_best.head(15).round(4).to_string(index=False))
    print(f"\nbest per family: {tops}")
    sig = pairwise[pairwise["p_perm"] < 0.05]
    print(f"\npairwise among {len(tops)} family bests: {len(pairwise)} "
          f"pairs, {len(sig)} with uncorrected p_perm < 0.05, "
          f"{(pairwise['p_perm_holm'] < 0.05).sum()} after Holm")
    print(f"\nwrote CSVs to {out}")


if __name__ == "__main__":
    main()
