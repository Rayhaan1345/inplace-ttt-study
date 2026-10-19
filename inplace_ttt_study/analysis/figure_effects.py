"""Decomposition figure: TTT minus control = update effect + training effect, per length and seed.

Reads the per-instance CSV from collect.py, computes the paired differences for each seed and the
per-instance average over seeds (the estimate the decision rule uses), with 95% bootstrap intervals, and
plots the three quantities by context length. 64k appears twice: plain RoPE and YaRN x2.

    python -m inplace_ttt_study.analysis.figure_effects runs/results/ruler_instances.csv --out docs/figures/effects.png \\
        [--exclude-tasks niah_single_2 niah_multikey_1]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

SETTINGS = [("4k", ""), ("8k", ""), ("16k", ""), ("32k", ""), ("64k", ""), ("64k", "_yarn2")]
LABELS = ["4k", "8k", "16k", "32k", "64k", "64k\nYaRN"]


def load(path, exclude):
    data = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["task"] not in exclude:
                data[(r["model"], r["length"])][r["task"]][int(r["idx"])] = float(r["score"])
    return data


def paired(data, a_models, b_models, L, rng, boots=2000):
    """Mean over tasks of the per-instance difference between the average of a_models and of b_models."""
    def per_instance(models):
        if any((m, L) not in data for m in models):
            return None
        tasks = set.intersection(*(set(data[(m, L)]) for m in models))
        return {t: {i: np.mean([data[(m, L)][t][i] for m in models])
                    for i in set.intersection(*(set(data[(m, L)][t]) for m in models))} for t in tasks}
    ga, gb = per_instance(a_models), per_instance(b_models)
    if ga is None or gb is None:
        return None
    diffs = [np.array([ga[t][i] - gb[t][i] for i in sorted(set(ga[t]) & set(gb[t]))]) for t in sorted(set(ga) & set(gb))]
    point = np.mean([d.mean() for d in diffs])
    draws = np.array([np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diffs]) for _ in range(boots)])
    return point, np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-tasks", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)
    data = load(args.csv, set(args.exclude_tasks))
    arms = {"seed 1": (["ttt_on_s1"], ["ttt_on_s1_frozen"], ["ttt_off_s1"]),
            "seed 2": (["ttt_on_s2"], ["ttt_on_s2_frozen"], ["ttt_off_s2"]),
            "both seeds": (["ttt_on_s1", "ttt_on_s2"], ["ttt_on_s1_frozen", "ttt_on_s2_frozen"], ["ttt_off_s1", "ttt_off_s2"])}
    effects = {"TTT - control": lambda on, fr, off, L: paired(data, on, off, L, rng),
               "update: on - frozen": lambda on, fr, off, L: paired(data, on, fr, L, rng),
               "training: frozen - control": lambda on, fr, off, L: paired(data, fr, off, L, rng)}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), dpi=150, sharey=True)
    styles = {"seed 1": dict(marker="o", ls="--", alpha=0.55), "seed 2": dict(marker="s", ls="--", alpha=0.55),
              "both seeds": dict(marker="D", ls="-", lw=2)}
    rows = []
    for ax, (name, fn) in zip(axes, effects.items()):
        for arm, (on, fr, off) in arms.items():
            pts = []
            for k, (L, suf) in enumerate(SETTINGS):
                r = fn([m + suf for m in on], [m + suf for m in fr], [m + suf for m in off], L)
                if r is not None:
                    pts.append((k, *r)); rows.append((name, arm, LABELS[k].replace("\n", " "), *r))
            if pts:
                x, y, lo, hi = zip(*pts)
                ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], capsize=2, ms=4, label=arm, **styles[arm])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_title(name, fontsize=10); ax.grid(alpha=0.3)
    axes[0].set_ylabel("RULER points (mean over tasks)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(args.out)
    for r in rows:
        print(f"{r[0]:28s} {r[1]:11s} {r[2]:9s} {r[3]:+.1f} [{r[4]:+.1f}, {r[5]:+.1f}]")


if __name__ == "__main__":
    main()
