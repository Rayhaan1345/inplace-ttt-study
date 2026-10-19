"""Tables and a figure from the per-instance CSV written by collect.py.

* Per model and length: the mean over tasks (the paper's RULER convention) with a 95% bootstrap
  interval that resamples instances within each task.
* Per task: accuracy per model.
* Paired differences between two models on the same instances (`--pairs a:b`), per length and
  per task, with paired bootstrap intervals. Either side may be several models joined with `+`,
  whose scores are averaged per instance first: `ttt_on_s1+ttt_on_s2:ttt_off_s1+ttt_off_s2` is the
  seed-averaged paired difference of the decision rule in docs/plan.md.

    python -m inplace_ttt_study.analysis.report runs/results/ruler_instances.csv --out docs/results \\
        --pairs ttt_on_s1:ttt_off_s1 ttt_on_s1:ttt_on_s1_frozen
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

LENGTHS = ["4k", "8k", "16k", "32k", "64k", "128k"]


def load(path):
    """-> {(model, length): {task: {idx: score}}}"""
    data = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for r in csv.DictReader(f):
            data[(r["model"], r["length"])][r["task"]][int(r["idx"])] = float(r["score"])
    return data


def aggregate(tasks, rng, boots=2000, other=None):
    """Mean over tasks of per-task means, with bootstrap draws. If `other` is given, both models are
    resampled with the same instance draws and the paired difference is returned."""
    names = sorted(tasks)
    arrays, arrays_other = [], []
    for t in names:
        idx = sorted(tasks[t]) if other is None else sorted(set(tasks[t]) & set(other[t]))
        arrays.append(np.array([tasks[t][i] for i in idx]))
        if other is not None:
            arrays_other.append(np.array([other[t][i] for i in idx]))
    point = np.mean([a.mean() for a in arrays])
    if other is not None:
        point -= np.mean([a.mean() for a in arrays_other])
    draws = np.empty(boots)
    for b in range(boots):
        means = []
        for k, a in enumerate(arrays):
            sel = rng.integers(0, len(a), len(a))
            means.append(a[sel].mean() - (arrays_other[k][sel].mean() if other is not None else 0.0))
        draws[b] = np.mean(means)
    return point, np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def group(data, spec, L):
    """Per-instance mean score of the '+'-joined models in `spec` at length L; None if one is missing."""
    members = spec.split("+")
    if any((m, L) not in data for m in members):
        return None
    tasks = set.intersection(*(set(data[(m, L)]) for m in members))
    return {t: {i: float(np.mean([data[(m, L)][t][i] for m in members]))
                for i in set.intersection(*(set(data[(m, L)][t]) for m in members))} for t in tasks}


def fmt(point, lo, hi):
    return f"{point:.1f} [{lo:.1f}, {hi:.1f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="model_a:model_b paired differences")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-tasks", nargs="*", default=[], help="tasks left out (sensitivity analysis)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    data = load(args.csv)
    for key in data:
        for t in args.exclude_tasks:
            data[key].pop(t, None)
    models = sorted({m for m, _ in data})
    lengths = [L for L in LENGTHS if any(L == l for _, l in data)]
    note = f" Excluded tasks: {', '.join(args.exclude_tasks)}." if args.exclude_tasks else ""
    lines = ["# RULER results", "", "Mean over tasks, 95% bootstrap interval over instances." + note, "",
             "| model | " + " | ".join(lengths) + " |", "|---|" + "---|" * len(lengths)]
    summary = {}
    n_tasks = max(len(d) for d in data.values())  # a length counts as complete when all tasks are present
    for m in models:
        cells = []
        for L in lengths:
            if (m, L) in data and len(data[(m, L)]) == n_tasks:
                summary[(m, L)] = aggregate(data[(m, L)], rng)
                cells.append(fmt(*summary[(m, L)]))
            elif (m, L) in data:
                cells.append(f"partial ({len(data[(m, L)])}/{n_tasks} tasks)")
            else:
                cells.append("")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    for L in lengths:
        tasks = sorted({t for (m, l), d in data.items() if l == L for t in d})
        lines += ["", f"## Per task at {L}", "", "| task | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for t in tasks:
            cells = [f"{np.mean(list(data[(m, L)][t].values())):.1f}" if (m, L) in data and t in data[(m, L)] else ""
                     for m in models]
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
    for pair in args.pairs:
        a, b = pair.split(":")
        tasks_all = sorted({t for (m, l), d in data.items() if m in a.split("+") for t in d})
        lines += ["", f"## Paired difference {a} - {b}", "", "| length | mean over tasks | " + " | ".join(tasks_all) + " |",
                  "|---|---|" + "---|" * len(tasks_all)]
        for L in lengths:
            ga, gb = group(data, a, L), group(data, b, L)
            if ga is None or gb is None or len(ga) < n_tasks or len(gb) < n_tasks:
                continue
            cells = [fmt(*aggregate(ga, rng, other=gb))]
            for t in tasks_all:
                cells.append(fmt(*aggregate({t: ga[t]}, rng, other={t: gb[t]})) if t in ga and t in gb else "")
            lines.append(f"| {L} | " + " | ".join(cells) + " |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "length", "mean", "ci_lo", "ci_hi"])
        for (m, L), v in summary.items():
            w.writerow([m, L, *[round(x, 2) for x in v]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
        for m in models:
            pts = [(i, *summary[(m, L)]) for i, L in enumerate(lengths) if (m, L) in summary]
            if pts:
                x, y, lo, hi = zip(*pts)
                ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], marker="o", capsize=3, label=m)
        ax.set_xticks(range(len(lengths))); ax.set_xticklabels(lengths); ax.set_ylabel("RULER (mean of tasks)")
        ax.set_xlabel("context length"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "ruler_by_length.png"))
    except ImportError:
        pass
    print("\n".join(lines[:4 + len(models)]))


if __name__ == "__main__":
    main()
