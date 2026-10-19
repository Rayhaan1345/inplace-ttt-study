"""Per-task LaTeX tables for the paper appendix, from the per-instance CSV written by collect.py.

Each table has one row per task plus a mean row with a 95% bootstrap interval (report.aggregate),
and one column per `model:label` spec at the given length. Missing or partial columns are left
blank.

    python -m inplace_ttt_study.analysis.latex_tables runs/results/ruler_instances.csv --length 32k \\
        --models base:Base ttt_off_s1:"Control s1" ttt_on_s1:"TTT s1" --out paper/tables/pertask_32k.tex
"""
import argparse

import numpy as np

from .report import aggregate, load

TASK_NAMES = {"niah_single_1": "NIAH single 1", "niah_single_2": "NIAH single 2", "niah_single_3": "NIAH single 3",
              "niah_multikey_1": "NIAH multikey 1", "niah_multikey_2": "NIAH multikey 2",
              "niah_multikey_3": "NIAH multikey 3", "niah_multiquery": "NIAH multiquery",
              "niah_multivalue": "NIAH multivalue", "vt": "Variable tracking", "cwe": "Common words",
              "fwe": "Frequent words", "qa_squad": "QA SQuAD", "qa_hotpotqa": "QA HotpotQA"}
ORDER = list(TASK_NAMES)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--length", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="model:label pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-tasks", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    data = load(args.csv)
    for key in data:
        for t in args.exclude_tasks:
            data[key].pop(t, None)
    specs = [m.split(":", 1) for m in args.models]
    n_tasks = len(ORDER) - len(args.exclude_tasks)
    cols = []
    for model, label in specs:
        d = data.get((model, args.length))
        complete = d is not None and len(d) == n_tasks
        cols.append((label, d, aggregate(d, rng) if complete else None))
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "Task & " + " & ".join(c[0] for c in cols) + " \\\\", "\\midrule"]
    for t in ORDER:
        if t in args.exclude_tasks:
            continue
        cells = [f"{np.mean(list(d[t].values())):.1f}" if d and t in d else "" for _, d, _ in cols]
        lines.append(TASK_NAMES[t] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\midrule", "Mean & " + " & ".join(f"{m:.1f} [{lo:.1f}, {hi:.1f}]" if s else "" for _, _, s in cols for m, lo, hi in [s or (0, 0, 0)]) + " \\\\",
              "\\bottomrule", "\\end{tabular}"]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{args.out}: {len(cols)} columns at {args.length}")


if __name__ == "__main__":
    main()
