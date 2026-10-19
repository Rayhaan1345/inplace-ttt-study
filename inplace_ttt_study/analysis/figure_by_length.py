"""RULER mean over tasks by context length for the main arms of one seed, from report.py's summary.csv.

Plain-RoPE results are drawn as lines with bootstrap intervals; the 64k YaRN x2 results of the same
models are drawn as square markers.

    python -m inplace_ttt_study.analysis.figure_by_length runs/results/report/summary.csv --seed 1 --out docs/figures/ruler_seed1.png
"""
import argparse
import csv

LENGTHS = ["4k", "8k", "16k", "32k", "64k"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary_csv")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(args.summary_csv)))
    s = args.seed
    names = {"base": "base", f"ttt_off_s{s}": "control (CT only)", f"ttt_on_s{s}": "TTT-trained, update on",
             f"ttt_on_s{s}_frozen": "TTT-trained, update off"}
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    for m, label in names.items():
        pts = {r["length"]: (float(r["mean"]), float(r["ci_lo"]), float(r["ci_hi"])) for r in rows if r["model"] == m}
        x = [i for i, L in enumerate(LENGTHS) if L in pts]
        y = [pts[LENGTHS[i]][0] for i in x]
        err = [[pts[LENGTHS[i]][0] - pts[LENGTHS[i]][1] for i in x], [pts[LENGTHS[i]][2] - pts[LENGTHS[i]][0] for i in x]]
        line = ax.errorbar(x, y, yerr=err, marker="o", ms=4, capsize=2, label=label)
        yarn = {r["length"]: float(r["mean"]) for r in rows if r["model"] == m + "_yarn2"}
        if "64k" in yarn:
            ax.plot([LENGTHS.index("64k") + 0.25], [yarn["64k"]], marker="s", ms=5, ls="none", color=line[0].get_color())
    ax.plot([], [], marker="s", ms=5, ls="none", color="gray", label="64k with YaRN x2")
    ax.set_xticks(range(len(LENGTHS))); ax.set_xticklabels(LENGTHS)
    ax.set_xlabel("context length"); ax.set_ylabel("RULER, mean of 13 tasks")
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5); ax.set_title(f"Seed {s}, 500 instances per task", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
