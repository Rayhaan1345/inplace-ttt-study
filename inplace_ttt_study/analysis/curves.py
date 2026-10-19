"""Training-curve figure from the metrics.jsonl files of several runs.

Plots loss and pre-clipping gradient norm per optimizer step (points) with a running mean
(line); same-seed arms consume identical batches, so their per-step losses are directly
comparable.

    python -m inplace_ttt_study.analysis.curves --out docs/results/curves.png ttt_on_seed1=runs/checkpoints/ttt_on_seed1 ...
"""
import argparse
import json
import os

import numpy as np


def read(run_dir):
    steps, loss, gnorm = [], [], []
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if "step" in r:
                steps.append(r["step"]); loss.append(r["loss"]); gnorm.append(r["grad_norm"])
    return np.array(steps), np.array(loss), np.array(gnorm)


def running_mean(x, k):
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="name=run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    for spec in args.runs:
        name, run_dir = spec.split("=", 1)
        steps, loss, gnorm = read(run_dir)
        k = min(args.window, len(steps))
        for ax, y in zip(axes, (loss, gnorm)):
            line, = ax.plot(steps[k - 1:], running_mean(y, k), label=name)
            ax.plot(steps, y, ".", ms=2, alpha=0.25, color=line.get_color())
    axes[0].set_ylabel("training loss"); axes[1].set_ylabel("gradient norm (before clipping)"); axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("optimizer step"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True); fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
