"""Collect per-instance RULER scores from OpenCompass prediction files into one CSV.

OpenCompass's RULER evaluators report only task-level accuracy. For confidence intervals and paired
comparisons, the per-instance score is recomputed here with the same rule (case-insensitive
containment of each gold string in the prediction; the QA tasks count a hit if any gold string
matches, the other tasks average over gold strings) and checked against OpenCompass's own number
whenever its results file exists.

    python -m inplace_ttt_study.analysis.collect runs/results/ruler --out runs/results/ruler_instances.csv

Rows: tag, length, model, task, idx, score. `tag` and `length` come from the work-dir name
`<tag>_<length>`; the latest OpenCompass run inside each work dir is used.
"""
import argparse
import ast
import csv
import glob
import json
import os
import re
import sys


def instance_score(task, prediction, gold):
    if isinstance(gold, str):  # some tasks store the reference list as its string repr
        gold = ast.literal_eval(gold) if gold.startswith("[") else [gold]
    refs = [str(r) for r in gold]
    hits = [r.lower() in prediction.lower() for r in refs]
    return 100.0 * (max(hits) if task.startswith("qa") else sum(hits) / len(hits))


def collect(root):
    rows = []
    for work in sorted(glob.glob(os.path.join(root, "*_*"))):
        runs = sorted(glob.glob(os.path.join(work, "20*")))
        if not runs:
            continue
        tag, length = os.path.basename(work).rsplit("_", 1)
        run = runs[-1]
        for pred_file in sorted(glob.glob(os.path.join(run, "predictions", "*", "ruler_*.json"))):
            model = os.path.basename(os.path.dirname(pred_file))
            m = re.fullmatch(rf"ruler_(.+)_{length}(?:_(\d+))?\.json", os.path.basename(pred_file))
            if not m:
                continue
            task, part = m.group(1), int(m.group(2) or 0)
            with open(pred_file) as f:
                for i, entry in json.load(f).items():
                    rows.append(dict(tag=tag, length=length, model=model, task=task, idx=part * 10000 + int(i),
                                     score=instance_score(task, entry["prediction"], entry["gold"])))
        cross_check(rows, run, tag, length)
    return rows


def cross_check(rows, run, tag, length):
    """Compare each finished task's recomputed mean with OpenCompass's results file, if present."""
    for result_file in glob.glob(os.path.join(run, "results", "*", f"ruler_*_{length}.json")):
        model = os.path.basename(os.path.dirname(result_file))
        task = re.fullmatch(rf"ruler_(.+)_{length}\.json", os.path.basename(result_file)).group(1)
        ours = [r["score"] for r in rows if (r["tag"], r["length"], r["model"], r["task"]) == (tag, length, model, task)]
        with open(result_file) as f:
            official = json.load(f)["score"]
        if ours and abs(sum(ours) / len(ours) - official) > 0.05:
            print(f"warning: {tag} {length} {model} {task}: recomputed {sum(ours)/len(ours):.2f} "
                  f"vs OpenCompass {official:.2f}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory holding the <tag>_<length> work dirs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = collect(args.root)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "length", "model", "task", "idx", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} instances -> {args.out}")


if __name__ == "__main__":
    main()
