"""Break down wrong RULER answers by failure mode, per model and task.

Modes: ``new_question`` (the answer starts a fresh "Question:" block instead of answering, the
signature of the synthetic retrieval format in the training corpus), ``empty``, ``other``.

    python -m inplace_ttt_study.analysis.failure_modes runs/results/ruler/seed1_4k
"""
import argparse
import ast
import collections
import glob
import json
import os
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_dir", help="an OpenCompass work dir such as runs/results/ruler/seed1_4k")
    args = ap.parse_args()
    run = sorted(glob.glob(os.path.join(args.work_dir, "20*")))[-1]
    counts = collections.defaultdict(collections.Counter)
    for f in sorted(glob.glob(os.path.join(run, "predictions", "*", "ruler_*.json"))):
        model = os.path.basename(os.path.dirname(f))
        task = re.sub(r"_\d+\.json$", "", os.path.basename(f))[6:]
        for e in json.load(open(f)).values():
            gold = e["gold"]
            gold = ast.literal_eval(gold) if isinstance(gold, str) and gold.startswith("[") else ([gold] if isinstance(gold, str) else gold)
            pred = e["prediction"]
            if any(str(g).lower() in pred.lower() for g in gold):
                mode = "correct"
            elif pred.lstrip().lower().startswith("question"):
                mode = "new_question"
            elif not pred.strip():
                mode = "empty"
            else:
                mode = "other"
            counts[(model, task)][mode] += 1
    models = sorted({m for m, _ in counts}); tasks = sorted({t for _, t in counts})
    print("| task | " + " | ".join(models) + " |")
    print("|---|" + "---|" * len(models))
    for t in tasks:
        cells = []
        for m in models:
            c = counts[(m, t)]; n = sum(c.values())
            cells.append(f"{100*c['correct']/n:.0f}% ok, {100*c['new_question']/n:.0f}% new-question, {100*c['other']/n:.0f}% other" if n else "")
        print(f"| {t} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
