# In-Place TTT: a controlled small-budget study on Qwen3-4B

Code for a controlled study of [In-Place Test-Time Training](https://arxiv.org/abs/2604.06169)
(ByteDance Seed, ICLR 2026) on Qwen3-4B-Base. The paper reports RULER gains from continued
pretraining with In-Place TTT against a matched control, but releases no checkpoints or data.
This repository reproduces the matched comparison at a reduced budget (3B tokens, 32k context,
two seeds) and adds the ablations the paper does not report:

* an **ablation** of the trained TTT model with the test-time update disabled (`ttt_lr=0`), which
  separates the effect of the fast-weight update from the effect of training with the module;
* an inference-time sweep of the inner learning rate;
* the **chunk size** question (the paper's ablation prefers 512-1024, the released config uses 4096);
* per-task and per-length RULER breakdowns with confidence intervals, and inference cost.

The study plan, written before the runs, is in [`docs/plan.md`](docs/plan.md).
The findings are written up in [`docs/report.md`](docs/report.md); every table, run id and failure-mode
breakdown behind them is in [`docs/results.md`](docs/results.md).

## Layout

```
configs/train/        VeOmni configs: ttt_on.yaml and ttt_off.yaml differ only in ttt_mode; smoke.yaml
env/                  setup_train_env.sh, setup_eval_env.sh (conda), setup_upstream.sh (pinned clone + patches)
patches/              three small patches on upstream In-Place-TTT (layer validation; per-step metrics; opt-in clipping)
inplace_ttt_study/    data/ (corpus builder), checks/ (checkpoint sanity checks), eval/ (OpenCompass harness),
                      analysis/ (tables and figures)
slurm/                sbatch templates; common.sh resolves paths from local.env
scripts/              one-line launchers: build_data, smoke, train, check_ckpt, dcp_to_hf, diagnostics, eval, main_runs
paper/                LaTeX source of the write-up (ICLR 2026 format); make_tables.sh regenerates its appendix tables
```

Everything runs from the repository root. Large artifacts (models, data, checkpoints, results)
live under `$STUDY_ROOT`, which defaults to `./runs`.

## Setup

```bash
cp local.env.example local.env          # set CONDA_ROOT and STUDY_ROOT (needs ~2 TB)
bash env/setup_train_env.sh             # conda env `ttt`: torch 2.8.0, flash-attn 2.8.3, VeOmni @9b91e16, transformers 4.57.3
bash env/setup_eval_env.sh              # conda env `ttt-eval`: OpenCompass 0.5.4 on the same torch/transformers
bash env/setup_upstream.sh              # third_party/In-Place-TTT at be23248 with patches/ applied
huggingface-cli download Qwen/Qwen3-4B-Base --local-dir "$STUDY_ROOT/models/Qwen3-4B-Base"
```

The upstream code is used as released except for `patches/`: validation that `ttt_layers` are
within the model (the released Qwen3 config lists a 37th layer for a 36-layer model), a
`metrics.jsonl` with loss, gradient norm, learning rate and tokens per step, an abort if two
data-parallel ranks receive the same first micro-batch, and an opt-in `ttt_clip` field that
implements the paper's inference-time delta clipping (off by default, matching the release).

## Data

```bash
scripts/build_data.sh                   # CPU job, ~3 h; writes $STUDY_ROOT/data/corpus/{train/,heldout.jsonl,manifest.json}
```

The corpus is 3.25B tokens packed into 32,736-token sequences from four public sources: PG-19
books (`emozilla/pg19`, 1.3B), arXiv papers (`ccdv/arxiv-summarization`, 0.75B), permissively
licensed GitHub code grouped by repository (`codeparrot/github-code-clean`, 0.9B) and synthetic
retrieval documents (0.3B, see `inplace_ttt_study/data/synthetic.py`). Documents longer than one
sequence continue into the next, sequences are globally shuffled with a fixed seed, and the
manifest records dataset revisions and per-source token counts. Both arms read identical batches
because VeOmni shuffles with the training seed before splitting across ranks.

## Training

```bash
BUDGET="books=2e6 arxiv=1.2e6 code=1.2e6 synthetic=0.6e6" OUT=$STUDY_ROOT/data/corpus_tiny SHARDS=4 scripts/build_data.sh
scripts/smoke.sh                                             # 10 steps on 4 GPUs + checkpoint checks
scripts/train.sh ttt_on_seed1  configs/train/ttt_on.yaml  1  # 1431 steps x 64 x 32k = 3.0B tokens
scripts/train.sh ttt_off_seed1 configs/train/ttt_off.yaml 1
scripts/check_ckpt.sh $STUDY_ROOT/checkpoints/ttt_on_seed1/checkpoints/global_step_1431/hf_ckpt   # several dirs allowed
scripts/dcp_to_hf.sh ttt_on_seed1:477 ttt_off_seed1:477                        # HF export of intermediate checkpoints
scripts/diagnostics.sh 1          # checks, per-group gradient norms, fast-weight delta norms, inference cost (1 GPU)
```

Extra VeOmni overrides can follow the launcher arguments (for example `--model.foundation '{...}'`
for the chunk-size ablation). Checkpoints are saved at 1B, 2B and 3B tokens.

## Evaluation

```bash
scripts/eval.sh calib "4k 32k 64k" 100 base=$STUDY_ROOT/models/Qwen3-4B-Base       # harness calibration
scripts/eval.sh main "4k 8k 16k 32k 64k" 500 ttt_on_s1=<hf_ckpt> ttt_on_s1_frozen=<hf_ckpt>:ttt_lr=0 ttt_off_s1=<hf_ckpt>
```

Evaluation uses the paper's framework (OpenCompass RULER, 13 tasks, greedy decoding, 64 new
tokens) through `inplace_ttt_study/eval`. Variants such as `:ttt_lr=0`, `:ttt_mode=false`,
`:ttt_clip=1e-5` (the paper's inference-time clipping, absent from the released code) or `:yarn=2`
are symlinked copies of a checkpoint with an edited `config.json`.

## Analysis

```bash
python -m inplace_ttt_study.analysis.collect $STUDY_ROOT/results/ruler --out $STUDY_ROOT/results/ruler_instances.csv
python -m inplace_ttt_study.analysis.report $STUDY_ROOT/results/ruler_instances.csv --out $STUDY_ROOT/results/report \
    --pairs ttt_on_s1:ttt_off_s1 ttt_on_s1:ttt_on_s1_frozen ttt_on_s1+ttt_on_s2:ttt_off_s1+ttt_off_s2
python -m inplace_ttt_study.analysis.figure_by_length $STUDY_ROOT/results/report/summary.csv --seed 1 --out docs/figures/ruler_seed1.png
python -m inplace_ttt_study.analysis.curves --out docs/figures/curves_seed1.png ttt_on_seed1=$STUDY_ROOT/checkpoints/ttt_on_seed1 ttt_off_seed1=$STUDY_ROOT/checkpoints/ttt_off_seed1
python -m inplace_ttt_study.analysis.failure_modes $STUDY_ROOT/results/ruler/seed1_32k
```

`collect` rescores every prediction with RULER's containment rule into one per-instance CSV
(tag, length, model, task, idx, score) and cross-checks the task means against OpenCompass's own
results. `report` writes the mean-over-tasks table with 95% bootstrap intervals, per-task tables
and paired differences on the same instances; a side of `--pairs` joined with `+` is averaged per
instance first (the seed-averaged comparison of the decision rule in the study plan), and
`--exclude-tasks` gives the sensitivity analysis. `failure_modes` splits wrong answers into the
"new Question:" format failure, empty answers and the rest.

## License

The upstream code in `third_party/` is Apache-2.0 (ByteDance). The code in this repository is
released under the same license unless stated otherwise.
