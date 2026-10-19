#!/bin/bash
# Smoke test on 4 GPUs: 10 training steps of the ttt_on recipe on the tiny corpus, then the checkpoint checks.
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=24 --mem=256G --time=01:00:00 --job-name=smoke
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source slurm/common.sh
source "$CONDA_ROOT/bin/activate" ttt
export MODELING_BACKEND=hf TORCH_NCCL_AVOID_RECORD_STREAMS=1
CORPUS=${CORPUS:-$STUDY_ROOT/data/corpus_tiny}
OUT="$STUDY_ROOT/checkpoints/smoke"; rm -rf "$OUT"; mkdir -p "$OUT"
torchrun --standalone --nnodes=1 --nproc-per-node=4 "$UPSTREAM/tasks/train_torch.py" configs/train/smoke.yaml \
  --model.model_path "$STUDY_ROOT/models/$MODEL" --model.config_path "$STUDY_ROOT/models/$MODEL" \
  --model.tokenizer_path "$STUDY_ROOT/models/$MODEL" --data.train_path "$CORPUS/train" \
  --train.output_dir "$OUT" --train.use_wandb false ${EXTRA:-}
echo "=== metrics.jsonl ==="; cat "$OUT/metrics.jsonl"
CKPT=$(ls -d "$OUT"/checkpoints/global_step_*/hf_ckpt | tail -1)
python -m inplace_ttt_study.checks.check_checkpoint --ckpt "$CKPT" --base "$STUDY_ROOT/models/$MODEL" \
  --heldout "$CORPUS/heldout.jsonl" --n 2 --seq-len 8192
