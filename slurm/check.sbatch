#!/bin/bash
# One-GPU job running the checkpoint sanity checks on one or more checkpoints. Env: CKPTS (hf_ckpt dirs), CORPUS.
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=6 --mem=64G --time=02:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source slurm/common.sh
source "$CONDA_ROOT/bin/activate" ttt
CORPUS=${CORPUS:-$STUDY_ROOT/data/corpus}
for CKPT in $CKPTS; do
  echo "[$(date -u +%FT%TZ)] checking $CKPT"
  python -m inplace_ttt_study.checks.check_checkpoint --ckpt "$CKPT" --base "$STUDY_ROOT/models/$MODEL" \
    --heldout "$CORPUS/heldout.jsonl" --n "${N:-8}" --seq-len "${SEQ_LEN:-16384}" --out "$CKPT/checks.json"
done
