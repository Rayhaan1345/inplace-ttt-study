#!/bin/bash
# One-GPU job running the post-training diagnostics on the final checkpoints of one seed:
# checkpoint checks, per-group gradient norms, fast-weight delta norms (TTT arm) and inference cost.
# Env: SEED (1), CORPUS (default data/corpus).
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=6 --mem=96G --time=06:00:00 --job-name=diagnostics
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source slurm/common.sh
source "$CONDA_ROOT/bin/activate" ttt
export MODELING_BACKEND=hf
SEED=${SEED:-1}; CORPUS=${CORPUS:-$STUDY_ROOT/data/corpus}; HELDOUT="$CORPUS/heldout.jsonl"; BASE="$STUDY_ROOT/models/$MODEL"
ON="$STUDY_ROOT/checkpoints/ttt_on_seed$SEED/checkpoints/global_step_1431/hf_ckpt"
OFF="$STUDY_ROOT/checkpoints/ttt_off_seed$SEED/checkpoints/global_step_1431/hf_ckpt"
for CKPT in "$ON" "$OFF"; do
  echo "[$(date -u +%FT%TZ)] checks $CKPT"
  python -m inplace_ttt_study.checks.check_checkpoint --ckpt "$CKPT" --base "$BASE" --heldout "$HELDOUT" --n 8 --seq-len 16384 --out "$CKPT/checks.json"
  echo "[$(date -u +%FT%TZ)] gradient norms $CKPT"
  python -m inplace_ttt_study.checks.grad_norms --ckpt "$CKPT" --heldout "$HELDOUT" --n 2 --seq-len 16384 --out "$CKPT/grad_norms.json"
  echo "[$(date -u +%FT%TZ)] inference cost $CKPT"
  python -m inplace_ttt_study.checks.inference_cost --ckpt "$CKPT" --heldout "$HELDOUT" --out "$CKPT/inference_cost.json"
done
echo "[$(date -u +%FT%TZ)] delta norms $ON"
python -m inplace_ttt_study.checks.delta_norms --ckpt "$ON" --heldout "$HELDOUT" --n 8 --seq-len 32768 --out "$ON/delta_norms.json"
echo "[$(date -u +%FT%TZ)] gradient norms of the base model (reference)"
python -m inplace_ttt_study.checks.grad_norms --ckpt "$BASE" --heldout "$HELDOUT" --n 2 --seq-len 16384 --out "$STUDY_ROOT/results/base_grad_norms.json"
