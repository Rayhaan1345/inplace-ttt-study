#!/bin/bash
# One training run on a single node. Submitted by scripts/train.sh, which sets RUN, CONFIG, SEED, EXTRA
# (extra VeOmni overrides) and optionally CORPUS (default $STUDY_ROOT/data/corpus).
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --gres=gpu:4 --cpus-per-task=24 --mem=256G --time=3-00:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source slurm/common.sh
source "$CONDA_ROOT/bin/activate" ttt
export MODELING_BACKEND=hf TORCH_NCCL_AVOID_RECORD_STREAMS=1
OUT="$STUDY_ROOT/checkpoints/$RUN"; mkdir -p "$OUT"; CORPUS=${CORPUS:-$STUDY_ROOT/data/corpus}
if [ -n "${WAIT_FOR:-}" ]; then  # optional: hold the allocation until a file (e.g. the corpus manifest) exists
  until [ -f "$WAIT_FOR" ]; do sleep 60; done; echo "[$(date -u +%FT%TZ)] $WAIT_FOR present"
fi
echo "run=$RUN config=$CONFIG seed=$SEED gpus=$SLURM_GPUS_ON_NODE node=$(hostname) job=$SLURM_JOB_ID extra=[$EXTRA]"
git -C "$UPSTREAM" rev-parse HEAD > "$OUT/upstream_commit.txt"; git rev-parse HEAD > "$OUT/study_commit.txt" 2>/dev/null || true
torchrun --standalone --nnodes=1 --nproc-per-node="$SLURM_GPUS_ON_NODE" "$UPSTREAM/tasks/train_torch.py" "$CONFIG" \
  --model.model_path "$STUDY_ROOT/models/$MODEL" --model.config_path "$STUDY_ROOT/models/$MODEL" \
  --model.tokenizer_path "$STUDY_ROOT/models/$MODEL" --data.train_path "$CORPUS/train" \
  --train.output_dir "$OUT" --train.seed "$SEED" --train.use_wandb false $EXTRA
