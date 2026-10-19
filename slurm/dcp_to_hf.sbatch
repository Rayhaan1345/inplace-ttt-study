#!/bin/bash
# Convert intermediate DCP checkpoints to HuggingFace format (CPU job). Env: ITEMS="RUN:STEP RUN:STEP ...".
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G --time=04:00:00 --job-name=dcp-to-hf
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source slurm/common.sh
source "$CONDA_ROOT/bin/activate" ttt
for item in $ITEMS; do
  RUN=${item%%:*}; STEP=${item##*:}
  CKPT="$STUDY_ROOT/checkpoints/$RUN/checkpoints/global_step_$STEP"
  if [ -f "$CKPT/hf_ckpt/config.json" ]; then echo "$item: already converted"; continue; fi
  echo "[$(date -u +%FT%TZ)] converting $item"
  python "$UPSTREAM/scripts/merge_dcp_to_hf.py" --load-dir "$CKPT" --save-dir "$CKPT/hf_ckpt" \
    --model-assets-dir "$STUDY_ROOT/checkpoints/$RUN/model_assets" --shard-size 5000000000
  ls "$CKPT/hf_ckpt"
done
