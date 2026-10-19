#!/bin/bash
# RULER evaluation with OpenCompass for several model variants, one context length after another in
# the same allocation (fewer queue round trips). Submitted by scripts/eval.sh, which sets LENGTHS,
# NUM_SAMPLES, SPECS and TAG.
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=16 --mem=192G --time=2-00:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source slurm/common.sh
source "$CONDA_ROOT/bin/activate" ttt-eval
export COMPASS_DATA_CACHE="$STUDY_ROOT/opencompass_cache"
REPO=$PWD
for LENGTH in $LENGTHS; do
  WORK="$STUDY_ROOT/results/ruler/${TAG}_${LENGTH}"; mkdir -p "$WORK"; cd "$REPO"
  python -m inplace_ttt_study.eval.make_config --length "$LENGTH" --num-samples "$NUM_SAMPLES" --workers "$SLURM_GPUS_ON_NODE" \
    --variants-dir "$STUDY_ROOT/variants" --out "$WORK/config.py" $SPECS
  echo "[$(date -u +%FT%TZ)] $LENGTH -> $WORK ($(hostname), job $SLURM_JOB_ID)"
  cd "$WORK"  # OpenCompass writes tmp/ and outputs relative to the cwd; keep them out of the repo
  opencompass "$WORK/config.py" -w "$WORK" --dump-eval-details
done
