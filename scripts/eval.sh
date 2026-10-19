#!/usr/bin/env bash
# Submit a RULER evaluation.  scripts/eval.sh TAG LENGTHS NUM_SAMPLES name=path[:key=val,...] ...
# LENGTHS is one or more context lengths in quotes, run back to back in one job.
# Env: GPUS (4), TIME (2-00:00:00), CPUS_PER_GPU (4), MEM_PER_GPU (48 GB), DEPENDENCY (sbatch --dependency spec).
# A realistic TIME lets the scheduler backfill.
# e.g. scripts/eval.sh calib "4k 32k 64k" 100 base=runs/models/Qwen3-4B-Base
set -euo pipefail
cd "$(dirname "$0")/.."; source slurm/common.sh
export TAG=$1 LENGTHS=$2 NUM_SAMPLES=$3; shift 3; export SPECS="$*"; GPUS=${GPUS:-4}
sbatch --job-name "eval-$TAG-${LENGTHS// /+}" ${DEPENDENCY:+--dependency="$DEPENDENCY"} --time "${TIME:-2-00:00:00}" --gres "gpu:$GPUS" --cpus-per-task $((${CPUS_PER_GPU:-4} * GPUS)) \
  --mem "$((${MEM_PER_GPU:-48} * GPUS))G" \
  --output "$STUDY_ROOT/logs/%x-%j.out" slurm/eval.sbatch
