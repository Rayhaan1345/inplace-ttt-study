#!/usr/bin/env bash
# Submit a training run.  scripts/train.sh RUN CONFIG [SEED=1] [GPUS=4] [extra VeOmni overrides...]
# e.g. scripts/train.sh ttt_on_seed1 configs/train/ttt_on.yaml 1 4
# Env: CORPUS (default data/corpus), TIME (3-00:00:00), CPUS_PER_GPU (6), MEM_PER_GPU (64 GB),
#      DEPENDENCY (an sbatch --dependency spec such as afterok:123:124),
#      WAIT_FOR (a file the job waits for before training, e.g. the corpus manifest, so GPUs can be
#      claimed while data is still building).
# Modest CPU and memory requests schedule faster on shared nodes; training needs about 3 CPU threads per GPU.
set -euo pipefail
cd "$(dirname "$0")/.."; source slurm/common.sh
export RUN=$1 CONFIG=$2 SEED=${3:-1}; GPUS=${4:-4}; shift $(( $# < 4 ? $# : 4 )); export EXTRA="$*"
sbatch --job-name "$RUN" ${DEPENDENCY:+--dependency="$DEPENDENCY"} --time "${TIME:-3-00:00:00}" --gres "gpu:$GPUS" --cpus-per-task $((${CPUS_PER_GPU:-6} * GPUS)) \
  --mem "$((${MEM_PER_GPU:-64} * GPUS))G" \
  --output "$STUDY_ROOT/logs/%x-%j.out" slurm/train.sbatch
