#!/usr/bin/env bash
# Run the post-training diagnostics for one seed on one GPU.  scripts/diagnostics.sh [SEED=1]
set -euo pipefail
cd "$(dirname "$0")/.."; source slurm/common.sh
export SEED=${1:-1}
sbatch --job-name "diagnostics-seed$SEED" ${DEPENDENCY:+--dependency="$DEPENDENCY"} --output "$STUDY_ROOT/logs/%x-%j.out" slurm/diagnostics.sbatch
