#!/usr/bin/env bash
# Build the training corpus as a CPU job. Default budget is the study's 3.25B-token mixture.
set -euo pipefail
cd "$(dirname "$0")/.."; source slurm/common.sh
export BUDGET=${BUDGET:-"books=1.3e9 arxiv=0.75e9 code=0.9e9 synthetic=0.3e9"}
sbatch --output "$STUDY_ROOT/logs/%x-%j.out" slurm/build_data.sbatch
