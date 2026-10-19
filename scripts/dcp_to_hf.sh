#!/usr/bin/env bash
# Convert saved DCP checkpoints to HuggingFace format in one CPU job.
#   scripts/dcp_to_hf.sh RUN:STEP [RUN:STEP ...]     e.g. scripts/dcp_to_hf.sh ttt_on_seed1:477 ttt_off_seed1:477
set -euo pipefail
cd "$(dirname "$0")/.."; source slurm/common.sh
export ITEMS="$*"
sbatch --job-name "dcp-to-hf" --output "$STUDY_ROOT/logs/%x-%j.out" slurm/dcp_to_hf.sbatch
