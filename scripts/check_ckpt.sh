#!/usr/bin/env bash
# Run the checkpoint checks on one GPU.  scripts/check_ckpt.sh <hf_ckpt dir> [<hf_ckpt dir> ...]   (env: CORPUS, N, SEQ_LEN)
set -euo pipefail
cd "$(dirname "$0")/.."; source slurm/common.sh
export CKPTS="$*"
sbatch --job-name "check-ckpt" --output "$STUDY_ROOT/logs/%x-%j.out" slurm/check.sbatch
