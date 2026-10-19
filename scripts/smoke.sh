#!/usr/bin/env bash
# Run the smoke test (needs runs/data/corpus_tiny from scripts/build_data.sh with a small BUDGET).
set -euo pipefail
cd "$(dirname "$0")/.."; source slurm/common.sh
sbatch --output "$STUDY_ROOT/logs/%x-%j.out" slurm/smoke.sbatch
