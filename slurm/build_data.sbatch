#!/bin/bash
# CPU-only job that builds the packed corpus. Env: BUDGET (per-source tokens), OUT (default data/corpus), SHARDS (64).
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --mem=96G --time=08:00:00 --job-name=build-corpus
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source slurm/common.sh
source "$CONDA_ROOT/bin/activate" ttt
python -m inplace_ttt_study.data.build_corpus --out "${OUT:-$STUDY_ROOT/data/corpus}" --shards "${SHARDS:-64}" \
  --tokenizer "$STUDY_ROOT/models/$MODEL" --budget $BUDGET
