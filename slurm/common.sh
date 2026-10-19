# Sourced by the sbatch scripts and scripts/*.sh after `cd` to the repo root.
# Optional local.env overrides CONDA_ROOT and STUDY_ROOT; defaults work for a plain checkout.
REPO_ROOT=$PWD
[ -f local.env ] && source local.env
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
STUDY_ROOT=${STUDY_ROOT:-$REPO_ROOT/runs}
UPSTREAM=$REPO_ROOT/third_party/In-Place-TTT
MODEL=${MODEL:-Qwen3-4B-Base}
export PYTHONPATH="$REPO_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$STUDY_ROOT/hf_cache} TOKENIZERS_PARALLELISM=false
mkdir -p "$STUDY_ROOT"/{models,data,checkpoints,results,logs}
