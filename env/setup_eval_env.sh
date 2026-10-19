#!/usr/bin/env bash
# Evaluation environment: OpenCompass (the framework used by the paper) on the same torch/transformers
# versions as training. The TTT inference classes come from third_party/In-Place-TTT via PYTHONPATH.
# Usage: bash env/setup_eval_env.sh [ENV]
set -euo pipefail
cd "$(dirname "$0")/.."; source slurm/common.sh
ENV=${1:-ttt-eval}
source "$CONDA_ROOT/bin/activate"
conda create -y -n "$ENV" python=3.11
source "$CONDA_ROOT/bin/activate" "$ENV"
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
WHL=flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
curl -fsSL -o "/tmp/$WHL" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$WHL"
pip install "/tmp/$WHL" && rm -f "/tmp/$WHL"
pip install opencompass==0.5.4 transformers==4.57.3 "pyarrow<21" accelerate einops opt_einsum wonderwords nltk  # pyarrow>=21 needs numpy 2, opencompass pins numpy<2
python -c "import opencompass, transformers, flash_attn; print('opencompass', opencompass.__version__, '| transformers', transformers.__version__)"
