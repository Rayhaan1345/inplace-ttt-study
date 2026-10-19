#!/usr/bin/env bash
# Training environment: the stack pinned by the upstream In-Place-TTT README (torch 2.8.0 cu128,
# flash-attn 2.8.3, VeOmni @ 9b91e16, transformers 4.57.3).  Usage: bash env/setup_train_env.sh [ENV]
set -euo pipefail
cd "$(dirname "$0")/.."; source slurm/common.sh
ENV=${1:-ttt}
source "$CONDA_ROOT/bin/activate"
conda create -y -n "$ENV" python=3.11
source "$CONDA_ROOT/bin/activate" "$ENV"
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
WHL=flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
curl -fsSL -o "/tmp/$WHL" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$WHL"
pip install "/tmp/$WHL" && rm -f "/tmp/$WHL"
pip install "veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@9b91e164bea9e17f17ed490aab5e076c2335ca25"
pip install liger-kernel torchdata blobfile tiktoken timm opt_einsum einops wandb huggingface_hub transformers==4.57.3
python -c "import torch, transformers, flash_attn, veomni; print('torch', torch.__version__, '| transformers', transformers.__version__, '| flash_attn', flash_attn.__version__)"
