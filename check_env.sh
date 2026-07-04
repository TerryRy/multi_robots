#!/bin/bash
#SBATCH -J dora_check
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:05:00
#SBATCH --mem=8G
#SBATCH --mail-user=txueae@connect.ust.hk
#SBATCH --mail-type=NONE

VENV_DIR=$HOME/ip/venv
source ${VENV_DIR}/bin/activate

echo "=== Python ==="
which python
python --version

echo ""
echo "=== PyTorch ==="
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('cuda_version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')
print('gpus:', torch.cuda.device_count() if torch.cuda.is_available() else 0)
"

echo ""
echo "=== Packages ==="
python -c "
import transformers; print('transformers:', transformers.__version__)
import timm; print('timm:', timm.__version__)
import Box2D; print('Box2D: OK')" 2>/dev/null || echo "Box2D: NOT INSTALLED"

echo ""
echo "=== HF Cache ==="
ls -lh $HOME/ip/models/hf_cache/hub/ 2>/dev/null || echo "no hub cache"
ls -lh $HOME/ip/models/hf_cache/models--openvla--openvla-7b/snapshots/*/ 2>/dev/null || echo "no model cache"
