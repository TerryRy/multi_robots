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
export PYTHONUNBUFFERED=1
source ${VENV_DIR}/bin/activate 2>/dev/null

echo "=== Python ==="
which python
python --version
echo ""

echo "=== System CUDA Driver ==="
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

echo "=== PyTorch ==="
python -c "
try:
    import torch
    print('torch:', torch.__version__)
    print('cuda_available:', torch.cuda.is_available())
    print('cuda_version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')
    print('gpu_count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)
except ImportError:
    print('torch: NOT INSTALLED')
" 2>&1
echo ""

echo "=== Site Packages ==="
python -c "import sys; print('\n'.join(sys.path))" 2>&1
echo ""

echo "=== Key Packages ==="
for pkg in transformers timm Box2D; do
    python -c "import $pkg; print('$pkg: OK')" 2>/dev/null && continue
    echo "$pkg: MISSING"
done
echo ""

echo "=== HF Cache ==="
find $HOME/ip/models/hf_cache -maxdepth 4 -type f -name "*.safetensors" 2>/dev/null | head -5
find $HOME/ip/models/hf_cache -maxdepth 4 -type f -name "*.json" 2>/dev/null | head -5
echo "Size:"
du -sh $HOME/ip/models/hf_cache 2>/dev/null || echo "0"
