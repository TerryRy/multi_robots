#!/bin/bash
#SBATCH -J install_fa2
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH --mem=32G

source $HOME/ip/venv/bin/activate
export PYTHONUNBUFFERED=1

echo "Python: $(python3 --version)"
echo "Torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

echo ""
echo "Trying pre-built wheel..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu124torch2.6-cp311-cp311-linux_x86_64.whl 2>&1 | tail -5

if python3 -c "from flash_attn import flash_attn_func" 2>/dev/null; then
    echo "Flash Attention 2 installed OK (pre-built)"
else
    echo "Pre-built failed, building from source..."
    pip install flash-attn --no-build-isolation 2>&1 | tail -10
    python3 -c "from flash_attn import flash_attn_func; print('Flash Attention 2 installed OK (source)')"
fi

echo "Done at $(date)"
