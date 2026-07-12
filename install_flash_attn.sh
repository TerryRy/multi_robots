#!/bin/bash
#SBATCH -J install_fa2
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH --mem=64G

source $HOME/ip/venv/bin/activate
export PYTHONUNBUFFERED=1

echo "Python: $(python3 --version)"
echo "Torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# find CUDA_HOME via pytorch first
CUDA_HOME=$(python3 -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.CUDA_HOME or '')" 2>/dev/null)
if [ -z "$CUDA_HOME" ]; then
    CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null
fi
if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    CUDA_HOME=$(ls -d /usr/local/cuda-12* /usr/local/cuda 2>/dev/null | sort -V | tail -1)
fi
echo "CUDA_HOME=$CUDA_HOME"

if [ -z "$CUDA_HOME" ]; then
    echo "ERROR: Cannot find CUDA. flash-attn requires CUDA toolkit."
    exit 1
fi

export CUDA_HOME
echo ""

echo "Trying pre-built wheel..."
# torch 2.6.0+cu124, Python 3.11
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu124torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>&1 | tail -5

if python3 -c "from flash_attn import flash_attn_func" 2>/dev/null; then
    echo "Flash Attention 2 installed OK (pre-built)"
    echo "Done at $(date)"
    exit 0
fi

echo "Pre-built failed, building from source..."
echo "CUDA_HOME=$CUDA_HOME"
pip install ninja packaging 2>/dev/null
MAX_JOBS=4 pip install flash-attn --no-build-isolation 2>&1 | tail -20

echo ""
echo "Verifying..."
python3 -c "from flash_attn import flash_attn_func; print('Flash Attention 2 installed OK')"
echo "Done at $(date)"
