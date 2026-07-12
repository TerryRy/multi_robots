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

# find CUDA_HOME
CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null
if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    CUDA_HOME=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
fi
if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    echo "CUDA_HOME not found, trying default paths..."
    for p in /usr/local/cuda /opt/cuda; do
        if [ -d "$p" ]; then CUDA_HOME="$p"; break; fi
    done
fi
echo "CUDA_HOME=$CUDA_HOME"

if [ -z "$CUDA_HOME" ]; then
    echo "ERROR: Cannot find CUDA. flash-attn requires CUDA toolkit."
    exit 1
fi

export CUDA_HOME
echo ""

echo "Building flash-attn from source (this takes ~5 min)..."
pip install ninja packaging 2>/dev/null
MAX_JOBS=4 pip install flash-attn --no-build-isolation 2>&1 | tail -20

echo ""
echo "Verifying..."
python3 -c "from flash_attn import flash_attn_func; print('Flash Attention 2 installed OK')"
echo "Done at $(date)"
