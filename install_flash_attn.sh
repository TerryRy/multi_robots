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

# find real CUDA_HOME: use nvcc that can report version
CUDA_HOME=""
for nvcc_cand in $(which -a nvcc 2>/dev/null) /usr/local/cuda*/bin/nvcc /opt/cuda*/bin/nvcc; do
    if [ -x "$nvcc_cand" ] && "$nvcc_cand" --version 2>/dev/null | grep -q release; then
        CUDA_HOME=$(dirname $(dirname "$nvcc_cand"))
        echo "Found valid nvcc at $nvcc_cand"
        break
    fi
done

if [ -z "$CUDA_HOME" ]; then
    echo "ERROR: no valid nvcc found. Try: conda install -c nvidia cuda-nvcc"
    exit 1
fi

export CUDA_HOME
echo "CUDA_HOME=$CUDA_HOME"
echo ""

echo "Building flash-attn from source (this takes ~5 min)..."
pip install ninja packaging -q 2>/dev/null
MAX_JOBS=4 pip install flash-attn --no-build-isolation 2>&1 | grep -E "error|Successfully|Building" | tail -20

echo ""
echo "Verifying..."
python3 -c "from flash_attn import flash_attn_func; print('Flash Attention 2 installed OK')" 2>&1
echo "Done at $(date)"
