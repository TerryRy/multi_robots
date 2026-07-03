#!/bin/bash
#SBATCH -J dora_setup
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH --mem=32G
#SBATCH --mail-user=txueae@connect.ust.hk
#SBATCH --mail-type=END

# ==================== 用法 ====================
# 一次性环境准备: 创建 venv + 安装依赖 + 下载模型
# 只需要在首次部署时跑一次
#
#   sbatch --export=ALL,HF_TOKEN='hf_你的token' setup.sh
#
# 如果已有 venv, 只想重新下载模型:
#   sbatch --export=ALL,HF_TOKEN='hf_你的token' setup.sh --model-only
# =============================================

MODEL_ONLY=${1:-""}

# ==================== 路径 ====================
export HF_TOKEN="${HF_TOKEN:?需要设置 HF_TOKEN}"
PROJ_DIR=$HOME/ip/dorabot_minions-master
VENV_DIR=$HOME/ip/venv
CACHE_DIR=$HOME/ip/models/hf_cache
mkdir -p ${CACHE_DIR}

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
cd ${PROJ_DIR}

# ==================== 1. 创建 venv + 装依赖 ====================
if [ "$MODEL_ONLY" != "--model-only" ]; then
    echo ""
    echo "=== Creating venv (inherits CUDA from base) ==="
    python -m venv ${VENV_DIR} --system-site-packages
    source ${VENV_DIR}/bin/activate

    echo "=== Installing Python packages ==="
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    pip install transformers accelerate peft bitsandbytes
    pip install box2d-py 2>/dev/null || pip install Box2D 2>/dev/null || pip install pybox2d 2>/dev/null || echo "WARNING: Box2D not installed; try: conda install -c conda-forge swig && pip install pybox2d"
    pip install pygame networkx shapely protobuf cmd2 matplotlib pillow
    deactivate
    echo "Dependencies installed."
fi

# ==================== 2. 下载 OpenVLA-7B ====================
echo ""
echo "=== Downloading OpenVLA-7B ==="
export HF_HOME=${CACHE_DIR}
source ${VENV_DIR}/bin/activate

python -c "
from transformers import AutoModel, AutoTokenizer
import os

model_id = 'openvla/openvla-7b'
cache_dir = os.environ['HF_HOME']
print(f'Downloading {model_id} to persistent cache: {cache_dir}')
print('(This will take a few minutes for a 7B model)...')

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True,
)
print('Model OK. Params:', sum(p.numel() for p in model.parameters()) / 1e9, 'B')

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
print('Tokenizer OK.')

hub_path = os.path.join(cache_dir, 'hub')
if os.path.exists(hub_path):
    snaps = [d for d in os.listdir(hub_path) if d.startswith('models--')]
    print(f'Cached: {snaps}')
"
deactivate

echo ""
echo "=== Setup complete ==="
echo "venv: ${VENV_DIR}"
echo "cache: ${CACHE_DIR}"
echo "Setup completed at $(date)"
