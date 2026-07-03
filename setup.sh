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
# 一次性环境准备: 下载模型 + 安装依赖
# 只需要在首次部署时跑一次
#
#   sbatch --export=ALL,HF_TOKEN='hf_你的token' setup.sh
#
# 如果已经安装过依赖,只想重新下载模型:
#   sbatch --export=ALL,HF_TOKEN='hf_你的token' setup.sh --model-only
# =============================================

MODEL_ONLY=${1:-""}

# ==================== 环境设置 ====================
export HF_TOKEN="${HF_TOKEN:?需要设置 HF_TOKEN}"
export HF_HOME=/tmp/${USER}/huggingface_cache

PROJ_DIR=$HOME/ip/dorabot_minions-master
MODEL_DIR=$HOME/ip/models/openvla-7b
mkdir -p ${MODEL_DIR} ${HF_HOME}

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
cd ${PROJ_DIR}

# ==================== 1. Python 依赖 ====================
# 注意: 使用 base conda 环境 (继承 CUDA 配置)
# 不要创建 venv/conda env, 直接 pip install 到 base
if [ "$MODEL_ONLY" != "--model-only" ]; then
    echo ""
    echo "=== Installing Python dependencies (to base environment) ==="
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install transformers accelerate peft bitsandbytes
    pip install pybox2d pygame networkx shapely protobuf cmd2 matplotlib pillow

    echo "Dependencies installed."
fi

# ==================== 2. 下载 OpenVLA-7B ====================
echo ""
echo "=== Downloading OpenVLA-7B to ${MODEL_DIR} ==="
python -c "
from transformers import AutoModel, AutoTokenizer
import os

model_id = 'openvla/openvla-7b'
cache_dir = os.environ['HF_HOME']

print(f'Downloading {model_id} to cache ({cache_dir})...')

# 下载并保存到持久目录
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True,
)
model.save_pretrained('${MODEL_DIR}')
del model

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
tokenizer.save_pretrained('${MODEL_DIR}')

print(f'Model saved to ${MODEL_DIR}')
"

echo ""
echo "=== Setup complete ==="
echo "Model: ${MODEL_DIR}"
ls -lh ${MODEL_DIR} | head -5
echo ""
echo "Setup completed at $(date)"
