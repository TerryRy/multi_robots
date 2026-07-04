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
PROJ_DIR=$HOME/ip/multi_robots
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
    # NVIDIA H800, Driver 570 → CUDA 12.x
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
    pip install "transformers>=4.37.0,<4.40.0" accelerate "peft>=0.10.0,<0.13.0" bitsandbytes "timm>=0.9.10,<1.0.0"
    pip install box2d-py 2>/dev/null || pip install Box2D 2>/dev/null || pip install pybox2d 2>/dev/null || echo "WARNING: Box2D not installed; try: conda install -c conda-forge swig && pip install pybox2d"
    pip install pygame networkx shapely "protobuf<4.0" cmd2 matplotlib pillow
    deactivate
    echo "Dependencies installed."
fi

# ==================== 2. 下载 OpenVLA-7B ====================
echo ""
echo "=== Downloading OpenVLA-7B ==="
export HF_HOME=${CACHE_DIR}
export PYTHONUNBUFFERED=1
source ${VENV_DIR}/bin/activate

python -c "
import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer
import os, glob

model_id = 'openvla/openvla-7b'
cache_dir = os.environ['HF_HOME']
print(f'Downloading {model_id} to cache: {cache_dir}')
print('(7B model, may take 5-10 minutes)...')

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    trust_remote_code=True,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
print('Model OK. Params: {:.2f}B'.format(sum(p.numel() for p in model.parameters()) / 1e9))

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
print('Tokenizer OK.')

# 验证权重文件存在
snap_dir = os.path.join(cache_dir, 'models--openvla--openvla-7b', 'snapshots')
if os.path.exists(snap_dir):
    for snap in os.listdir(snap_dir):
        files = glob.glob(os.path.join(snap_dir, snap, '*'))
        total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / 1e9
        print(f'  snapshot {snap}: {len(files)} files, {total_size:.1f}GB')
"
deactivate

echo ""
echo "=== Setup complete ==="
echo "venv: ${VENV_DIR}"
echo "cache: ${CACHE_DIR}"
echo "Setup completed at $(date)"
