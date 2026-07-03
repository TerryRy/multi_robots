#!/bin/bash
#SBATCH -J dora_vla_train
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH --mail-user=txueae@connect.ust.hk
#SBATCH --mail-type=BEGIN,END,FAIL

# ==================== 用法 ====================
# VLA 训练: 需要先跑 setup.sh 下载模型到本地
# 不需要 HF Token (读取本地 ~/ip/models/openvla-7b)
#
#   sbatch train.sh <stage> [extra_args...]
#     stage: 1 | 2 | 3 | 4
#
# 示例:
#   sbatch train.sh 1                        # Stage 1: 从头训练
#   sbatch train.sh 2                        # Stage 2: 加载 stage_1
#   sbatch train.sh 3 --epochs 30 --lr 5e-5  # 自定义参数
# =============================================

STAGE=${1:?"Usage: $0 <stage> [extra_args]; stage=1|2|3|4"}
shift  # 移除 stage, 剩余参数传递给 train.py
EXTRA_ARGS="$@"

# ==================== 环境设置 ====================
export HF_HOME=/tmp/${USER}/huggingface_cache
mkdir -p ${HF_HOME}

# source activate dora_env  # 取消注释以启用 conda/env

PROJ_DIR=$HOME/ip/dorabot_minions-master
DATA_DIR=${PROJ_DIR}/data/trajectories
WEIGHT_DIR=${PROJ_DIR}/weights
SRC_DIR=${PROJ_DIR}/src
mkdir -p ${WEIGHT_DIR} ${DATA_DIR}

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

cd ${SRC_DIR}

MODEL="$HOME/ip/models/openvla-7b"
EPOCHS=20
BATCH=1
LR=1e-4

# 拼接基础参数
BASE_ARGS="--model ${MODEL} --use-lora --lora-rank 128 --epochs ${EPOCHS} --batch-size ${BATCH} --lr ${LR}"

case $STAGE in
  1)
    echo "========== Stage 1: train from scratch (${AGENTS} agents) =========="
    python vla/train.py \
        --data "${DATA_DIR}/stage_1" \
        ${BASE_ARGS} \
        --save-dir "${WEIGHT_DIR}/stage_1" \
        ${EXTRA_ARGS}
    ;;
  2)
    echo "========== Stage 2: load stage_1, mix 20% stage_1 + 80% stage_2 =========="
    python vla/train.py \
        --data-mix "${DATA_DIR}/stage_1:0.2,${DATA_DIR}/stage_2:0.8" \
        ${BASE_ARGS} \
        --load-encoder "${WEIGHT_DIR}/stage_1/encoder.pt" \
        --load-action-head "${WEIGHT_DIR}/stage_1/diffusion_head.pt" \
        --load-lora "${WEIGHT_DIR}/stage_1/lora_adapter/" \
        --save-dir "${WEIGHT_DIR}/stage_2" \
        ${EXTRA_ARGS}
    ;;
  3)
    echo "========== Stage 3: load stage_2, mix 15+15+70% =========="
    python vla/train.py \
        --data-mix "${DATA_DIR}/stage_1:0.15,${DATA_DIR}/stage_2:0.15,${DATA_DIR}/stage_3:0.7" \
        ${BASE_ARGS} \
        --load-encoder "${WEIGHT_DIR}/stage_2/encoder.pt" \
        --load-action-head "${WEIGHT_DIR}/stage_2/diffusion_head.pt" \
        --load-lora "${WEIGHT_DIR}/stage_2/lora_adapter/" \
        --save-dir "${WEIGHT_DIR}/stage_3" \
        ${EXTRA_ARGS}
    ;;
  4)
    echo "========== Stage 4: load stage_3, mix 10+10+10+70% (pressure test) =========="
    python vla/train.py \
        --data-mix "${DATA_DIR}/stage_1:0.1,${DATA_DIR}/stage_2:0.1,${DATA_DIR}/stage_3:0.1,${DATA_DIR}/stage_4:0.7" \
        ${BASE_ARGS} \
        --load-encoder "${WEIGHT_DIR}/stage_3/encoder.pt" \
        --load-action-head "${WEIGHT_DIR}/stage_3/diffusion_head.pt" \
        --load-lora "${WEIGHT_DIR}/stage_3/lora_adapter/" \
        --save-dir "${WEIGHT_DIR}/stage_4" \
        ${EXTRA_ARGS}
    ;;
  *)
    echo "Error: stage must be 1, 2, 3, or 4 (got: $STAGE)"
    exit 1
    ;;
esac

echo ""
echo "Stage ${STAGE} completed at $(date)"
ls -lh "${WEIGHT_DIR}/stage_${STAGE}/"
