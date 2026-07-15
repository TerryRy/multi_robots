#!/bin/bash
#SBATCH -J dora_eval
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH --mem=64G
#SBATCH --mail-user=txueae@connect.ust.hk
#SBATCH --mail-type=END

# ==================== 用法 ====================
# 评估训练好的 VLA 模型 vs Expert (手写规划器)
#
#   sbatch eval.sh <stage> [sim_minutes] [extra_args...]
#     stage: 1 | 2 | 3 | 4 | quick
#     quick: 仅验证 VLA 推理不崩溃 + waypoint 非零, 不对比 Expert
#     sim_minutes: 评估时长 (默认 5, quick 默认 2)
#
# 示例:
#   sbatch eval.sh 1                     # 在 stage_1 上评估 5 分钟
#   sbatch eval.sh 2 10                  # 在 stage_2 上评估 10 分钟
#   sbatch eval.sh 1 5 --agent 4 --port 4 4   # 自定义场景
#   sbatch eval.sh quick                 # 快速管线验证 (2min)
#   sbatch eval.sh fresh                 # 评估 fresh 训练 (stage_3 场景)
# =============================================

STAGE=${1:?"Usage: $0 <stage> [sim_minutes] [extra_args]; stage=1|2|3|4|quick|fresh"}

SIM_MINUTES=""
if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then
    SIM_MINUTES="$2"
    shift 2
else
    shift 1
fi
EXTRA_ARGS="${@}"

if [ "$STAGE" = "quick" ]; then
    SIM_MINUTES=${SIM_MINUTES:-2}
else
    SIM_MINUTES=${SIM_MINUTES:-5}
fi

# ==================== 路径 ====================
VENV_DIR=$HOME/ip/venv
source ${VENV_DIR}/bin/activate

export PYTHONUNBUFFERED=1
PROJ_DIR=$HOME/ip/multi_robots
SRC_DIR=${PROJ_DIR}/src
WEIGHT_DIR=${PROJ_DIR}/weights_wheel
CACHE_DIR=$HOME/ip/models/hf_cache
export HF_HOME=${CACHE_DIR}

MODEL="openvla/openvla-7b"

# 根据 stage 确定默认 agent/port 数
case $STAGE in
  1) AGENTS=2; LOAD_P=2; UNLOAD_P=2; MAP_W=20; MAP_H=12 ;;
  2) AGENTS=4; LOAD_P=4; UNLOAD_P=4; MAP_W=30; MAP_H=16 ;;
  3) AGENTS=5; LOAD_P=5; UNLOAD_P=5; MAP_W=40; MAP_H=20 ;;
  4) AGENTS=7; LOAD_P=7; UNLOAD_P=7; MAP_W=45; MAP_H=20 ;;
  quick) AGENTS=2; LOAD_P=2; UNLOAD_P=2; MAP_W=20; MAP_H=12 ;;
  fresh) AGENTS=5; LOAD_P=5; UNLOAD_P=5; MAP_W=40; MAP_H=20 ;;
  *) echo "Error: stage must be 1-4, quick, or fresh"; exit 1 ;;
esac

CKPT_DIR="${WEIGHT_DIR}/stage_${STAGE}"
echo "=============================="
echo "Validation: stage ${STAGE}"
echo "Checkpoint: ${CKPT_DIR}"
echo "Agents: ${AGENTS} | Ports: ${LOAD_P}/${UNLOAD_P}"
echo "Map: ${MAP_W}x${MAP_H} | Duration: ${SIM_MINUTES} min"
echo "=============================="
echo ""

cd ${SRC_DIR}

# ==================== 1. VLA 模型评估 ====================
echo ""
echo ">>> [1/2] VLA model inference (${AGENTS} agents, ${SIM_MINUTES} min) <<<"
timeout $((SIM_MINUTES * 60 + 60)) python simulator.py \
    --vla --vla-model ${MODEL} --vla-device cuda \
    --vla-checkpoint ${CKPT_DIR} \
    -t ${SIM_MINUTES} \
    --agent ${AGENTS} --port ${LOAD_P} ${UNLOAD_P} \
    --size ${MAP_W} ${MAP_H} \
    ${EXTRA_ARGS}
echo ">>> VLA inference done <<<"

# ==================== 2. Expert 基线 ====================
if [ "$STAGE" != "quick" ]; then
    echo ""
    echo "=============================="
    echo ""
    echo ">>> [2/2] Expert baseline (hand-written planners, ${SIM_MINUTES} min) <<<"
timeout $((SIM_MINUTES * 60 * 5 + 60)) python simulator.py \
        -t ${SIM_MINUTES} \
        --agent ${AGENTS} --port ${LOAD_P} ${UNLOAD_P} \
        --size ${MAP_W} ${MAP_H} \
        ${EXTRA_ARGS}
    echo ">>> Expert baseline done <<<"
fi

echo ""
echo "=============================="
echo "Validation complete at $(date)"
echo "=============================="
