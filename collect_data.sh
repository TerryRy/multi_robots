#!/bin/bash
#SBATCH -J dora_collect
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH --mem=32G
#SBATCH --mail-user=txueae@connect.ust.hk
#SBATCH --mail-type=BEGIN,END,FAIL

# 数据收集: 生成训练数据
# 统一使用 HRVOPlanner (主动避让)
# 地图按 robot 密度 0.011/m² 等比缩放
# 航点步距 stride=4 (8航点覆盖 32步 = 0.53s)

STAGE=${1:?"Usage: $0 <stage> [sim_minutes]; stage=1|2|3|4"}
SIM_TIME=${2:-""}

LP="DullPlanner"
STRIDE=4

case $STAGE in
  1)
    AGENTS=2; LOAD_PORTS=2; UNLOAD_PORTS=2
    MAP_W=20; MAP_H=12
    SIM_TIME=${SIM_TIME:-10}
    MIX_ARGS="--vla-stride ${STRIDE}"
    ;;
  2)
    AGENTS=4; LOAD_PORTS=4; UNLOAD_PORTS=4
    MAP_W=30; MAP_H=16
    SIM_TIME=${SIM_TIME:-10}
    MIX_ARGS="--vla-stride ${STRIDE} \
              --mix expert:0.6,policy:0.3,random:0.1 \
              --vla-model openvla/openvla-7b --vla-device cuda \
              --vla-checkpoint $HOME/ip/multi_robots/weights/stage_1"
    ;;
  3)
    AGENTS=5; LOAD_PORTS=5; UNLOAD_PORTS=5
    MAP_W=40; MAP_H=20
    SIM_TIME=${SIM_TIME:-12}
    MIX_ARGS="--vla-stride ${STRIDE} \
              --mix expert:0.4,policy:0.4,random:0.2 \
              --vla-model openvla/openvla-7b --vla-device cuda \
              --vla-checkpoint $HOME/ip/multi_robots/weights/stage_2"
    ;;
  4)
    AGENTS=7; LOAD_PORTS=7; UNLOAD_PORTS=7
    MAP_W=45; MAP_H=20
    SIM_TIME=${SIM_TIME:-15}
    MIX_ARGS="--vla-stride ${STRIDE} \
              --mix expert:0.2,policy:0.6,random:0.2 \
              --vla-model openvla/openvla-7b --vla-device cuda \
              --vla-checkpoint $HOME/ip/multi_robots/weights/stage_3"
    ;;
  *)
    echo "Error: stage must be 1, 2, 3, or 4 (got: $STAGE)"
    exit 1
    ;;
esac

VENV_DIR=$HOME/ip/venv
source ${VENV_DIR}/bin/activate
export HF_HOME=$HOME/ip/models/hf_cache
export PYTHONUNBUFFERED=1

PROJ_DIR=$HOME/ip/multi_robots
DATA_DIR=${PROJ_DIR}/data/trajectories
SRC_DIR=${PROJ_DIR}/src
mkdir -p ${DATA_DIR}

echo "========================"
echo "Stage $STAGE: ${AGENTS} agents, ${LOAD_PORTS}/${UNLOAD_PORTS} ports"
echo "Map: ${MAP_W}x${MAP_H} | LP: ${LP} | Stride: ${STRIDE}"
echo "Mix: ${MIX_ARGS:-pure expert}"
echo "Data: ${DATA_DIR}/stage_${STAGE}/"
echo "========================"
echo ""

cd ${SRC_DIR}

python simulator.py --vla-collect -t ${SIM_TIME} \
    --agent ${AGENTS} --port ${LOAD_PORTS} ${UNLOAD_PORTS} \
    --size ${MAP_W} ${MAP_H} \
    ${MIX_ARGS}

echo ""
echo "Moving data to stage_${STAGE}..."
mkdir -p ${DATA_DIR}/stage_${STAGE}
mv data/trajectories/session_*.jsonl ${DATA_DIR}/stage_${STAGE}/ 2>/dev/null || true
for d in data/trajectories/session_*/; do
    [ -d "$d" ] && mv "$d" ${DATA_DIR}/stage_${STAGE}/ 2>/dev/null || true
done

count=$(ls ${DATA_DIR}/stage_${STAGE}/*.jsonl 2>/dev/null | wc -l)
records=0
for f in ${DATA_DIR}/stage_${STAGE}/*.jsonl; do
    [ -f "$f" ] && r=$(wc -l < "$f") && records=$((records + r))
done
echo "Stage ${STAGE}: ${count} files, ${records} records"
echo ""
echo "Collection completed at $(date)"
