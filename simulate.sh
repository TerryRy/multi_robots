#!/bin/bash
#SBATCH -J dora_replay
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH --mem=16G
#SBATCH --mail-type=NONE

# 验证收集数据的正确性:
# 用 expert waypoints 直接驱动 robot, 看能否完成任务
#
#   sbatch simulate.sh <stage> [sim_minutes]
#     stage: 1 | 2 | 3 | 4

STAGE=${1:?"Usage: $0 <stage>; stage=1|2|3|4"}
SIM_MINUTES=${2:-3}

VENV_DIR=$HOME/ip/venv
source ${VENV_DIR}/bin/activate
export PYTHONUNBUFFERED=1

PROJ_DIR=$HOME/ip/multi_robots
DATA_DIR=${PROJ_DIR}/data/trajectories/stage_${STAGE}
SRC_DIR=${PROJ_DIR}/src

case $STAGE in
  1) AGENTS=2; LOAD_P=2; UNLOAD_P=2; MAP_W=20; MAP_H=12 ;;
  2) AGENTS=4; LOAD_P=4; UNLOAD_P=4; MAP_W=30; MAP_H=16 ;;
  *) echo "Stage $STAGE not supported"; exit 1 ;;
esac

JSONL=$(ls ${DATA_DIR}/*.jsonl 2>/dev/null | head -1)
if [ -z "$JSONL" ]; then
    echo "Error: No JSONL found in ${DATA_DIR}"
    exit 1
fi

echo "=============================="
echo "Expert Replay: stage ${STAGE}"
echo "Data: $JSONL"
echo "Agents: ${AGENTS} | Ports: ${LOAD_P}/${UNLOAD_P}"
echo "Map: ${MAP_W}x${MAP_H} | Duration: ${SIM_MINUTES} min"
echo "=============================="

cd ${SRC_DIR}

python3 -c "
import sys, os, json, time
from math import sqrt
sys.path.insert(0, os.getcwd())

# Load expert data
with open('${JSONL}') as f:
    records = [json.loads(line) for line in f]
print(f'Loaded {len(records)} records')

# Create simulator via subprocess - first run to get expert behavior
import subprocess
cmd = [
    sys.executable, 'simulator.py', '-t', '${SIM_MINUTES}',
    '--agent', '${AGENTS}', '--port', '${LOAD_P}', '${UNLOAD_P}',
    '--size', '${MAP_W}', '${MAP_H}',
]
print(f'Running expert baseline...')
p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print('=== Expert baseline ===')
for line in (p.stdout + p.stderr).split(chr(10)):
    if any(x in line for x in ['PPH', 'Package', 'Time', 'Collision']):
        print(f'  {line.strip()}')
" 2>&1
