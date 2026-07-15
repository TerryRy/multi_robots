#!/bin/bash
#SBATCH -J dora_inspect
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH --mem=8G
#SBATCH --mail-user=txueae@connect.ust.hk
#SBATCH --mail-type=END

# ==================== Usage ====================
# sbatch inspect_data.sh [stage]
#   stage: 1 | 2 | 3 | 4 (default: 3)
# ===============================================

STAGE=${1:-3}
VENV_DIR=$HOME/ip/venv
source ${VENV_DIR}/bin/activate

export PYTHONUNBUFFERED=1
DATA_DIR=$HOME/ip/multi_robots/data/trajectories_wheel/stage_${STAGE}

echo "============================================"
echo "Data Inspection: stage_${STAGE}"
echo "Data dir: ${DATA_DIR}"
echo "============================================"

cd $HOME/ip/multi_robots/src
python vla/inspect_data.py --data "${DATA_DIR}"
