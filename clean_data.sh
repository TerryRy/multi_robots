#!/bin/bash
#SBATCH -J dora_clean
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:01:00
#SBATCH --mem=1G

STAGE=${1:-3}
DATA_DIR=$HOME/ip/multi_robots/data/trajectories/stage_${STAGE}
rm -rf ${DATA_DIR}/*.jsonl ${DATA_DIR}/session_*
echo "Cleaned: ${DATA_DIR}"
