#!/bin/bash
#SBATCH -J dora_check
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:05:00
#SBATCH --mem=8G
#SBATCH --mail-type=NONE

# ==================== 用法 ====================
# 检测 VLA 收集的数据是否正确
#   sbatch check_data.sh <stage>
#     stage: 1 | 2 | 3 | 4 (默认 1)
# =============================================

STAGE=${1:-1}
DATA_DIR="$HOME/ip/multi_robots/data/trajectories/stage_${STAGE}"

echo "============================================"
echo "Data check: stage ${STAGE}"
echo "Path: ${DATA_DIR}"
echo "============================================"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: $DATA_DIR not found"
    exit 1
fi

echo ""
echo "--- File stats ---"
TOTAL=$(cat "${DATA_DIR}"/*.jsonl 2>/dev/null | wc -l)
echo "Total records: ${TOTAL}"

echo ""
echo "--- Waypoint check ---"
python3 -c "
import json, os
data_dir = '${DATA_DIR}'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jsonl')])
if not files:
    print('No jsonl files')
    exit(0)
ok, bad = 0, 0
for fn in files[:1]:
    with open(os.path.join(data_dir, fn)) as f:
        for i, line in enumerate(f):
            if i >= 5: break
            rec = json.loads(line)
            for aid in sorted(rec['target_action']):
                wps = rec['target_action'][aid]
                d = len(set(tuple(w) for w in wps))
                tag = 'OK' if d > 1 else 'REPEATED'
                if d > 1: ok += 1
                else: bad += 1
                print(f'  [{i}] Agent {aid}: {len(wps)} wps, {d} distinct [{tag}]')
print(f'Result: {ok} ok, {bad} bad')
if bad > 0:
    print('WARNING: Found repeated waypoints (old bug)!')
"

echo ""
echo "--- Image check ---"
python3 -c "
import json, os
data_dir = '${DATA_DIR}'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jsonl')])
if files:
    with open(os.path.join(data_dir, files[0])) as f:
        for i, line in enumerate(f):
            if i >= 3: break
            rec = json.loads(line)
            img_path = rec.get('image_path')
            if img_path:
                full = os.path.join('$HOME/ip/multi_robots/data/trajectories', img_path)
                exists = os.path.exists(full)
                size = os.path.getsize(full) if exists else 0
                print(f'  [{i}] img: exists={exists}, size={size}B')
            else:
                print(f'  [{i}] no image')
"

echo ""
echo "--- Text sample ---"
python3 -c "
import json, os
data_dir = '${DATA_DIR}'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jsonl')])
if files:
    with open(os.path.join(data_dir, files[0])) as f:
        rec = json.loads(f.readline())
    print(rec['text_prompt'][:200])
"

echo ""
echo "============================================"
echo "Done."
echo "============================================"
