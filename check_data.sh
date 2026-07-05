#!/bin/bash
# 检测 VLA 收集的数据是否正确
# 用法: bash check_data.sh [data_dir]
#       默认 data/trajectories/stage_1

DATA_DIR=${1:-"data/trajectories/stage_1"}
PROJ_DIR=$(cd "$(dirname "$0")" && pwd)
ABS_DIR="${PROJ_DIR}/${DATA_DIR}"

if [ ! -d "$ABS_DIR" ]; then
    echo "Error: $ABS_DIR not found"
    exit 1
fi

echo "============================================"
echo "Data check: ${DATA_DIR}"
echo "============================================"

# 文件统计
echo ""
echo "--- File stats ---"
ls -lh "${ABS_DIR}"/*.jsonl 2>/dev/null || echo "No jsonl files"
TOTAL=$(cat "${ABS_DIR}"/*.jsonl 2>/dev/null | wc -l)
echo "Total records: ${TOTAL}"

# 检测前几条记录的航点
echo ""
echo "--- Waypoint check (first 5 records) ---"
python3 -c "
import json, os, sys
data_dir = '${ABS_DIR}'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jsonl')])
if not files:
    sys.exit(0)
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
print(f'Result: {ok} OK, {bad} BAD')
if bad > 0:
    print('WARNING: Found repeated waypoints (old bug)!')
    sys.exit(1)
" 2>&1

# 检测图像
echo ""
echo "--- Image check ---"
python3 -c "
import json, os
data_dir = '${ABS_DIR}'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jsonl')])
if not files:
    sys.exit(0)
with open(os.path.join(data_dir, files[0])) as f:
    for i, line in enumerate(f):
        if i >= 3: break
        rec = json.loads(line)
        img_path = rec.get('image_path')
        if img_path:
            full = os.path.join('${PROJ_DIR}', 'data/trajectories', img_path)
            exists = os.path.exists(full)
            size = os.path.getsize(full) if exists else 0
            print(f'  [{i}] {img_path}: exists={exists}, size={size}B')
        else:
            print(f'  [{i}] no image')
" 2>&1

# 文本样本
echo ""
echo "--- Text sample (first 200 chars) ---"
python3 -c "
import json, os
data_dir = '${ABS_DIR}'
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
