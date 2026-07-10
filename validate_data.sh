#!/bin/bash
#SBATCH -J dora_validate
#SBATCH -p normal
#SBATCH -A mscitsuperpod
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:05:00
#SBATCH --mem=8G
#SBATCH --mail-type=NONE

# 检测收集到的 expert waypoints 是否正确指向目标
#   sbatch validate_data.sh <stage>

STAGE=${1:?"Usage: $0 <stage>; stage=1|2|3|4"}
DATA_DIR="$HOME/ip/multi_robots/data/trajectories/stage_${STAGE}"

echo "=============================================="
echo "Data Validation: stage ${STAGE}"
echo "Path: ${DATA_DIR}"
echo "=============================================="

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: $DATA_DIR not found"
    exit 1
fi

python3 -c "
import json, os, math
from collections import Counter

data_dir = '${DATA_DIR}'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jsonl')])
if not files:
    print('No JSONL files found')
    exit(1)

total_records = 0
correct_dir = 0
total_agents = 0
good_agents = 0
state_counter = Counter()
still_speed = 0
moving_agents = 0

for fn in files:
    with open(os.path.join(data_dir, fn)) as f:
        for line in f:
            total_records += 1
            rec = json.loads(line)

            for aid in sorted(rec['target_action'].keys()):
                total_agents += 1
                feat = rec['features']['agents'][int(aid)]
                wps = rec['target_action'][aid]

                # Agent state (one-hot)
                states = ['IDLE','LOADING','QUEUING','HALT','CRUISE','PREQUEUE']
                state_idx = feat[5:11].index(1.0) if 1.0 in feat[5:11] else -1
                state_str = states[state_idx] if state_idx >= 0 else 'UNKNOWN'
                state_counter[state_str] += 1

                # Position and goal
                px, py = feat[0], feat[1]
                dxg, dyg = feat[12], feat[13]
                dest_x, dest_y = px + dxg, py + dyg

                # First waypoint
                wx, wy = wps[0]
                wp_dist = math.hypot(wx - px, wy - py)

                # Direction check
                vec_to_dest = (dest_x - px, dest_y - py)
                vec_to_wp = (wx - px, wy - py)
                dot = vec_to_wp[0] * vec_to_dest[0] + vec_to_wp[1] * vec_to_dest[1]

                dest_dist = math.hypot(dxg, dyg)
                is_moving = wp_dist >= 0.02
                points_toward = dot > 0

                if is_moving:
                    moving_agents += 1
                    if points_toward:
                        good_agents += 1

                if is_moving and state_str == 'CRUISE':
                    correct_dir += 1 if points_toward else 0

                if total_agents <= 16 and is_moving:
                    arrow = '→' if points_toward else '✗'
                    print(f'  A{aid} [{state_str:8s}] pos=({px:.1f},{py:.1f}) '
                          f'dest=({dest_x:.1f},{dest_y:.1f}) '
                          f'wp1=({wx:.2f},{wy:.2f}) dist={wp_dist:.3f}m {arrow}')

print()
print('=== State distribution ===')
for state, count in state_counter.most_common():
    pct = 100 * count / total_agents
    bar = '█' * (pct // 5)
    print(f'  {state:10s}: {count:5d} ({pct:5.1f}%) {bar}')

print()
print(f'Total records:     {total_records}')
print(f'Total agents:      {total_agents}')
print(f'Moving agents:     {moving_agents}')
print()
if moving_agents > 0:
    print(f'Direction OK (CRUISE + moving): {correct_dir}/{moving_agents} ({100*correct_dir//moving_agents}%)')
print(f'Moving toward dest:    {good_agents}/{moving_agents} ({100*good_agents//moving_agents}%)')
print()
if correct_dir / max(moving_agents, 1) > 0.8:
    print('✓ Data quality: GOOD (>80% cruise waypoints point toward dest)')
elif correct_dir / max(moving_agents, 1) > 0.5:
    print('~ Data quality: FAIR (50-80%)')
else:
    print('✗ Data quality: POOR (<50%)')
print('==============================================')
"
