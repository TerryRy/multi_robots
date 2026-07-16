"""
Data quality inspection for VLA collected trajectories.
Run on SLURM cluster (lightweight, no GPU needed):

    python src/vla/inspect_data.py --data data/trajectories/stage_3
"""
import json
import os
import argparse
import math
import numpy as np
from collections import Counter


feature_names = {
    0: "pos_x", 1: "pos_y",
    2: "heading_cos", 3: "heading_sin",
    4: "speed",
    5: "IDLE", 6: "LOADING", 7: "QUEUING", 8: "HALT", 9: "CRUISE", 10: "PREQUEUE",
    11: "carrying",
    12: "goal_dx", 13: "goal_dy",
    14: "goal_heading_cos", 15: "goal_heading_sin",
}

state_labels = {5: "IDLE", 6: "LOADING", 7: "QUEUING", 8: "HALT", 9: "CRUISE", 10: "PREQUEUE"}


def extract_state_idx(feat):
    """Return state index (5-10) from one-hot or -1 if none."""
    for i in range(5, 11):
        if feat[i] > 0.5:
            return i
    return -1


def compute_angle(vx, vy):
    return math.atan2(vy, vx)


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def inspect(args):
    data_root = args.data
    sample_limit = args.limit

    all_records = []
    for fname in sorted(os.listdir(data_root)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(data_root, fname)
        recs = load_jsonl(fpath)
        all_records.extend(recs)
        if sample_limit and len(all_records) >= sample_limit:
            all_records = all_records[:sample_limit]
            break

    print(f"\n{'='*60}")
    print(f"Data: {data_root}")
    print(f"Files: {len([f for f in os.listdir(data_root) if f.endswith('.jsonl')])}")
    print(f"Records: {len(all_records)}")
    print(f"{'='*60}\n")

    if not all_records:
        print("No records found.")
        return

    # Stats accumulators
    goal_distances = []
    speeds = []
    states = Counter()
    collision_count = 0
    agent_counts = Counter()
    total_agent_samples = 0
    v_forward_all = []
    omega_all = []
    mean_vfwd = []
    mean_omega = []
    cruise_moving = 0
    cruise_total = 0
    stopped_still = 0
    stopped_total = 0
    agent_fingerprints = {}  # agent_id → set of (pos_x, pos_y, goal_dx, goal_dy)
    agent_positions = {}     # agent_id → list of positions

    for ri, record in enumerate(all_records):
        text = record.get("text_prompt", "")
        features = record.get("features", {})
        agents_raw = features.get("agents", [])
        target = record.get("target_action", {})

        # Filter out padded agents (all zeros)
        agents = [a for a in agents_raw if any(v != 0.0 for v in a[:5])]

        if "Collisions:" in text and "AA=0 AO=0" not in text:
            collision_count += 1

        agent_counts[len(agents)] += 1

        for ai, feat in enumerate(agents):
            sidx = str(ai)
            if sidx not in target:
                continue

            total_agent_samples += 1

            pos_x, pos_y = feat[0], feat[1]
            heading = math.atan2(feat[3], feat[2])
            speed = feat[4]
            speeds.append(speed)
            state_idx = extract_state_idx(feat)
            if state_idx >= 0:
                states[state_idx] += 1

            goal_dx = feat[12]
            goal_dy = feat[13]
            goal_dist = math.sqrt(goal_dx**2 + goal_dy**2)
            # Track unique (position, goal) per agent
            fid = f"agent_{int(ai)}"
            fp = (round(pos_x, 0), round(pos_y, 0), round(goal_dx, 0), round(goal_dy, 0))
            if fid not in agent_fingerprints:
                agent_fingerprints[fid] = set()
            agent_fingerprints[fid].add(fp)
            if fid not in agent_positions:
                agent_positions[fid] = []
            agent_positions[fid].append((pos_x, pos_y))
            goal_distances.append(goal_dist)

            pairs = target[sidx]
            if len(pairs) < 2:
                continue

            # Extract v_forward and omega from (v, ω) pairs
            v_vals = [p[0] for p in pairs]
            w_vals = [p[1] for p in pairs]
            v_forward_all.extend(v_vals)
            omega_all.extend(w_vals)

            mean_v = sum(v_vals) / len(v_vals)
            mean_w = sum(w_vals) / len(w_vals)
            mean_vfwd.append(mean_v)
            mean_omega.append(mean_w)

            # State-action consistency: CRUISE agents should have v > 0
            if state_idx == 9:  # CRUISE
                is_moving = 1 if mean_v > 0.3 else 0
                cruise_moving += is_moving
                cruise_total += 1
            elif state_idx in (5, 6, 8):  # IDLE, LOADING, HALT
                is_stopped = 1 if mean_v < 0.05 else 0
                stopped_still += is_stopped
                stopped_total += 1

            # Print first N records
            if ri < 5:
                heading_deg = math.degrees(heading) % 360
                goal_angle_deg = math.degrees(math.atan2(goal_dy, goal_dx)) % 360
                state_str = state_labels.get(state_idx, "UNKNOWN")
                print(f"  Record {ri}, Agent {ai}:")
                print(f"    pos=({pos_x:.1f},{pos_y:.1f}) heading={heading_deg:.0f}deg")
                print(f"    goal=({goal_dx:.1f},{goal_dy:.1f}) dist={goal_dist:.1f} goal_angle={goal_angle_deg:.0f}deg")
                print(f"    state={state_str} speed={speed:.2f}")
                if mean_v > 0.01:
                    print(f"    v_forward mean={mean_v:.3f}  omega mean={mean_w:.3f}")
                else:
                    print(f"    v_forward mean={mean_v:.3f} (stationary)")
                print()

    # Summary
    goal_dist_arr = np.array(goal_distances)
    speed_arr = np.array(speeds)

    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    print(f"\n[Agent Count Distribution]")
    for k in sorted(agent_counts):
        print(f"  {k} agents: {agent_counts[k]} records ({100*agent_counts[k]/len(all_records):.0f}%)")

    v_arr = np.array(v_forward_all)
    w_arr = np.array(omega_all)
    mv_arr = np.array(mean_vfwd)

    print(f"\n[v_forward Distribution] (forward speed per agent)")
    print(f"  count={len(v_arr)}")
    print(f"  mean={v_arr.mean():.3f} median={np.median(v_arr):.3f} std={v_arr.std():.3f}")
    print(f"  min={v_arr.min():.3f} 25%={np.percentile(v_arr,25):.3f} 75%={np.percentile(v_arr,75):.3f} max={v_arr.max():.3f}")
    stopped_v = 100 * (v_arr < 0.01).sum() / len(v_arr) if len(v_arr) > 0 else 0
    print(f"  stopped (v<0.01): {stopped_v:.0f}%   moving (v>0.5): {100*(v_arr>0.5).sum()/len(v_arr):.0f}%")

    print(f"\n[omega Distribution] (turning rate)")
    print(f"  count={len(w_arr)}")
    print(f"  mean={w_arr.mean():.3f} median={np.median(w_arr):.3f} std={w_arr.std():.3f}")
    print(f"  min={w_arr.min():.3f} 25%={np.percentile(w_arr,25):.3f} 75%={np.percentile(w_arr,75):.3f} max={w_arr.max():.3f}")
    straight_w = 100 * (abs(w_arr) < 0.1).sum() / len(w_arr) if len(w_arr) > 0 else 0
    big_turn = 100 * (abs(w_arr) > 2.0).sum() / len(w_arr) if len(w_arr) > 0 else 0
    print(f"  straight (|ω|<0.1): {straight_w:.0f}%   sharp turn (|ω|>2): {big_turn:.0f}%")

    print(f"\n[State×Action Consistency]")
    print(f"  CRUISE agents moving (v>0.3): {cruise_moving}/{cruise_total} ({100*cruise_moving/max(cruise_total,1):.0f}%)")
    print(f"  IDLE/LOAD/HALT agents stopped (v<0.05): {stopped_still}/{stopped_total} ({100*stopped_still/max(stopped_total,1):.0f}%)")

    print(f"\n[Goal Distance]")
    goal_dist_arr = np.array(goal_distances)
    print(f"  mean={goal_dist_arr.mean():.1f} median={np.median(goal_dist_arr):.1f}")
    print(f"  min={goal_dist_arr.min():.1f} max={goal_dist_arr.max():.1f}")

    print(f"\n[Speed Distribution] (feature[4])")
    speed_arr = np.array(speeds)
    print(f"  mean={speed_arr.mean():.2f} median={np.median(speed_arr):.2f}")

    print(f"\n[State Distribution]")
    for sidx in sorted(states.keys()):
        label = state_labels.get(sidx, f"idx{sidx}")
        count = states[sidx]
        print(f"  {label}: {count} ({100*count/total_agent_samples:.0f}%)")

    collision_pct = 100 * collision_count / len(all_records)
    print(f"\n[Collisions in Text Prompt]")
    print(f"  {collision_count}/{len(all_records)} records ({collision_pct:.0f}%)")

    print(f"\n[Data Diversity]")
    for fid in sorted(agent_fingerprints.keys()):
        n_fp = len(agent_fingerprints[fid])
        n_total = len(agent_positions[fid])
        print(f"  {fid}: {n_fp} unique (pos,goal) combos out of {n_total} records")
    fps_list = [len(agent_fingerprints[fid]) for fid in agent_fingerprints]
    if fps_list:
        avg_fp = sum(fps_list) / len(fps_list)
        print(f"  average unique combos per agent: {avg_fp:.0f}")
        if avg_fp < 5:
            print(f"  ⚠ LOW DIVERSITY: each agent repeats <5 behavior patterns")

    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")
    if len(v_arr) > 0:
        median_v = np.median(v_arr)
        cruise_ok = 100 * cruise_moving / max(cruise_total, 1)
        print(f"  v_forward median: {median_v:.2f}  CRUISE moving: {cruise_ok:.0f}%")
        if median_v > 0.5 and cruise_ok > 80:
            print(f"  DATA QUALITY: GOOD - expert moves correctly")
        elif median_v > 0.1 and cruise_ok > 50:
            print(f"  DATA QUALITY: ACCEPTABLE")
        else:
            print(f"  DATA QUALITY: POOR - agent motion does not match state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect VLA data quality")
    parser.add_argument("--data", type=str, default="data/trajectories/stage_3",
                        help="Data directory with JSONL files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of records to scan")
    args = parser.parse_args()
    inspect(args)
