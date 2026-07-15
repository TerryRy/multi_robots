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
    wp_magnitudes = []
    wp_angles = []
    goal_angles = []
    direction_errors = []
    goal_distances = []
    speeds = []
    states = Counter()
    collision_count = 0
    agent_counts = Counter()
    has_moved_count = 0
    total_agent_samples = 0
    step_uniformities = []
    trajectory_curvatures = []
    heading_changes = []
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

            wps = target[sidx]
            if len(wps) < 2:
                continue

            first_wp = wps[0]
            last_wp = wps[-1]
            wp_dx = last_wp[0] - first_wp[0]
            wp_dy = last_wp[1] - first_wp[1]
            wp_mag = math.sqrt(wp_dx**2 + wp_dy**2)
            wp_magnitudes.append(wp_mag)

            if wp_mag > 0.01:
                has_moved_count += 1
                wp_angle_global = compute_angle(wp_dx, wp_dy)
                # Rotate to local frame for direction vs goal
                wp_local_x = wp_dx * math.cos(heading) + wp_dy * math.sin(heading)
                wp_local_y = -wp_dx * math.sin(heading) + wp_dy * math.cos(heading)
                wp_angles.append(compute_angle(wp_local_x, wp_local_y))
                # Goal in local frame
                goal_local_x = goal_dx * math.cos(heading) + goal_dy * math.sin(heading)
                goal_local_y = -goal_dx * math.sin(heading) + goal_dy * math.cos(heading)
                goal_angle_local = compute_angle(goal_local_x, goal_local_y)
                goal_angles.append(goal_angle_local)

                # Direction error (absolute angle diff)
                err = abs(compute_angle(wp_local_x, wp_local_y) - compute_angle(goal_local_x, goal_local_y))
                while err > math.pi:
                    err = 2 * math.pi - err
                direction_errors.append(math.degrees(err))

                # --- Diversity checks ---
                # 1. Step uniformity: std of 7 inter-waypoint distances
                step_dists = []
                for k in range(len(wps) - 1):
                    sd = math.sqrt((wps[k+1][0]-wps[k][0])**2 + (wps[k+1][1]-wps[k][1])**2)
                    step_dists.append(sd)
                if len(step_dists) > 1:
                    step_mean = sum(step_dists) / len(step_dists)
                    step_var = sum((d - step_mean)**2 for d in step_dists) / len(step_dists)
                    step_uniformities.append(step_var**0.5 / max(step_mean, 1e-8))
                # 2. Curvature: max perpendicular distance from first-last line
                ax, ay = first_wp[0], first_wp[1]
                bx, by = last_wp[0], last_wp[1]
                line_len = math.sqrt((bx-ax)**2 + (by-ay)**2)
                if line_len > 0.01:
                    max_curv = 0.0
                    for k in range(1, len(wps) - 1):
                        px, py = wps[k]
                        cross = abs((bx-ax)*(py-ay) - (by-ay)*(px-ax))
                        max_curv = max(max_curv, cross / line_len)
                    trajectory_curvatures.append(max_curv)
                # 3. Heading change: total angle swept over 7 steps
                if len(wps) >= 3:
                    total_turn = 0.0
                    prev_angle = compute_angle(wps[1][0]-wps[0][0], wps[1][1]-wps[0][1])
                    for k in range(2, len(wps)):
                        cur_angle = compute_angle(wps[k][0]-wps[k-1][0], wps[k][1]-wps[k-1][1])
                        diff = cur_angle - prev_angle
                        while diff > math.pi: diff -= 2*math.pi
                        while diff < -math.pi: diff += 2*math.pi
                        total_turn += abs(diff)
                        prev_angle = cur_angle
                    heading_changes.append(math.degrees(total_turn))

            # Print first N records
            if ri < 5:
                heading_deg = math.degrees(heading) % 360
                goal_angle_deg = math.degrees(math.atan2(goal_dy, goal_dx)) % 360
                state_str = state_labels.get(state_idx, "UNKNOWN")
                print(f"  Record {ri}, Agent {ai}:")
                print(f"    pos=({pos_x:.1f},{pos_y:.1f}) heading={heading_deg:.0f}deg")
                print(f"    goal=({goal_dx:.1f},{goal_dy:.1f}) dist={goal_dist:.1f} goal_angle={goal_angle_deg:.0f}deg")
                print(f"    state={state_str} speed={speed:.2f}")
                if wp_mag > 0.01:
                    print(f"    waypoint displacement: {wp_mag:.3f} dir_local={math.degrees(compute_angle(wp_local_x, wp_local_y)):.0f}deg")
                    print(f"    goal_local dir: {math.degrees(compute_angle(goal_local_x, goal_local_y)):.0f}deg")
                    print(f"    direction error: {math.degrees(err):.1f}deg")
                else:
                    print(f"    waypoint displacement: {wp_mag:.3f} (almost stationary)")
                print()

    # Summary
    wp_mag_arr = np.array(wp_magnitudes)
    err_arr = np.array(direction_errors)
    goal_dist_arr = np.array(goal_distances)
    speed_arr = np.array(speeds)

    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    print(f"\n[Agent Count Distribution]")
    for k in sorted(agent_counts):
        print(f"  {k} agents: {agent_counts[k]} records ({100*agent_counts[k]/len(all_records):.0f}%)")

    print(f"\n[Waypoint Displacement] (first_wp → last_wp)")
    print(f"  count={len(wp_mag_arr)}, moved={has_moved_count}/{total_agent_samples}")
    print(f"  mean={wp_mag_arr.mean():.3f} median={np.median(wp_mag_arr):.3f} std={wp_mag_arr.std():.3f}")
    print(f"  min={wp_mag_arr.min():.3f} 25%={np.percentile(wp_mag_arr,25):.3f} 75%={np.percentile(wp_mag_arr,75):.3f} max={wp_mag_arr.max():.3f}")
    if len(wp_mag_arr) > 0:
        zero_pct = 100 * (wp_mag_arr < 0.01).sum() / len(wp_mag_arr)
        small_pct = 100 * ((wp_mag_arr >= 0.01) & (wp_mag_arr < 0.5)).sum() / len(wp_mag_arr)
        ok_pct = 100 * (wp_mag_arr >= 0.5).sum() / len(wp_mag_arr)
        print(f"  stationary (<0.01): {zero_pct:.0f}%  small (0.01-0.5): {small_pct:.0f}%  good (>=0.5): {ok_pct:.0f}%")

    print(f"\n[Direction Error] (waypoint vs goal, in degrees)")
    if len(err_arr) > 0:
        print(f"  count={len(err_arr)}")
        print(f"  mean={err_arr.mean():.1f} median={np.median(err_arr):.1f} std={err_arr.std():.1f}")
        print(f"  min={err_arr.min():.1f} 25%={np.percentile(err_arr,25):.1f} 75%={np.percentile(err_arr,75):.1f} max={err_arr.max():.1f}")
        good_pct = 100 * (err_arr < 30).sum() / len(err_arr)
        ok_pct = 100 * ((err_arr >= 30) & (err_arr < 90)).sum() / len(err_arr)
        bad_pct = 100 * (err_arr >= 90).sum() / len(err_arr)
        print(f"  good (<30deg): {good_pct:.0f}%  ok (30-90deg): {ok_pct:.0f}%  bad (>=90deg): {bad_pct:.0f}%")
        wrong_dir = 100 * (err_arr > 90).sum() / len(err_arr)
        print(f"  going away from goal (>90deg): {wrong_dir:.0f}%")

    print(f"\n[Goal Distance]")
    print(f"  mean={goal_dist_arr.mean():.1f} median={np.median(goal_dist_arr):.1f}")
    print(f"  min={goal_dist_arr.min():.1f} max={goal_dist_arr.max():.1f}")

    print(f"\n[Speed Distribution]")
    print(f"  mean={speed_arr.mean():.2f} median={np.median(speed_arr):.2f}")
    if len(speed_arr) > 0:
        stopped = 100 * (speed_arr < 0.01).sum() / len(speed_arr)
        print(f"  stopped (<0.01): {stopped:.0f}%")

    print(f"\n[Trajectory Diversity] (moving samples only)")
    step_arr = np.array(step_uniformities)
    curv_arr = np.array(trajectory_curvatures)
    turn_arr = np.array(heading_changes)
    if len(step_arr) > 0:
        print(f"  Step CV (std/mean of 7 step distances):")
        print(f"    median={np.median(step_arr):.3f} 25%={np.percentile(step_arr,25):.3f} 75%={np.percentile(step_arr,75):.3f}")
        print(f"    CV=0 = perfectly uniform steps (constant speed)")
        uniform_pct = 100 * (step_arr < 0.01).sum() / len(step_arr)
        print(f"    uniform steps (CV<0.01): {uniform_pct:.0f}%")
    if len(curv_arr) > 0:
        print(f"  Curvature (max deviation from straight line):")
        print(f"    median={np.median(curv_arr):.3f} 25%={np.percentile(curv_arr,25):.3f} 75%={np.percentile(curv_arr,75):.3f}")
        straight_pct = 100 * (curv_arr < 0.01).sum() / len(curv_arr)
        print(f"    perfectly straight (curv<0.01): {straight_pct:.0f}%")
    if len(turn_arr) > 0:
        print(f"  Total heading change over 8 waypoints (degrees):")
        print(f"    median={np.median(turn_arr):.1f} 25%={np.percentile(turn_arr,25):.1f} 75%={np.percentile(turn_arr,75):.1f}")
        no_turn_pct = 100 * (turn_arr < 0.5).sum() / len(turn_arr)
        print(f"    no turning (<0.5deg): {no_turn_pct:.0f}%")

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
        if n_fp <= 3:
            for fp in agent_fingerprints[fid]:
                count = agent_positions[fid].count((fp[0], fp[1]))
                print(f"    (x={fp[0]:.0f}, y={fp[1]:.0f}, gdx={fp[2]:.0f}, gdy={fp[3]:.0f})")
    fps_list = [len(agent_fingerprints[fid]) for fid in agent_fingerprints]
    if fps_list:
        avg_fp = sum(fps_list) / len(fps_list)
        print(f"  average unique combos per agent: {avg_fp:.0f}")
        if avg_fp < 5:
            print(f"  ⚠ LOW DIVERSITY: each agent repeats <5 behavior patterns")

    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")
    if len(err_arr) > 0 and len(wp_mag_arr) > 0:
        median_err = np.median(err_arr)
        median_mag = np.median(wp_mag_arr)
        good_pct = 100 * (err_arr < 30).sum() / len(err_arr)
        zero_pct = 100 * (wp_mag_arr < 0.01).sum() / len(wp_mag_arr)
        print(f"  Direction error median: {median_err:.0f}deg ({good_pct:.0f}% good)")
        print(f"  Waypoint magnitude median: {median_mag:.3f} ({zero_pct:.0f}% stationary)")
        if len(step_arr) > 0:
            step_cv_med = np.median(step_arr)
            straight_pct = 100 * (curv_arr < 0.01).sum() / len(curv_arr) if len(curv_arr) > 0 else 0
            print(f"  Step uniformity CV median: {step_cv_med:.3f}  Straight lines: {straight_pct:.0f}%")
        if median_err < 20 and median_mag > 0.5:
            print(f"  DATA QUALITY: GOOD")
            if len(step_arr) > 0 and np.median(step_arr) < 0.01 and straight_pct > 90:
                print(f"  DIVERSITY: POOR - all trajectories are identical straight lines")
        elif median_err < 45 and median_mag > 0.1:
            print(f"  DATA QUALITY: ACCEPTABLE")
        else:
            print(f"  DATA QUALITY: POOR - investigate further")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect VLA data quality")
    parser.add_argument("--data", type=str, default="data/trajectories/stage_3",
                        help="Data directory with JSONL files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of records to scan")
    args = parser.parse_args()
    inspect(args)
