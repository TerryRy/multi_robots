import json
import os
import time


class DataCollector:
    """
    Records (state, hand-written expert waypoint) pairs for VLA training.
    Supports both discrete (BC) and continuous (Diffusion) action modes.
    """

    def __init__(self, save_dir="data/trajectories", action_mode="diffusion",
                 chunk_size=8, collect_every_n_steps=4):
        self.save_dir = save_dir
        self.action_mode = action_mode
        self.chunk_size = chunk_size
        self.collect_every_n_steps = collect_every_n_steps
        self._step_counter = 0
        self._buffer = []
        self._session_id = f"session_{int(time.time())}"
        os.makedirs(save_dir, exist_ok=True)

    def collect(self, text_prompt, features, agents_data, expert_waypoints_dict):
        """
        Record one frame.

        text_prompt: serialized text for VLA input
        features: numerical feature dict
        agents_data: list of per-agent dict (from StateSerializer)
        expert_waypoints_dict: {agent_id: [(world_x, world_y), ...]}  from hand-written planner
        """

        if len(expert_waypoints_dict) == 0:
            return

        max_len = max(len(wp) for wp in expert_waypoints_dict.values())
        for aid, wps in expert_waypoints_dict.items():
            while len(wps) < max_len:
                wps.append(wps[-1] if wps else (0.0, 0.0))

        if self.action_mode == "discrete":
            target = self._discretize_waypoints(expert_waypoints_dict)
        else:
            target = {
                str(aid): [(round(wx, 4), round(wy, 4)) for wx, wy in wps]
                for aid, wps in expert_waypoints_dict.items()
            }

        record = {
            "text_prompt": text_prompt,
            "features": features,
            "target_action": target,
            "agent_count": len(agents_data),
            "timestamp": time.time(),
        }

        self._buffer.append(record)
        self._step_counter += 1

    def step(self):
        self._step_counter += 1

    def should_collect(self):
        return self._step_counter % self.collect_every_n_steps == 0

    def flush(self):
        if len(self._buffer) == 0:
            return
        filepath = os.path.join(self.save_dir, f"{self._session_id}.jsonl")
        with open(filepath, 'a') as f:
            for record in self._buffer:
                f.write(json.dumps(record) + "\n")
        self._buffer.clear()

    def _discretize_waypoints(self, expert_waypoints_dict, n_bins_x=25, n_bins_y=17,
                               max_dx=0.6, max_dy=0.4):
        discretized = {}
        for aid, wps in expert_waypoints_dict.items():
            tokens = []
            for wx, wy in wps:
                idx_x = min(n_bins_x - 1, max(0, int((wx + max_dx) / (2 * max_dx) * n_bins_x)))
                idx_y = min(n_bins_y - 1, max(0, int((wy + max_dy) / (2 * max_dy) * n_bins_y)))
                tokens.append(idx_x * n_bins_y + idx_y)
            discretized[str(aid)] = tokens
        return discretized
