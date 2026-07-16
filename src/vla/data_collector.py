import json
import os
import time
from PIL import Image as PILImage


class DataCollector:
    """
    Records (state, hand-written expert waypoint) pairs for VLA training.
    Saves top-down images as PNG alongside JSONL records.
    """

    def __init__(self, save_dir="data/trajectories_wheel", action_mode="continuous",
                 chunk_size=8, collect_every_n_steps=4, renderer=None):
        self.save_dir = save_dir
        self.action_mode = action_mode
        self.chunk_size = chunk_size
        self.collect_every_n_steps = collect_every_n_steps
        self.renderer = renderer
        self._step_counter = 0
        self._buffer = []
        self._session_id = f"session_{int(time.time())}"
        self._img_dir = os.path.join(save_dir, self._session_id)
        os.makedirs(self._img_dir, exist_ok=True)
        self._frame_counter = 0

    def collect(self, text_prompt, features, agents_data, expert_waypoints_dict,
                simulator=None):
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
                str(aid): [(round(v, 4), round(w, 4)) for v, w in wps]
                for aid, wps in expert_waypoints_dict.items()
            }

        image_path = None
        if self.renderer is not None and simulator is not None:
            img_array = self.renderer.render(simulator)
            img = PILImage.fromarray(img_array)
            fname = f"frame_{self._frame_counter:06d}.png"
            img.save(os.path.join(self._img_dir, fname))
            image_path = os.path.join(self._session_id, fname)
            self._frame_counter += 1

        record = {
            "text_prompt": text_prompt,
            "features": features,
            "target_action": target,
            "image_path": image_path,
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
