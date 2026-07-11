import os
import random
from math import cos, sin, atan2, pi as math_pi
from vla.state_serializer import StateSerializer
from vla.waypoint_tracker import WaypointTracker
from vla.data_collector import DataCollector
from vla.renderer import SimulatorRenderer
from agents.agent_state_machine import AgentState

DEBUG = False


class VLAController:

    def __init__(self, agents, b2_objects, config):
        self.agents = agents
        self.b2_objects = b2_objects
        self.config = config

        self.chunk_size = config.get("chunk_size", 8)
        self.steps_per_sec = config.get("steps_per_sec", 60)
        self.dt = 1.0 / self.steps_per_sec
        self.use_mock = config.get("use_mock", True)
        self.collect_data = config.get("collect_data", False)
        self.action_mode = config.get("action_mode", "continuous")

        self.serializer = StateSerializer()
        self.renderer = SimulatorRenderer(img_size=224)
        self.trackers = {}
        for agent in agents:
            self.trackers[agent.id] = WaypointTracker(agent, self.dt)

        self._model = None
        if self.use_mock:
            from vla.model_loader import load_mock_policy
            self._model = load_mock_policy(self.action_mode)
        else:
            ckpt = config.get("checkpoint_dir", "")
            fast_ckpt = os.path.join(ckpt, "fast_lm.pt") if ckpt else ""
            if os.path.isfile(fast_ckpt):
                from vla.model_loader import FastVLAPolicy
                device = config.get("device", "cpu")
                print(f"  Fast checkpoint detected: {fast_ckpt}")
                self._model = FastVLAPolicy(
                    encoder_path=os.path.join(ckpt, "encoder.pt"),
                    fast_lm_path=os.path.join(ckpt, "fast_lm.pt"),
                    action_head_path=os.path.join(ckpt, "diffusion_head.pt"),
                    device=device,
                    action_mode=self.action_mode,
                    chunk_size=self.chunk_size,
                )
            else:
                from vla.model_loader import load_openvla_policy
                model_id = config.get("model_id", "openvla/openvla-7b")
                device = config.get("device", "cpu")
                self._model = load_openvla_policy(
                    model_id=model_id, device=device,
                    action_mode=self.action_mode, chunk_size=self.chunk_size,
                )

        if not self.use_mock and config.get("checkpoint_dir") and not os.path.isfile(
            os.path.join(config["checkpoint_dir"], "fast_lm.pt")):
            ckpt = config["checkpoint_dir"]
            print(f"Loading VLA checkpoint: {ckpt}")
            self._model.load_checkpoint(ckpt)

        self._data_collector = None
        if self.collect_data:
            self._data_collector = DataCollector(
                action_mode=self.action_mode,
                chunk_size=self.chunk_size,
                collect_every_n_steps=self.chunk_size,
                renderer=self.renderer,
            )

        self._step_counter = 0
        self._last_vla_call_step = -999
        self._pos_history = {agent.id: [] for agent in agents}
        self._pending_state = []
        self._mix = self._parse_mix(config.get("mix_ratios", "expert:1.0"))
        self._current_controller = None
        self._stride = config.get("stride", 1)

    def _parse_mix(self, mix_str):
        result = {}
        for part in mix_str.split(","):
            kv = part.strip().split(":")
            result[kv[0]] = float(kv[1])
        return result

    def _pick_controller(self):
        r = random.random()
        cum = 0.0
        for name in ["expert", "policy", "random"]:
            cum += self._mix.get(name, 0.0)
            if r < cum:
                return name
        return "expert"

    def step(self, simulator):
        self._step_counter += 1

        if self.collect_data:
            for agent in self.agents:
                pos = agent.position
                px = pos.x if hasattr(pos, 'x') else pos[0]
                py = pos.y if hasattr(pos, 'y') else pos[1]
                self._pos_history[agent.id].append((px, py))

            if self._step_counter == 1:
                self._current_controller = self._pick_controller()
                if DEBUG:
                    print(f"  Controller: {self._current_controller}")

            if self._current_controller == "policy" and self._model is not None:
                self._run_policy_control(simulator)
            elif self._current_controller == "random":
                self._run_random_control(simulator)

            if self._step_counter % self.chunk_size == 0:
                text_prompt, features, agents_data = self.serializer.serialize(simulator)
                img = self.renderer.render(simulator)
                self._pending_state.append({
                    "step": self._step_counter,
                    "text_prompt": text_prompt,
                    "features": features,
                    "agents_data": agents_data,
                    "img": img,
                    "agent_ids": [ag["id"] for ag in agents_data],
                })

            while self._pending_state:
                ps = self._pending_state[0]
                start = ps["step"]
                total_steps = self.chunk_size * self._stride
                if len(self._pos_history[ps["agent_ids"][0]]) < start + total_steps + 1:
                    break
                self._pending_state.pop(0)
                waypoints = {}
                for aid in ps["agent_ids"]:
                    hist = self._pos_history[aid]
                    raw = hist[start + 1:start + total_steps + 1]
                    wps = [raw[i] for i in range(0, total_steps, self._stride)]
                    if len(wps) == self.chunk_size:
                        waypoints[str(aid)] = wps
                if waypoints and self._data_collector is not None:
                    self._data_collector.collect(
                        ps["text_prompt"], ps["features"], ps["agents_data"],
                        waypoints, simulator=simulator,
                    )
            return

        needs_inference = (self._step_counter - self._last_vla_call_step) >= self.chunk_size
        needs_inference |= any(not t.has_waypoints() for t in self.trackers.values())

        if needs_inference and self._model is not None:
            text_prompt, features, agents_data = self.serializer.serialize(simulator)
            img = self.renderer.render(simulator)
            vla_output = self._model.predict(text_prompt, features, images=img)
            self._dispatch_waypoints(vla_output, agents_data, simulator)

            if self._step_counter <= 5 or self._step_counter % 60 == 0:
                for ag in agents_data:
                    trk = self.trackers.get(ag["id"])
                    has_wps = trk.has_waypoints() if trk else False
                    wp_str = ""
                    if has_wps:
                        wp = trk.waypoint_queue[0]
                        d = ag["destination"]
                        if d:
                            dd = ((wp[0]-d[0])**2 + (wp[1]-d[1])**2)**0.5
                            wp_str = f" 1st_wp=({wp[0]:.1f},{wp[1]:.1f}) dist2dest={dd:.1f}"
                    print(f"  VLA: Agent {ag['id']} pos=({ag['position'][0]:.1f},{ag['position'][1]:.1f}) "
                          f"dest={ag['destination']} has_wps={has_wps}{wp_str}")

            self._last_vla_call_step = self._step_counter

        for agent in self.agents:
            tracker = self.trackers.get(agent.id)
            if tracker is None:
                continue

            if hasattr(agent, 'state') and agent.state not in (AgentState.CRUISE, AgentState.PREQUEUE, AgentState.QUEUING):
                agent.linear_velocity = (0.0, 0.0)
                continue

            if not tracker.has_waypoints():
                continue

            position = agent.position
            if hasattr(position, 'x'):
                from geometry import Point
                position = Point(position.x, position.y)
            vx, vy = tracker.compute_velocity(position)
            agent.linear_velocity = (vx, vy)

    def _run_policy_control(self, simulator):
        text_prompt, features, agents_data = self.serializer.serialize(simulator)
        img = self.renderer.render(simulator)
        vla_output = self._model.predict(text_prompt, features, images=img)
        self._dispatch_waypoints(vla_output, agents_data, simulator)

    def _run_random_control(self, simulator):
        for agent in self.agents:
            import random as rnd
            dest = agent.destination_location
            if dest is not None and hasattr(dest, 'x'):
                dx = dest.x - agent.position.x
                dy = dest.y - agent.position.y
                goal_angle = atan2(dy, dx)
            else:
                goal_angle = rnd.uniform(0, 2 * math_pi)
            angle = goal_angle + rnd.uniform(-0.5, 0.5)
            speed = rnd.uniform(0, agent.cruise_speed * 0.5)
            agent.linear_velocity = (cos(angle) * speed, sin(angle) * speed)

    def _dispatch_waypoints(self, vla_output, agents_data, simulator):
        num_agents = len(agents_data)
        for i in range(min(num_agents, len(self.agents))):
            aid = agents_data[i]["id"]
            key = str(aid)
            waypoints = vla_output.get(key, [])

            if self.action_mode == "discrete":
                waypoints = self._discrete_to_global(waypoints, agents_data[i])

            global_waypoints = []
            for step_i, wp in enumerate(waypoints):
                if len(wp) < 2:
                    continue
                wx, wy = wp[0], wp[1]
                global_waypoints.append((wx, wy))

            if len(global_waypoints) == 0:
                continue
            self.trackers[aid].set_waypoints(global_waypoints)

    def _discrete_to_global(self, tokens, agent_info, n_bins_x=25, n_bins_y=17,
                             max_dx=0.6, max_dy=0.4):
        waypoints = []
        px, py = agent_info["position"]
        heading = agent_info["angle_rad"]
        for token in tokens:
            idx_x = token // n_bins_y
            idx_y = token % n_bins_y
            dx = (idx_x / n_bins_x) * 2 * max_dx - max_dx
            dy = (idx_y / n_bins_y) * 2 * max_dy - max_dy
            gx = px + dx * cos(heading) - dy * sin(heading)
            gy = py + dx * sin(heading) + dy * cos(heading)
            waypoints.append((gx, gy))
        return waypoints

    def flush_data(self):
        if self._data_collector is not None:
            self._data_collector.flush()
        self.renderer.close()
