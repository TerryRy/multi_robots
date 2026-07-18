import random
import math
from math import cos, sin, atan2, pi as math_pi
from vla.state_serializer import StateSerializer
from vla.data_collector import DataCollector
from vla.renderer import SimulatorRenderer
from vla.wheel_kinematics import motion_to_control, control_to_motion
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

        self.serializer = StateSerializer()
        self.renderer = SimulatorRenderer(img_size=224)

        # Action buffers: 8 (v_forward, omega) pairs per agent
        self._action_buffer = {agent.id: [] for agent in agents}
        # Track heading for omega computation during inference
        self._heading = {agent.id: 0.0 for agent in agents}

        self._model = None
        if self.use_mock:
            from vla.model_loader import load_mock_policy
            self._model = load_mock_policy(self.action_mode)
        else:
            from vla.model_loader import load_openvla_policy
            model_id = config.get("model_id", "openvla/openvla-7b")
            device = config.get("device", "cpu")
            self._model = load_openvla_policy(
                model_id=model_id, device=device,
                action_mode=self.action_mode, chunk_size=self.chunk_size,
            )

        self._model.goal_scale = config.get("goal_scale", 1.0)

        if not self.use_mock and config.get("checkpoint_dir"):
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
            self._prev_velocity = {agent.id: (0.0, 0.0) for agent in agents}
            self._prev_heading = {agent.id: 0.0 for agent in agents}
            # Record history of (v_forward, omega) per agent
            self._action_history = {agent.id: [] for agent in agents}
            self._pending_state = []

        self._step_counter = 0
        self._last_vla_call_step = -999
        self._mix = self._parse_mix(config.get("mix_ratios", "expert:1.0"))
        self._current_controller = None
        self._stride = config.get("stride", 1)

    @property
    def action_mode(self):
        return "continuous"

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

    def _record_action(self, simulator):
        """Record current (v_forward, omega) from expert planner motion."""
        for agent in self.agents:
            vx, vy = agent.linear_velocity
            prev_vx, prev_vy = self._prev_velocity.get(agent.id, (0.0, 0.0))
            prev_h = self._prev_heading.get(agent.id)
            v_fwd, omega = motion_to_control(vx, vy, prev_vx, prev_vy, self.dt, prev_h)
            self._action_history[agent.id].append((v_fwd, omega))
            self._prev_velocity[agent.id] = (vx, vy)
            body = self.b2_objects.get(agent.id)
            self._prev_heading[agent.id] = body.angle if body else atan2(vy, vx)

    def _apply_action(self, agent, v_fwd, omega):
        """Apply (v_forward, omega) to agent: set linear_velocity and heading."""
        body = self.b2_objects.get(agent.id)
        heading = body.angle if body else 0.0
        vx, vy, new_heading = control_to_motion(v_fwd, omega, heading, self.dt)
        agent.linear_velocity = (vx, vy)
        agent.speed = abs(v_fwd)
        agent.wheel_heading = new_heading
        if body:
            body.angle = new_heading

    def _refill_buffers(self, simulator):
        """Run model inference and fill action buffers for all agents."""
        text_prompt, features, agents_data = self.serializer.serialize(simulator)
        vla_output = self._model.predict(text_prompt, features)
        for agent in self.agents:
            key = str(agent.id)
            action_pairs = vla_output.get(key, [])
            if len(action_pairs) == self.chunk_size:
                # action_pairs is list of 8 (v, ω) pairs
                self._action_buffer[agent.id] = list(action_pairs)

    def step(self, simulator):
        self._step_counter += 1

        if self.collect_data:
            self._record_action(simulator)

            if self._step_counter % self.chunk_size == 1:
                self._current_controller = self._pick_controller()
                if DEBUG:
                    print(f"  Controller: {self._current_controller}")

            if self._current_controller == "policy" and self._model is not None:
                if self._step_counter % self.chunk_size == 1:
                    self._refill_buffers(simulator)
                for agent in self.agents:
                    buf = self._action_buffer.get(agent.id)
                    if buf:
                        v_fwd, omega = buf.pop(0)
                        self._apply_action(agent, v_fwd, omega)
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

            # Extract (v_forward, omega) sequences from history
            while self._pending_state:
                ps = self._pending_state[0]
                start = ps["step"]
                total_steps = self.chunk_size * self._stride
                if len(self._action_history[ps["agent_ids"][0]]) < start + total_steps + 1:
                    break
                self._pending_state.pop(0)
                action_seq = {}
                for aid in ps["agent_ids"]:
                    hist = self._action_history[aid]
                    raw = hist[start + 1:start + total_steps + 1]
                    samples = [raw[i] for i in range(0, total_steps, self._stride)]
                    if len(samples) == self.chunk_size:
                        action_seq[str(aid)] = samples
                if action_seq and self._data_collector is not None:
                    self._data_collector.collect(
                        ps["text_prompt"], ps["features"], ps["agents_data"],
                        action_seq, simulator=simulator,
                    )
            return

        # === NON-COLLECTION: normal VLA inference ===
        needs_inference = (self._step_counter - self._last_vla_call_step) >= self.chunk_size
        needs_inference |= any(len(buf) == 0 for buf in self._action_buffer.values())

        if needs_inference and self._model is not None:
            self._refill_buffers(simulator)

            if self._step_counter <= 5 or self._step_counter % 60 == 0:
                for agent in self.agents:
                    buf = self._action_buffer.get(agent.id, [])
                    pos = agent.position
                    body = self.b2_objects.get(agent.id)
                    heading = body.angle if body else 0.0
                    first = buf[0] if buf else (0, 0)
                    print(f"  VLA: Agent {agent.id} pos=({pos.x:.1f},{pos.y:.1f}) "
                          f"v={first[0]:.3f} ω={first[1]:.3f} heading={math.degrees(heading):.0f} "
                          f"buf={len(buf)}")

            self._last_vla_call_step = self._step_counter

        for agent in self.agents:
            if hasattr(agent, 'state') and agent.state not in (AgentState.CRUISE, AgentState.PREQUEUE, AgentState.QUEUING):
                agent.linear_velocity = (0.0, 0.0)
                agent.speed = 0.0
                continue

            buf = self._action_buffer.get(agent.id)
            if not buf:
                continue
            v_fwd, omega = buf.pop(0)
            self._apply_action(agent, v_fwd, omega)

    def _run_random_control(self, simulator):
        for agent in self.agents:
            import random as rnd
            v_fwd = rnd.uniform(0, 1.0)
            omega = rnd.uniform(-2.0, 2.0)
            self._apply_action(agent, v_fwd, omega)

    def flush_data(self):
        if self._data_collector is not None:
            self._data_collector.flush()
        self.renderer.close()
