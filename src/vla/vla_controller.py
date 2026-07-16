import random
import math
from math import cos, sin, atan2, pi as math_pi
from vla.state_serializer import StateSerializer
from vla.data_collector import DataCollector
from vla.renderer import SimulatorRenderer
from vla.wheel_kinematics import wheels_to_velocity, velocity_to_wheels
from agents.agent_state_machine import AgentState

DEBUG = False
NUM_AGENTS = 5
NUM_WHEELS = NUM_AGENTS * 2


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

        # Wheel buffers: model outputs 10 wheels × 8 steps = 80 values
        self._wheel_buffers = {i: [] for i in range(NUM_WHEELS)}

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
            self._wheel_history = {i: [] for i in range(NUM_WHEELS)}
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

    def _compute_wheel_speeds(self, agent, prev_vx, prev_vy):
        """Convert current linear velocity to (left, right, mean_speed)."""
        vx, vy = agent.linear_velocity
        left, right = velocity_to_wheels(vx, vy, prev_vx, prev_vy, self.dt)
        mean_speed = (vx * vx + vy * vy) ** 0.5
        return left, right, mean_speed

    def _apply_wheel_pair(self, agent, left, right):
        """Apply (left,right) wheel pair to agent and update heading."""
        body = self.b2_objects.get(agent.id)
        heading = body.angle if body else 0.0
        vx, vy, omega = wheels_to_velocity(left, right, heading)
        new_heading = heading + omega * self.dt
        agent.linear_velocity = (vx, vy)
        agent.speed = (vx * vx + vy * vy) ** 0.5
        agent.wheel_heading = new_heading
        if body:
            body.angle = new_heading

    def _refill_wheel_buffers(self, simulator):
        """Run model inference and fill wheel buffers."""
        text_prompt, features, agents_data = self.serializer.serialize(simulator)
        vla_output = self._model.predict(text_prompt, features)
        for wi in range(NUM_WHEELS):
            key = str(wi)
            wheel_steps = vla_output.get(key, [])
            if len(wheel_steps) == self.chunk_size:
                self._wheel_buffers[wi] = list(wheel_steps)

    def _record_wheel_expert(self, simulator):
        """Record expert planner (vx,vy) as wheel velocities."""
        for agent in self.agents:
            prev_vx, prev_vy = self._prev_velocity.get(agent.id, (0.0, 0.0))
            left, right, _ = self._compute_wheel_speeds(agent, prev_vx, prev_vy)
            self._wheel_history[agent.id * 2].append(left)
            self._wheel_history[agent.id * 2 + 1].append(right)
            self._prev_velocity[agent.id] = agent.linear_velocity

    def _pop_and_apply(self, agent):
        """Pop one step from wheel buffers and apply to agent."""
        left_buf = self._wheel_buffers.get(agent.id * 2)
        right_buf = self._wheel_buffers.get(agent.id * 2 + 1)
        if left_buf and right_buf and len(left_buf) > 0 and len(right_buf) > 0:
            left = left_buf.pop(0)
            right = right_buf.pop(0)
            self._apply_wheel_pair(agent, left, right)

    def step(self, simulator):
        self._step_counter += 1

        if self.collect_data:
            self._record_wheel_expert(simulator)

            if self._step_counter % self.chunk_size == 1:
                self._current_controller = self._pick_controller()
                if DEBUG:
                    print(f"  Controller: {self._current_controller}")

            if self._current_controller == "policy" and self._model is not None:
                if self._step_counter % self.chunk_size == 1:
                    self._refill_wheel_buffers(simulator)
                if self._wheel_buffers.get(0):
                    for agent in self.agents:
                        self._pop_and_apply(agent)
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
                    "wheel_ids": list(range(NUM_WHEELS)),
                })

            while self._pending_state:
                ps = self._pending_state[0]
                start = ps["step"]
                total_steps = self.chunk_size * self._stride
                if len(self._wheel_history[0]) < start + total_steps + 1:
                    break
                self._pending_state.pop(0)
                wheel_seq = {}
                for wi in range(NUM_WHEELS):
                    hist = self._wheel_history[wi]
                    raw = hist[start + 1:start + total_steps + 1]
                    samples = [raw[i] for i in range(0, total_steps, self._stride)]
                    if len(samples) == self.chunk_size:
                        wheel_seq[str(wi)] = samples
                if wheel_seq and self._data_collector is not None:
                    self._data_collector.collect(
                        ps["text_prompt"], ps["features"], ps["agents_data"],
                        wheel_seq, simulator=simulator,
                    )
            return

        # === NON-COLLECTION: normal VLA inference ===
        needs_inference = (self._step_counter - self._last_vla_call_step) >= self.chunk_size
        needs_inference |= any(len(self._wheel_buffers[wi]) == 0 for wi in range(NUM_WHEELS))

        if needs_inference and self._model is not None:
            self._refill_wheel_buffers(simulator)

            if self._step_counter <= 5 or self._step_counter % 60 == 0:
                for agent in self.agents:
                    buf_l = self._wheel_buffers.get(agent.id * 2, [])
                    buf_r = self._wheel_buffers.get(agent.id * 2 + 1, [])
                    pos = agent.position
                    print(f"  VLA: Agent {agent.id} pos=({pos.x:.1f},{pos.y:.1f}) "
                          f"wheel_buf={len(buf_l)},{len(buf_r)}")

            self._last_vla_call_step = self._step_counter

        for agent in self.agents:
            if hasattr(agent, 'state') and agent.state not in (AgentState.CRUISE, AgentState.PREQUEUE, AgentState.QUEUING):
                agent.linear_velocity = (0.0, 0.0)
                agent.speed = 0.0
                continue
            self._pop_and_apply(agent)

    def _run_random_control(self, simulator):
        for agent in self.agents:
            import random as rnd
            left = rnd.uniform(-1.0, 1.0)
            right = rnd.uniform(-1.0, 1.0)
            self._apply_wheel_pair(agent, left, right)

    def flush_data(self):
        if self._data_collector is not None:
            self._data_collector.flush()
        self.renderer.close()
