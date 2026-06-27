"""
Model loader: supports MockVLAPolicy (for testing) and OpenVLAPolicy (real model).
OpenVLAPolicy uses an LLM backbone + MLP feature encoder + action head.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Mock Policy (kept for fast interface testing) ----

class MockVLAPolicy:
    """Outputs straight-line waypoints toward agent destination."""

    def __init__(self, action_mode="continuous"):
        self.action_mode = action_mode

    def predict(self, text_prompt, features_dict):
        agents = features_dict.get("agents", [])
        num_agents = features_dict.get("num_agents", 0)
        waypoints = {}

        for i in range(min(num_agents, len(agents))):
            feat = agents[i]
            px, py = feat[0], feat[1]
            dx_goal, dy_goal = feat[12], feat[13]
            dist = math.sqrt(dx_goal * dx_goal + dy_goal * dy_goal)
            if dist < 0.01:
                waypoints[str(i)] = [(0.0, 0.0)] * 8
                continue
            step_x = dx_goal / 8.0
            step_y = dy_goal / 8.0
            wps = []
            cx, cy = px, py
            for _ in range(8):
                cx += step_x
                cy += step_y
                wps.append((round(cx, 4), round(cy, 4)))
            waypoints[str(i)] = wps
        return waypoints


# ---- MLP Feature Encoder ----

class AgentFeatureEncoder(nn.Module):
    """Maps per-agent 55-dim features → LLM embedding space."""

    def __init__(self, input_dim=55, hidden_dim=512, output_dim=1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, agent_features):
        """
        agent_features: (batch_size, max_agents, input_dim)
        Returns: (batch_size, max_agents, output_dim)
        """
        b, n, d = agent_features.shape
        flat = agent_features.reshape(b * n, d)
        encoded = self.mlp(flat)
        return encoded.reshape(b, n, -1)


# ---- Action Head ----

class WaypointActionHead(nn.Module):
    """
    Maps LLM hidden states → waypoint predictions.
    Supports:
      - discrete: class logits per waypoint step
      - continuous: raw (Δx, Δy) regression
    """

    def __init__(self, hidden_dim=1536, chunk_size=8, num_agents=12,
                 action_mode="continuous",
                 n_bins_x=25, n_bins_y=17, max_dx=0.6, max_dy=0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.num_agents = num_agents
        self.action_mode = action_mode
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.num_classes = n_bins_x * n_bins_y

        if action_mode == "discrete":
            self.head = nn.Linear(hidden_dim, self.chunk_size * self.num_classes)
        else:
            self.head = nn.Linear(hidden_dim, self.chunk_size * 2)

    def forward(self, hidden_states):
        """
        hidden_states: (batch_size, max_agents, hidden_dim) or (batch, seq_len, hidden_dim)
        Returns: waypoint dict per batch item
        """
        logits = self.head(hidden_states)
        return logits

    def decode_to_waypoints(self, logits, agent_positions, agent_headings):
        """
        Convert raw logits → waypoints in global frame.
        agent_positions: list of (x, y)
        agent_headings: list of radians
        """
        batch_size = logits.shape[0]
        num_agents = min(logits.shape[1], len(agent_positions))

        all_waypoints = []
        for b in range(batch_size):
            batch_wps = {}
            for a in range(num_agents):
                px, py = agent_positions[a]
                heading = agent_headings[a]

                if self.action_mode == "discrete":
                    agent_logits = logits[b, a].reshape(self.chunk_size, self.num_classes)
                    class_ids = agent_logits.argmax(dim=-1)
                    wps = self._discrete_class_to_delta(class_ids)
                else:
                    agent_logits = logits[b, a].reshape(self.chunk_size, 2)
                    wps = [(agent_logits[i, 0].item(), agent_logits[i, 1].item())
                           for i in range(self.chunk_size)]

                global_wps = []
                for dx, dy in wps:
                    gx = px + dx * math.cos(heading) - dy * math.sin(heading)
                    gy = py + dx * math.sin(heading) + dy * math.cos(heading)
                    global_wps.append((gx, gy))
                batch_wps[str(a)] = global_wps
            all_waypoints.append(batch_wps)
        return all_waypoints

    def _discrete_class_to_delta(self, class_ids):
        wps = []
        for cid in class_ids:
            idx_x = cid // self.n_bins_y
            idx_y = cid % self.n_bins_y
            dx = (idx_x / self.n_bins_x) * 2 * self.max_dx - self.max_dx
            dy = (idx_y / self.n_bins_y) * 2 * self.max_dy - self.max_dy
            wps.append((dx, dy))
        return wps


# ---- OpenVLA-compatible Policy ----

class OpenVLAPolicy:
    """
    LLM backbone + MLP encoder + action head.
    Compatible with SmolLM2, Qwen2.5, Llama, etc.
    Uses the pipeline: text_prompt + features → LLM → action head → waypoints.
    """

    def __init__(self, model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
                 device="cpu", action_mode="continuous", use_mock=False):
        self.model_id = model_id
        self.device = device
        self.action_mode = action_mode
        self.use_mock = use_mock

        if use_mock:
            self._mock = MockVLAPolicy(action_mode)
            self._ready = True
            return

        self._ready = False
        self._load_model()

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device if self.device == "cpu" else None,
        )
        hidden_dim = self.llm.config.hidden_size

        self.feature_encoder = AgentFeatureEncoder(
            input_dim=55, hidden_dim=512, output_dim=hidden_dim
        ).to(self.llm.device)

        self.action_head = WaypointActionHead(
            hidden_dim=hidden_dim, chunk_size=8, action_mode=self.action_mode,
        ).to(self.llm.device)

        self._ready = True

    def predict(self, text_prompt, features_dict):
        if self.use_mock:
            return self._mock.predict(text_prompt, features_dict)

        if not self._ready:
            return {str(i): [(0.0, 0.0)] * 8 for i in range(features_dict.get("num_agents", 0))}

        with torch.no_grad():
            agent_features = torch.tensor(features_dict["agents"], dtype=torch.float32,
                                          device=self.llm.device).unsqueeze(0)
            agent_embeds = self.feature_encoder(agent_features)

            inputs = self.tokenizer(
                text_prompt, return_tensors="pt", padding=True, truncation=True,
                max_length=1024
            ).to(self.llm.device)
            text_embeds = self.llm.get_input_embeddings()(inputs["input_ids"])

            combined_sequence = torch.cat([agent_embeds, text_embeds], dim=1)

            outputs = self.llm(inputs_embeds=combined_sequence, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, :agent_embeds.shape[1], :]  # agent tokens only

            logits = self.action_head(hidden)

            num_agents = features_dict.get("num_agents", 0)
            positions = [(agent_features[0, i, 0].item(), agent_features[0, i, 1].item())
                         for i in range(num_agents)]
            headings = [math.atan2(agent_features[0, i, 3].item(), agent_features[0, i, 2].item())
                         for i in range(num_agents)]

            waypoints_list = self.action_head.decode_to_waypoints(logits, positions, headings)
            return waypoints_list[0] if waypoints_list else {}


def load_mock_policy(action_mode="continuous"):
    return MockVLAPolicy(action_mode=action_mode)


def load_openvla_policy(model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
                        device="cpu", action_mode="continuous"):
    return OpenVLAPolicy(model_id=model_id, device=device, action_mode=action_mode)
