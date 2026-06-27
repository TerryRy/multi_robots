"""
OpenVLA-7B Policy: uses OpenVLA's pretrained Llama2-7B backbone
(replaces SigLIP visual encoder with custom MLP feature encoder).

Also supports MockVLAPolicy for fast interface testing.
"""
import math
import torch
import torch.nn as nn

# ---- Mock Policy (fast interface testing) ----

class MockVLAPolicy:
    def __init__(self, action_mode="continuous"):
        self.action_mode = action_mode

    def predict(self, text_prompt, features_dict):
        agents = features_dict.get("agents", [])
        num_agents = features_dict.get("num_agents", 0)
        waypoints = {}
        for i in range(min(num_agents, len(agents))):
            feat = agents[i]
            px, py = feat[0], feat[1]
            dx, dy = feat[12], feat[13]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.01:
                waypoints[str(i)] = [(0.0, 0.0)] * 8
                continue
            sx, sy = dx / 8.0, dy / 8.0
            wps, cx, cy = [], px, py
            for _ in range(8):
                cx, cy = cx + sx, cy + sy
                wps.append((round(cx, 4), round(cy, 4)))
            waypoints[str(i)] = wps
        return waypoints


# ---- MLP Feature Encoder ----

class AgentFeatureEncoder(nn.Module):
    """Maps per-agent 55-dim features -> LLM embedding space."""

    def __init__(self, input_dim=55, hidden_dim=512, output_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, agent_features):
        b, n, d = agent_features.shape
        return self.mlp(agent_features.reshape(b * n, d)).reshape(b, n, -1)


# ---- Action Head ----

class WaypointActionHead(nn.Module):
    def __init__(self, hidden_dim=4096, chunk_size=8, num_agents=12,
                 action_mode="continuous",
                 n_bins_x=25, n_bins_y=17, max_dx=0.6, max_dy=0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.action_mode = action_mode
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y
        self.max_dx = max_dx
        self.max_dy = max_dy

        if action_mode == "discrete":
            self.head = nn.Linear(hidden_dim, chunk_size * n_bins_x * n_bins_y)
        else:
            self.head = nn.Linear(hidden_dim, chunk_size * 2)

    def forward(self, hidden_states):
        return self.head(hidden_states)

    def decode_to_waypoints(self, logits, agent_positions, agent_headings):
        batch_size = logits.shape[0]
        num_agents = min(logits.shape[1], len(agent_positions))
        all_waypoints = []
        for b in range(batch_size):
            batch_wps = {}
            for a in range(num_agents):
                px, py = agent_positions[a]
                heading = agent_headings[a]
                raw = logits[b, a].reshape(self.chunk_size, 2)
                wps = [(raw[i, 0].item(), raw[i, 1].item()) for i in range(self.chunk_size)]
                global_wps = []
                for dx, dy in wps:
                    gx = px + dx * math.cos(heading) - dy * math.sin(heading)
                    gy = py + dx * math.sin(heading) + dy * math.cos(heading)
                    global_wps.append((gx, gy))
                batch_wps[str(a)] = global_wps
            all_waypoints.append(batch_wps)
        return all_waypoints


# ---- OpenVLA-7B Policy ----

class OpenVLAPolicy:
    """
    OpenVLA-7B backbone + custom MLP encoder + action head.

    Architecture:
      [55-dim features] -> MLP -> [4096-dim agent tokens]
      [text prompt]     -> tokenizer -> text tokens
      [agent tokens + text tokens] -> Llama2-7B (pretrained on Open X-Embodiment)
      [last hidden state of agent tokens] -> action head -> waypoints
    """

    def __init__(self, model_id="openvla/openvla-7b", device="cpu",
                 action_mode="continuous", use_mock=False):
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
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading OpenVLA-7B from {self.model_id}...")

        # Load the full VLM, extract the language model backbone
        full_model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device if self.device != "cpu" else None,
            low_cpu_mem_usage=True,
        )

        # Extract Llama2-7B backbone (frozen)
        if hasattr(full_model, 'language_model'):
            self.llm = full_model.language_model
        elif hasattr(full_model, 'model'):
            self.llm = full_model.model
        else:
            raise RuntimeError("Cannot find language model in OpenVLA checkpoint")

        del full_model                         # free vision encoder memory
        self.llm.to(self.device)

        hidden_dim = self.llm.config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # MLP encoder: 55-dim agent features -> same dim as LLM embeddings
        self.feature_encoder = AgentFeatureEncoder(
            input_dim=55, hidden_dim=512, output_dim=hidden_dim
        ).to(device=self.llm.device, dtype=self.llm.dtype)

        # Action head: LLM hidden -> waypoint predictions
        self.action_head = WaypointActionHead(
            hidden_dim=hidden_dim, chunk_size=8, action_mode=self.action_mode,
        ).to(device=self.llm.device, dtype=self.llm.dtype)

        self._ready = True
        print(f"OpenVLA-7B loaded. LLM hidden_dim={hidden_dim}")

    def predict(self, text_prompt, features_dict):
        if self.use_mock and hasattr(self, '_mock'):
            return self._mock.predict(text_prompt, features_dict)

        if not self._ready:
            n = features_dict.get("num_agents", 0)
            return {str(i): [(0.0, 0.0)] * 8 for i in range(n)}

        with torch.no_grad():
            agent_features = torch.tensor(
                features_dict["agents"], dtype=self.llm.dtype,
                device=self.llm.device
            ).unsqueeze(0)
            agent_embeds = self.feature_encoder(agent_features)

            inputs = self.tokenizer(
                text_prompt, return_tensors="pt", padding=True,
                truncation=True, max_length=1024
            ).to(self.llm.device)
            text_embeds = self.llm.get_input_embeddings()(inputs["input_ids"])

            combined = torch.cat([agent_embeds, text_embeds], dim=1)
            outputs = self.llm(inputs_embeds=combined, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, :agent_embeds.shape[1], :]
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


def load_openvla_policy(model_id="openvla/openvla-7b", device="cpu",
                        action_mode="continuous"):
    return OpenVLAPolicy(model_id=model_id, device=device, action_mode=action_mode)
