import math
import torch
import torch.nn as nn

from vla.diffusion_head import DiffusionActionHead, compute_diffusion_loss, ddim_sample


class MockVLAPolicy:
    def __init__(self, action_mode="continuous"):
        self.action_mode = action_mode

    def predict(self, text_prompt, features_dict, images=None):
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


class AgentFeatureEncoder(nn.Module):
    def __init__(self, input_dim=55, hidden_dim=512, output_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, agent_features):
        b, n, d = agent_features.shape
        x = self.mlp(agent_features.reshape(b * n, d)).reshape(b, n, -1)
        return self.norm(x)


class OpenVLAPolicy:
    """
    Full OpenVLA-7B pipeline with image + text + structured features.

    Architecture:
      Image (224x224) → SigLIP (frozen) → visual tokens → Projector (frozen) → [S, 4096]
      55-dim features  → MLP Encoder (trainable)                    → agent tokens [N, 4096]
      text prompt      → tokenizer (frozen) → text tokens [T]
      → concat [vis_tokens + agent_tokens + text_tokens] → Llama2-7B (LoRA)
      → hidden states for agent tokens → DiffusionActionHead (trainable) → waypoints
    """

    def __init__(self, model_id="openvla/openvla-7b", device="cpu",
                 action_mode="continuous", use_mock=False,
                 chunk_size=8, diffusion_steps=50, max_agents=14):
        self.model_id = model_id
        self.device = device
        self.action_mode = action_mode
        self.use_mock = use_mock
        self.chunk_size = chunk_size
        self.diffusion_steps = diffusion_steps
        self.max_agents = max_agents

        if use_mock:
            self._mock = MockVLAPolicy(action_mode)
            self._ready = True
            return

        self._ready = False
        self._load_model()

    def _load_model(self):
        from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor

        print(f"Loading OpenVLA-7B from {self.model_id}...")

        full_model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device if self.device != "cpu" else None,
            low_cpu_mem_usage=True,
        )

        for attr in ['vision_encoder', 'vision_tower', 'vision_backbone']:
            if hasattr(full_model, attr):
                self.vision_encoder = getattr(full_model, attr)
                break
        else:
            raise RuntimeError(
                f"Cannot find vision_encoder in OpenVLA checkpoint. "
                f"Available: {[a for a in dir(full_model) if not a.startswith('_')]}"
            )

        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        self.vision_encoder.eval()

        for attr in ['projector', 'connector', 'vision_projector']:
            if hasattr(full_model, attr):
                self.projector = getattr(full_model, attr)
                break
        else:
            raise RuntimeError(
                f"Cannot find projector in OpenVLA checkpoint. "
                f"Available: {[a for a in dir(full_model) if not a.startswith('_')]}"
            )

        for p in self.projector.parameters():
            p.requires_grad = False
        self.projector.eval()

        for attr in ['language_model', 'model', 'llm', 'llm_backbone', 'lm_backbone']:
            if hasattr(full_model, attr):
                self.llm = getattr(full_model, attr)
                break
        else:
            raise RuntimeError(
                f"Cannot find language model in OpenVLA checkpoint. "
                f"Available: {[a for a in dir(full_model) if not a.startswith('_')]}"
            )

        del full_model
        self.llm.to(self.device)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        hidden_dim = self.llm.config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.feature_encoder = AgentFeatureEncoder(
            input_dim=55, hidden_dim=512, output_dim=hidden_dim,
        ).to(device=self.device, dtype=self.llm.dtype)

        self.diffusion_head = DiffusionActionHead(
            hidden_dim=hidden_dim, chunk_size=self.chunk_size,
        ).to(device=self.device, dtype=self.llm.dtype)

        import torchvision.transforms as T
        self.img_preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self._ready = True
        print(f"OpenVLA-7B loaded. hidden_dim={hidden_dim}, "
              f"trainable: encoder={sum(p.numel() for p in self.feature_encoder.parameters())/1e3:.0f}K, "
              f"diffusion_head={sum(p.numel() for p in self.diffusion_head.parameters())/1e3:.0f}K")

    @torch.no_grad()
    def _encode_images(self, images):
        from PIL import Image
        import numpy as np
        if isinstance(images, np.ndarray):
            images = Image.fromarray(images)
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, list) and all(isinstance(x, np.ndarray) for x in images):
            images = [Image.fromarray(x) for x in images]

        pixel_values = torch.stack([self.img_preprocess(img) for img in images])
        pixel_values = torch.cat([pixel_values, pixel_values], dim=1)
        pixel_values = pixel_values.to(device=self.device, dtype=self.llm.dtype)

        if hasattr(self.vision_encoder, 'pixel_values') or hasattr(self.vision_encoder, 'forward'):
            visual_feat = self.vision_encoder(pixel_values)
        else:
            visual_feat = self.vision_encoder(pixel_values.to(self.llm.dtype))

        if isinstance(visual_feat, tuple):
            visual_feat = visual_feat[0]

        if len(visual_feat.shape) == 3:
            pass
        elif len(visual_feat.shape) == 4:
            b, c, h, w = visual_feat.shape
            visual_feat = visual_feat.reshape(b, c, -1).permute(0, 2, 1)

        visual_tokens = self.projector(visual_feat)
        return visual_tokens

    def forward(self, images, text_prompt, agent_features):
        """
        Args:
            images: PIL Image, numpy array, or list thereof [B, H, W, 3]
            text_prompt: str or list of str
            agent_features: [B, N, 55] tensor
        Returns:
            hidden_states: [B, N, hidden_dim] for agent tokens
        """
        B = agent_features.shape[0]

        visual_tokens = self._encode_images(images)

        agent_embeds = self.feature_encoder(agent_features)

        if isinstance(text_prompt, str):
            text_prompt = [text_prompt]

        encoded = self.tokenizer(
            text_prompt, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        ).to(self.device)
        text_embeds = self.llm.get_input_embeddings()(encoded["input_ids"])

        combined = torch.cat([visual_tokens, agent_embeds, text_embeds], dim=1)

        outputs = self.llm(
            inputs_embeds=combined,
            output_hidden_states=True,
        )

        n_agent = agent_embeds.shape[1]
        hidden = outputs.hidden_states[-1][:, visual_tokens.shape[1]:visual_tokens.shape[1] + n_agent, :]
        return hidden

    def predict(self, text_prompt, features_dict, images=None):
        if self.use_mock and hasattr(self, '_mock'):
            return self._mock.predict(text_prompt, features_dict)

        if not self._ready:
            n = features_dict.get("num_agents", 0)
            return {str(i): [(0.0, 0.0)] * self.chunk_size for i in range(n)}

        agent_features = torch.tensor(
            features_dict["agents"], dtype=self.llm.dtype,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            hidden = self.forward(images, text_prompt, agent_features)
            num_agents = features_dict.get("num_agents", 0)
            hidden = hidden[:, :num_agents, :]

            waypoints_tensor = ddim_sample(
                self.diffusion_head, hidden,
                ddim_steps=self.diffusion_steps,
                eta=0.0,
            )

            num_agents = features_dict.get("num_agents", 0)
            positions = [
                (agent_features[0, i, 0].item(), agent_features[0, i, 1].item())
                for i in range(num_agents)
            ]
            headings = [
                math.atan2(agent_features[0, i, 3].item(), agent_features[0, i, 2].item())
                for i in range(num_agents)
            ]

            result = {}
            for i in range(num_agents):
                px, py = positions[i]
                heading = headings[i]
                wps = []
                for j in range(self.chunk_size):
                    local_dx = waypoints_tensor[0, i, j * 2].item()
                    local_dy = waypoints_tensor[0, i, j * 2 + 1].item()
                    gx = px + local_dx * math.cos(heading) - local_dy * math.sin(heading)
                    gy = py + local_dx * math.sin(heading) + local_dy * math.cos(heading)
                    wps.append((round(gx, 4), round(gy, 4)))
                result[str(i)] = wps
            return result


def load_mock_policy(action_mode="continuous"):
    return MockVLAPolicy(action_mode=action_mode)


def load_openvla_policy(model_id="openvla/openvla-7b", device="cpu",
                        action_mode="continuous", chunk_size=8):
    return OpenVLAPolicy(
        model_id=model_id, device=device, action_mode=action_mode,
        use_mock=False, chunk_size=chunk_size,
    )
