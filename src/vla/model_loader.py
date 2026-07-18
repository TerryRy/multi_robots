import math
import torch
import torch.nn as nn

from vla.diffusion_head import DiffusionActionHead, compute_diffusion_loss, ddim_sample


class MockVLAPolicy:
    """Generates dummy (v_forward, omega) toward the goal for testing."""
    def __init__(self, action_mode="continuous"):
        self.action_mode = action_mode
        self.chunk_size = 8

    def predict(self, text_prompt, features_dict, images=None):
        agents = features_dict.get("agents", [])
        num_agents = features_dict.get("num_agents", len(agents))
        result = {}
        for i in range(min(num_agents, len(agents))):
            feat = agents[i]
            dx, dy = feat[12], feat[13]
            cos_h, sin_h = feat[2], feat[3]
            gx = dx * cos_h + dy * sin_h
            gy = -dx * sin_h + dy * cos_h
            dist = math.sqrt(gx * gx + gy * gy)
            if dist < 0.1:
                result[str(i)] = [(0.0, 0.0)] * self.chunk_size
                continue
            v_fwd = min(1.0, dist * 0.3)
            goal_angle = math.atan2(gy, gx)
            omega = max(-2.0, min(2.0, goal_angle * 2.0))
            result[str(i)] = [(v_fwd, omega)] * self.chunk_size
        return result


class FastProjector(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.config = type('obj', (object,), {'hidden_size': dim})
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.norm(self.proj(self.norm(x)) + x)


class FastVLAPolicy:
    def __init__(self, encoder_path=None, fast_lm_path=None, action_head_path=None,
                 device="cpu", action_mode="continuous", chunk_size=8, diffusion_steps=100,
                 hidden_dim=512):
        self.device = device
        self.action_mode = action_mode
        self.chunk_size = chunk_size
        self.diffusion_steps = diffusion_steps
        self.hidden_dim = hidden_dim

        self.feature_encoder = AgentFeatureEncoder(
            input_dim=59, hidden_dim=512, output_dim=hidden_dim,
        ).to(device)


        self.fast_lm = FastProjector(hidden_dim).to(device)

        self.diffusion_head = DiffusionActionHead(
            hidden_dim=hidden_dim, chunk_size=chunk_size,
        ).to(device)

        if encoder_path:
            self.feature_encoder.load_state_dict(
                torch.load(encoder_path, map_location=device, weights_only=True)
            )
            print(f"  Loaded fast encoder: {encoder_path}")
        if fast_lm_path:
            self.fast_lm.load_state_dict(
                torch.load(fast_lm_path, map_location=device, weights_only=True)
            )
            print(f"  Loaded fast_lm: {fast_lm_path}")
        if action_head_path:
            self.diffusion_head.load_state_dict(
                torch.load(action_head_path, map_location=device, weights_only=True)
            )
            print(f"  Loaded fast diffusion_head: {action_head_path}")

        self.feature_encoder.eval()
        self.fast_lm.eval()
        self.diffusion_head.eval()
        self._step_debug = 0

    def predict(self, text_prompt, features_dict, images=None):
        agent_features = torch.tensor(
            features_dict["agents"], dtype=torch.float32, device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            agent_embeds = self.feature_encoder(agent_features)
            hidden = self.fast_lm(agent_embeds)
            num_agents = features_dict.get("num_agents", 0)
            hidden = hidden[:, :num_agents, :]

            heading_cos = agent_features[0, :num_agents, 2:3]
            heading_sin = agent_features[0, :num_agents, 3:4]
            goal_global = agent_features[0, :num_agents, 12:14]
            goal_local_x = goal_global[:, 0:1] * heading_cos + goal_global[:, 1:2] * heading_sin
            goal_local_y = -goal_global[:, 0:1] * heading_sin + goal_global[:, 1:2] * heading_cos
            goal_feat = torch.cat([goal_local_x, goal_local_y], dim=-1)
            goal_feat = goal_feat / 40.0
            goal_feat = goal_feat * getattr(self, 'goal_scale', 1.0)

            waypoints_tensor = ddim_sample(
                self.diffusion_head, hidden,
                ddim_steps=self.diffusion_steps,
                eta=0.0,
                goal_features=goal_feat.unsqueeze(0),
            )

            V_MEAN, V_STD = 1.5, 1.5
            W_MEAN, W_STD = 0.0, 18.5
            result = {}
            for i in range(num_agents):
                pairs = []
                for j in range(self.chunk_size):
                    v_n = waypoints_tensor[0, i, j * 2].item()
                    w_n = waypoints_tensor[0, i, j * 2 + 1].item()
                    v = v_n * V_STD + V_MEAN
                    omega = w_n * W_STD + W_MEAN
                    pairs.append((v, omega))
                result[str(i)] = pairs
            return result


class AgentFeatureEncoder(nn.Module):
    def __init__(self, input_dim=59, hidden_dim=512, output_dim=4096):
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
      59-dim features  → MLP Encoder (trainable)                    → agent tokens [N, 4096]
      text prompt      → tokenizer (frozen) → text tokens [T]
      → concat [vis_tokens + agent_tokens + text_tokens] → Llama2-7B (LoRA)
      → hidden states for agent tokens → DiffusionActionHead (trainable) → waypoints
    """

    def __init__(self, model_id="openvla/openvla-7b", device="cpu",
                 action_mode="continuous", use_mock=False,
                 chunk_size=8, diffusion_steps=200, max_agents=14):
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

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            "device_map": self.device if self.device != "cpu" else None,
            "low_cpu_mem_usage": True,
        }
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

        full_model = AutoModelForVision2Seq.from_pretrained(self.model_id, **load_kwargs)

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
            input_dim=59, hidden_dim=512, output_dim=hidden_dim,
        ).to(device=self.device, dtype=self.llm.dtype)

        self.diffusion_head = DiffusionActionHead(
            hidden_dim=hidden_dim, chunk_size=self.chunk_size,
        ).to(device=self.device, dtype=torch.float32)

        import torchvision.transforms as T
        self.img_preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self._ready = True
        self._step_debug = 0
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

        vis_dtype = next(self.vision_encoder.parameters()).dtype
        pixel_values = torch.stack([self.img_preprocess(img) for img in images])
        pixel_values = torch.cat([pixel_values, pixel_values], dim=1)
        pixel_values = pixel_values.to(device=self.device, dtype=vis_dtype)

        visual_feat = self.vision_encoder(pixel_values)

        if isinstance(visual_feat, tuple):
            visual_feat = visual_feat[0]

        if len(visual_feat.shape) == 3:
            pass
        elif len(visual_feat.shape) == 4:
            b, c, h, w = visual_feat.shape
            visual_feat = visual_feat.reshape(b, c, -1).permute(0, 2, 1)

        visual_tokens = self.projector(visual_feat)
        return visual_tokens

    def forward(self, text_prompt, agent_features):
        """
        Args:
            text_prompt: str or list of str
            agent_features: [B, N, 59] tensor
        Returns:
            hidden_states: [B, N, hidden_dim] for agent tokens
        """
        B = agent_features.shape[0]

        n_vis = 0
        visual_tokens = torch.zeros(B, n_vis, self.llm.config.hidden_size, device=self.device, dtype=self.llm.dtype)

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
        hidden = outputs.hidden_states[-1][:, :n_agent, :]
        return hidden

    def load_checkpoint(self, checkpoint_dir):
        import os
        enc_path = os.path.join(checkpoint_dir, "encoder.pt")
        head_path = os.path.join(checkpoint_dir, "diffusion_head.pt")
        lora_path = os.path.join(checkpoint_dir, "lora_adapter")
        if os.path.exists(enc_path):
            self.feature_encoder.load_state_dict(
                torch.load(enc_path, map_location=self.device, weights_only=True)
            )
            print(f"  Loaded encoder: {enc_path}")
        if os.path.exists(head_path):
            self.diffusion_head.load_state_dict(
                torch.load(head_path, map_location=self.device, weights_only=True)
            )
            print(f"  Loaded diffusion head: {head_path}")
        if os.path.exists(lora_path):
            from peft import PeftModel
            self.llm = PeftModel.from_pretrained(self.llm, lora_path)
            print(f"  Loaded LoRA: {lora_path}")

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
            hidden = self.forward(text_prompt, agent_features)
            num_agents = features_dict.get("num_agents", 0)
            hidden = hidden[:, :num_agents, :]

            heading_cos = agent_features[0, :num_agents, 2:3]
            heading_sin = agent_features[0, :num_agents, 3:4]
            goal_global = agent_features[0, :num_agents, 12:14]
            goal_local_x = goal_global[:, 0:1] * heading_cos + goal_global[:, 1:2] * heading_sin
            goal_local_y = -goal_global[:, 0:1] * heading_sin + goal_global[:, 1:2] * heading_cos
            goal_feat = torch.cat([goal_local_x, goal_local_y], dim=-1)
            goal_feat = goal_feat / 40.0
            goal_feat = goal_feat * getattr(self, 'goal_scale', 1.0)

            waypoints_tensor = ddim_sample(
                self.diffusion_head, hidden,
                ddim_steps=self.diffusion_steps,
                eta=0.0,
                goal_features=goal_feat.unsqueeze(0),
            )

            if self._step_debug < 5:
                v_vals = waypoints_tensor[:, :, 0::2]
                w_vals = waypoints_tensor[:, :, 1::2]
                v_raw = v_vals * 1.5 + 1.5
                w_raw = w_vals * 18.5
                print(f"  DDIM norm: mean={waypoints_tensor.mean():.4f} std={waypoints_tensor.std():.4f} "
                      f"v(raw) mean={v_raw.mean():.2f} max={v_raw.max():.2f} "
                      f"ω(raw) mean={w_raw.mean():.2f} max={w_raw.max():.1f}")
                self._step_debug += 1

            V_MEAN, V_STD = 1.5, 1.5
            W_MEAN, W_STD = 0.0, 18.5
            result = {}
            for i in range(num_agents):
                pairs = []
                for j in range(self.chunk_size):
                    v_n = waypoints_tensor[0, i, j * 2].item()
                    w_n = waypoints_tensor[0, i, j * 2 + 1].item()
                    v = v_n * V_STD + V_MEAN
                    omega = w_n * W_STD + W_MEAN
                    pairs.append((v, omega))
                result[str(i)] = pairs
            return result


def load_mock_policy(action_mode="continuous"):
    return MockVLAPolicy(action_mode=action_mode)


def load_openvla_policy(model_id="openvla/openvla-7b", device="cpu",
                        action_mode="continuous", chunk_size=8):
    return OpenVLAPolicy(
        model_id=model_id, device=device, action_mode=action_mode,
        use_mock=False, chunk_size=chunk_size,
    )
