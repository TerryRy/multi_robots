"""
VLA training script with Diffusion + Curriculum Learning.

Usage:
    # Stage 1: train from scratch with one data directory
    python src/vla/train.py --data data/trajectories/ --model openvla/openvla-7b --epochs 20

    # Stage N: incremental training with curriculum mixing
    python src/vla/train.py \
        --data stage_1:0.2,stage_2:0.3,stage_3:0.5 \
        --load-encoder weights/stage_2/encoder.pt \
        --load-action-head weights/stage_2/diffusion_head.pt \
        --load-lora weights/stage_2/lora_adapter/ \
        --model openvla/openvla-7b --epochs 20 \
        --save-dir weights/stage_3/
"""
import json
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vla.model_loader import AgentFeatureEncoder
from vla.diffusion_head import DiffusionActionHead, compute_diffusion_loss


class VLADataset(Dataset):
    def __init__(self, data_dirs, sample_ratios=None, max_samples=None,
                 data_root="data/trajectories"):
        """
        data_dirs: list of directory paths (relative to data_root or absolute)
        sample_ratios: list of sampling weights for each directory,
                       or None for uniform over all samples
        """
        self.samples = []
        self.sources = []
        self.base_dirs = []

        for dir_idx, dir_path in enumerate(data_dirs):
            if not os.path.isabs(dir_path):
                abs_dir = os.path.join(data_root, dir_path)
            else:
                abs_dir = dir_path
            if not os.path.isdir(abs_dir):
                abs_dir = dir_path

            for fname in sorted(os.listdir(abs_dir)):
                if not fname.endswith(".jsonl"):
                    continue
                fpath = os.path.join(abs_dir, fname)
                with open(fpath) as f:
                    for line in f:
                        self.samples.append(json.loads(line))
                        self.sources.append(dir_idx)
                        self.base_dirs.append(abs_dir)

        if sample_ratios is not None:
            n_sources = len(data_dirs)
            counts = [0] * n_sources
            for s in self.sources:
                counts[s] += 1
            total = len(self.samples)
            inv_weights = [1.0 / max(r, 0.01) for r in sample_ratios[:n_sources]]
            self.instance_weights = []
            for s in self.sources:
                expected = total * sample_ratios[s]
                actual = counts[s]
                self.instance_weights.append(inv_weights[s] if actual > 0 else 0.0)
        else:
            self.instance_weights = [1.0] * len(self.samples)

        if max_samples and max_samples < len(self.samples):
            indices = random.choices(
                range(len(self.samples)),
                weights=self.instance_weights,
                k=max_samples,
            )
            self.samples = [self.samples[i] for i in indices]
            self.instance_weights = [1.0] * len(self.samples)

        print(f"VLADataset: {len(self.samples)} samples from {len(data_dirs)} sources")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]

        agents_feat = torch.tensor(record["features"]["agents"], dtype=torch.float32)
        text_prompt = record["text_prompt"]
        target = record["target_action"]

        agent_ids = sorted(target.keys(), key=int)
        target_wps = [torch.tensor(target[aid], dtype=torch.float32) for aid in agent_ids]
        target_tensor = torch.stack(target_wps)

        image = None
        image_path = record.get("image_path")
        if image_path:
            full_path = os.path.join(self.base_dirs[idx], image_path)
            if os.path.exists(full_path):
                image = np.array(Image.open(full_path).convert("RGB"))

        return agents_feat, text_prompt, target_tensor, image


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device} | Dtype: {dtype}")

    # ---- Data directories & curriculum mixing ----
    data_mix = args.data_mix
    data_dirs = []
    sample_ratios = []
    for entry in data_mix.split(","):
        parts = entry.strip().split(":")
        dir_path = parts[0]
        ratio = float(parts[1]) if len(parts) > 1 else 1.0
        data_dirs.append(dir_path)
        sample_ratios.append(ratio)

    total_ratio = sum(sample_ratios)
    sample_ratios = [r / total_ratio for r in sample_ratios]

    dataset = VLADataset(
        data_dirs, sample_ratios=sample_ratios,
        max_samples=args.max_samples, data_root=args.data_root,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_vla,
    )

    # ---- Model loading (mock or real OpenVLA) ----
    if args.mock:
        print("MOCK mode: creating dummy components for local verification")
        hidden_dim = 64
        class DummyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))
            def forward(self, x):
                B, N = x.shape[0], x.shape[1]
                return torch.zeros(B, 256, hidden_dim, device=x.device, dtype=x.dtype)
        class DummyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('obj', (object,), {'hidden_size': hidden_dim})
                self.dummy = nn.Parameter(torch.zeros(1))
            def get_input_embeddings(self):
                return nn.Identity()
            def forward(self, inputs_embeds=None, output_hidden_states=False):
                B, L, D = inputs_embeds.shape
                hidden = torch.zeros(B, L, D, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                hidden[:, :, 0] = 1.0
                return type('obj', (object,), {'hidden_states': [hidden, hidden, hidden]})
            def to(self, *args, **kwargs):
                return self
            def eval(self):
                return self
            def parameters(self):
                return iter([self.dummy])
            def train(self, mode=True):
                return self
        vision_encoder = DummyEncoder()
        projector = nn.Identity()
        llm = DummyLM()
        llm.to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model) if not args.mock else None
    else:
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if device.type == "cuda" and not args.quantize:
            load_kwargs["device_map"] = "auto"
        if args.quantize and device.type == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            )

        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading OpenVLA model: {args.model}")
        full_model = AutoModel.from_pretrained(args.model, **load_kwargs)

        if hasattr(full_model, 'vision_encoder'):
            vision_encoder = full_model.vision_encoder
        elif hasattr(full_model, 'vision_tower'):
            vision_encoder = full_model.vision_tower
        else:
            raise RuntimeError("Cannot find vision_encoder in checkpoint")

        for p in vision_encoder.parameters():
            p.requires_grad = False
        vision_encoder.eval()

        if hasattr(full_model, 'projector'):
            projector = full_model.projector
        else:
            raise RuntimeError("Cannot find projector in checkpoint")

        for p in projector.parameters():
            p.requires_grad = False
        projector.eval()

        if hasattr(full_model, "language_model"):
            llm = full_model.language_model
        elif hasattr(full_model, "model"):
            llm = full_model.model
        else:
            raise RuntimeError("Cannot find language_model in checkpoint")

        del full_model

        if args.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
            )
            llm = get_peft_model(llm, lora_config)
            llm.enable_input_require_grads()
        else:
            llm.eval()
            for p in llm.parameters():
                p.requires_grad = False

        hidden_dim = llm.config.hidden_size
        if not hasattr(llm, 'device_map') or llm.device_map is None:
            llm.to(device)

    # ---- MLP encoder + Diffusion head ----
    encoder = AgentFeatureEncoder(input_dim=55, hidden_dim=512, output_dim=hidden_dim)
    encoder = encoder.to(device=device, dtype=dtype)

    diffusion_head = DiffusionActionHead(
        hidden_dim=hidden_dim, chunk_size=args.chunk_size,
    )
    diffusion_head = diffusion_head.to(device=device, dtype=dtype)

    # ---- Load checkpoints (incremental training) ----
    if args.load_encoder:
        encoder.load_state_dict(torch.load(args.load_encoder, map_location=device))
        print(f"Loaded encoder: {args.load_encoder}")
    if args.load_action_head:
        diffusion_head.load_state_dict(torch.load(args.load_action_head, map_location=device))
        print(f"Loaded diffusion_head: {args.load_action_head}")
    if args.load_lora and args.use_lora:
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, args.load_lora)
        print(f"Loaded LoRA: {args.load_lora}")

    # ---- Image preprocessing ----
    img_preprocess = None
    if not args.mock:
        import torchvision.transforms as T
        from PIL import Image as PILImage
        img_preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    # ---- Parameter summary ----
    llm_params = sum(p.numel() for p in llm.parameters()) / 1e9
    trainable_llm = sum(p.numel() for p in llm.parameters() if p.requires_grad) / 1e6
    enc_params = sum(p.numel() for p in encoder.parameters()) / 1e3
    head_params = sum(p.numel() for p in diffusion_head.parameters()) / 1e3
    print(f"LLM: {llm_params:.1f}B total | {trainable_llm:.1f}M trainable (LoRA)")
    print(f"Encoder: {enc_params:.0f}K | DiffusionHead: {head_params:.0f}K")

    # ---- Optimizer ----
    params = list(encoder.parameters()) + list(diffusion_head.parameters())
    if args.use_lora:
        params += [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    from transformers import get_scheduler
    scheduler = get_scheduler(
        "cosine", optimizer,
        num_warmup_steps=int(args.epochs * len(loader) * 0.05),
        num_training_steps=args.epochs * len(loader),
    )

    # ---- Training loop (DDPM diffusion loss) ----
    print(f"\nTraining: {args.epochs} epochs x {len(loader)} steps/batch={args.batch_size}")
    global_step = 0
    for epoch in range(args.epochs):
        llm.train() if args.use_lora else None
        encoder.train()
        diffusion_head.train()
        total_loss = 0

        for batch in loader:
            agent_feat, text_prompts, target_tensor, images = batch

            B = agent_feat.shape[0]
            agent_feat = agent_feat.to(device=device, dtype=dtype)
            target_tensor = target_tensor.to(device=device, dtype=dtype)

            with torch.no_grad():
                if img_preprocess is not None and any(img is not None for img in images):
                    from PIL import Image as PILImage
                    pixel_values = []
                    for img in images:
                        if img is not None:
                            pil_img = PILImage.fromarray(img)
                        else:
                            pil_img = PILImage.new("RGB", (224, 224), (30, 30, 30))
                        pixel_values.append(img_preprocess(pil_img))
                    pixel_values = torch.stack(pixel_values).to(device=device, dtype=dtype)

                    visual_feat = vision_encoder(pixel_values)
                    if isinstance(visual_feat, tuple):
                        visual_feat = visual_feat[0]
                    if len(visual_feat.shape) == 4:
                        b, c, h, w = visual_feat.shape
                        visual_feat = visual_feat.reshape(b, c, -1).permute(0, 2, 1)
                    visual_tokens = projector(visual_feat)
                else:
                    n_patches = 256 if not args.mock else 16
                    visual_tokens = torch.zeros(B, n_patches, hidden_dim, device=device, dtype=dtype)

            agent_embeds = encoder(agent_feat)

            if tokenizer is not None:
                encoded = tokenizer(
                    text_prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024,
                ).to(device)
                text_embeds = llm.get_input_embeddings()(encoded["input_ids"])
                combined = torch.cat([visual_tokens, agent_embeds, text_embeds], dim=1)
                n_vis_tokens = visual_tokens.shape[1]
            else:
                combined = torch.cat([visual_tokens, agent_embeds], dim=1)
                n_vis_tokens = visual_tokens.shape[1]

            outputs = llm(inputs_embeds=combined, output_hidden_states=True)

            n_agent = agent_embeds.shape[1]
            hidden = outputs.hidden_states[-1][:, n_vis_tokens:n_vis_tokens + n_agent, :]

            n_active = target_tensor.shape[1]
            hidden = hidden[:, :n_active, :]

            target_flat = target_tensor.reshape(B, n_active, -1)

            loss = compute_diffusion_loss(diffusion_head, target_flat, hidden)

            optimizer.zero_grad()
            loss.backward()

            grad_params = list(encoder.parameters()) + list(diffusion_head.parameters())
            if args.use_lora:
                grad_params += [p for p in llm.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(grad_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % max(1, len(loader) // 5) == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  E{epoch+1} [{global_step:5d}] loss={loss.item():.4f} lr={lr_now:.2e}")

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1} avg_loss={avg_loss:.4f}")

    # ---- Save ----
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder.pt"))
    torch.save(diffusion_head.state_dict(), os.path.join(args.save_dir, "diffusion_head.pt"))
    if args.use_lora:
        llm.save_pretrained(os.path.join(args.save_dir, "lora_adapter"))
    print(f"Saved to {args.save_dir}/")


def collate_vla(batch):
    agent_feat_list = []
    text_list = []
    target_list = []
    image_list = []

    max_agents = max(item[0].shape[0] for item in batch)
    n_coords = batch[0][2].shape[-1]
    chunk_size = batch[0][2].shape[1]

    for af, text, target, img in batch:
        n = af.shape[0]
        if n < max_agents:
            pad_feat = torch.zeros(max_agents - n, af.shape[1], dtype=af.dtype)
            af = torch.cat([af, pad_feat], dim=0)
            pad_target = torch.zeros(max_agents - n, chunk_size, n_coords, dtype=target.dtype)
            target = torch.cat([target, pad_target], dim=0)
        agent_feat_list.append(af)
        text_list.append(text)
        target_list.append(target)
        image_list.append(img)

    agent_feat_tensor = torch.stack(agent_feat_list)
    target_tensor = torch.stack(target_list)
    return agent_feat_tensor, text_list, target_tensor, image_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                        help="Single data directory (overrides --data-mix)")
    parser.add_argument("--data-mix", type=str, default=None,
                        help="Curriculum: 'dir1:0.2,dir2:0.3,dir3:0.5'")
    parser.add_argument("--data-root", type=str, default="data/trajectories")
    parser.add_argument("--model", type=str, default="openvla/openvla-7b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="weights/")
    parser.add_argument("--load-encoder", type=str, default=None)
    parser.add_argument("--load-action-head", type=str, default=None)
    parser.add_argument("--load-lora", type=str, default=None)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mock", action="store_true",
                        help="Mock mode: dummy components for local training loop verification")
    args = parser.parse_args()

    if args.data:
        args.data_mix = f"{args.data}:1.0"
    elif args.data_mix is None:
        args.data_mix = "data/trajectories:1.0"

    train(args)
