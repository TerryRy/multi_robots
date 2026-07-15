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
from vla.model_loader import AgentFeatureEncoder, FastProjector
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
            abs_dir = dir_path
            if not os.path.isdir(abs_dir):
                abs_dir = os.path.join(data_root, dir_path)
            if not os.path.isdir(abs_dir):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                abs_dir = os.path.join(script_dir, "..", dir_path)
            if not os.path.isdir(abs_dir):
                abs_dir = os.path.join(script_dir, "..", data_root, dir_path)
            if not os.path.isdir(abs_dir):
                abs_dir = os.path.join(script_dir, "..", "..", dir_path)
            if not os.path.isdir(abs_dir):
                raise FileNotFoundError(
                    f"Data directory not found: tried '{dir_path}', "
                    f"'{os.path.join(data_root, dir_path)}', "
                    f"and relative to script dir '{script_dir}/..' "
                    f"and '{script_dir}/../..'")

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

        self._feature_dim = 0
        if self.samples:
            first = self.samples[0]
            if "features" in first and "agents" in first["features"]:
                agents = first["features"]["agents"]
                if agents and len(agents) > 0:
                    self._feature_dim = len(agents[0])
        if self._feature_dim > 0:
            print(f"VLADataset: detected feature_dim={self._feature_dim}")

        self._filter_stationary()

        if max_samples and max_samples < len(self.samples):
            indices = random.choices(
                range(len(self.samples)),
                weights=self.instance_weights,
                k=max_samples,
            )
            self.samples = [self.samples[i] for i in indices]
            self.instance_weights = [1.0] * len(self.samples)

        print(f"VLADataset: {len(self.samples)} samples from {len(data_dirs)} sources")

    @property
    def feature_dim(self):
        return self._feature_dim if self._feature_dim > 0 else 59

    def _filter_stationary(self):
        if len(self.samples) < 20:
            return
        filtered = []
        for i, sample in enumerate(self.samples):
            text = sample.get("text_prompt", "")
            if "Collisions:" in text and "AA=0 AO=0" not in text:
                continue
            target = sample.get("target_action", {})
            max_disp = 0.0
            for aid, wps in target.items():
                if len(wps) >= 2:
                    dx = wps[-1][0] - wps[0][0]
                    dy = wps[-1][1] - wps[0][1]
                    disp = (dx*dx + dy*dy)**0.5
                    if disp > max_disp:
                        max_disp = disp
            if max_disp < 0.01:
                continue
            filtered.append(i)
        if len(filtered) > max(10, len(self.samples) * 0.1):
            removed = len(self.samples) - len(filtered)
            self.samples = [self.samples[i] for i in filtered]
            self.sources = [self.sources[i] for i in filtered]
            self.base_dirs = [self.base_dirs[i] for i in filtered]
            self.instance_weights = [self.instance_weights[i] for i in filtered]
            print(f"  Filtered {removed} stationary frames, "
                  f"{len(self.samples)} remaining")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]

        agents_feat = torch.tensor(record["features"]["agents"], dtype=torch.float32)

        target_dim = 59
        if agents_feat.shape[-1] < target_dim:
            padding = torch.zeros(agents_feat.shape[0],
                                  target_dim - agents_feat.shape[-1],
                                  dtype=torch.float32)
            agents_feat = torch.cat([agents_feat, padding], dim=-1)

        text_prompt = record["text_prompt"]
        target = record["target_action"]

        agent_ids = sorted(target.keys(), key=int)
        target_wps = [torch.tensor(target[aid], dtype=torch.float32) for aid in agent_ids]
        target_tensor = torch.stack(target_wps)

        image = None

        return agents_feat, text_prompt, target_tensor, image


def train(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float32
    if is_main:
        print(f"Device: {device} | GPUs: {world_size} | Dtype: {dtype}")

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
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_vla,
        num_workers=4, pin_memory=True,
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
        fast_lm = None
        img_preprocess = None
    elif args.fast:
        print("FAST mode: encoder + lightweight MLP + diffusion head (no LLM)")
        hidden_dim = 512

        vision_encoder = None
        projector = None
        tokenizer = None
        img_preprocess = None

        fast_lm = FastProjector(hidden_dim).to(device=device, dtype=dtype)
        fast_lm_params = sum(p.numel() for p in fast_lm.parameters()) / 1e3
        print(f"  FastProjector: {fast_lm_params:.0f}K params")

        llm = None
    else:
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if args.quantize and device.type == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = dtype
            if device.type == "cuda":
                load_kwargs["device_map"] = "auto"

        from transformers import AutoTokenizer, AutoModelForVision2Seq
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Flash Attention 2 enabled")
        except ImportError:
            pass

        print(f"Loading OpenVLA model: {args.model}")
        full_model = AutoModelForVision2Seq.from_pretrained(args.model, **load_kwargs)

        for attr in ['vision_encoder', 'vision_tower', 'vision_backbone']:
            if hasattr(full_model, attr):
                vision_encoder = getattr(full_model, attr)
                break
        else:
            raise RuntimeError(
                f"Cannot find vision_encoder in checkpoint. "
                f"Available: {[a for a in dir(full_model) if not a.startswith('_')]}"
            )

        for p in vision_encoder.parameters():
            p.requires_grad = False
        vision_encoder.eval()

        for attr in ['projector', 'connector', 'vision_projector']:
            if hasattr(full_model, attr):
                projector = getattr(full_model, attr)
                break
        else:
            raise RuntimeError(
                f"Cannot find projector in checkpoint. "
                f"Available: {[a for a in dir(full_model) if not a.startswith('_')]}"
            )

        for p in projector.parameters():
            p.requires_grad = False
        projector.eval()

        for attr in ["language_model", "model", "llm", "llm_backbone", "lm_backbone"]:
            if hasattr(full_model, attr):
                llm = getattr(full_model, attr)
                break
        else:
            raise RuntimeError(
                f"Cannot find language_model in checkpoint. "
                f"Available: {[a for a in dir(full_model) if not a.startswith('_')]}"
            )

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
            if hasattr(llm, 'base_model'):
                llm.base_model.gradient_checkpointing_enable()
            else:
                try:
                    llm.gradient_checkpointing_enable()
                except Exception:
                    pass
        else:
            llm.eval()
            for p in llm.parameters():
                p.requires_grad = False

        hidden_dim = llm.config.hidden_size
        if not args.quantize:
            if not hasattr(llm, 'device_map') or llm.device_map is None:
                llm.to(device)
            llm = llm.float()
        fast_lm = None

    # ---- MLP encoder + Diffusion head ----
    encoder = AgentFeatureEncoder(input_dim=59, hidden_dim=512, output_dim=hidden_dim)
    encoder = encoder.to(device=device, dtype=dtype)

    diffusion_head = DiffusionActionHead(
        hidden_dim=hidden_dim, chunk_size=args.chunk_size,
    )
    diffusion_head = diffusion_head.to(device=device, dtype=dtype)

    # ---- Load checkpoints (incremental training) ----
    def _resolve_path(p):
        if p is None or os.path.isabs(p) or os.path.exists(p):
            return p
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(script_dir, "..", p)
        if os.path.exists(alt):
            return alt
        alt = os.path.join(script_dir, "..", "..", p)
        if os.path.exists(alt):
            return alt
        return p

    if args.load_encoder:
        path = _resolve_path(args.load_encoder)
        encoder.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded encoder: {path}")
    if args.load_action_head:
        path = _resolve_path(args.load_action_head)
        diffusion_head.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded diffusion_head: {path}")
    if args.load_lora and args.use_lora:
        from peft import PeftModel
        path = _resolve_path(args.load_lora)
        llm = PeftModel.from_pretrained(llm, path)
        for n, p in llm.named_parameters():
            if 'lora' in n:
                p.requires_grad = True
        llm.enable_input_require_grads()
        if hasattr(llm, 'base_model'):
            llm.base_model.gradient_checkpointing_enable()
        else:
            try:
                llm.gradient_checkpointing_enable()
            except Exception:
                pass
        print(f"Loaded LoRA: {path}")

    # ---- Image preprocessing ----
    img_preprocess = None
    if not args.mock and not args.fast:
        import torchvision.transforms as T
        from PIL import Image as PILImage
        img_preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    # ---- Parameter summary ----
    if args.fast:
        fast_params = sum(p.numel() for p in fast_lm.parameters()) / 1e3
        enc_params = sum(p.numel() for p in encoder.parameters()) / 1e3
        head_params = sum(p.numel() for p in diffusion_head.parameters()) / 1e3
        if is_main:
            print(f"FastProjector: {fast_params:.0f}K | Encoder: {enc_params:.0f}K | "
                  f"DiffusionHead: {head_params:.0f}K")
            print(f"Total trainable: {(fast_params + enc_params + head_params) / 1e3:.1f}M")
    elif not args.mock:
        llm_params = sum(p.numel() for p in llm.parameters()) / 1e9
        trainable_llm = sum(p.numel() for p in llm.parameters() if p.requires_grad) / 1e6
        enc_params = sum(p.numel() for p in encoder.parameters()) / 1e3
        head_params = sum(p.numel() for p in diffusion_head.parameters()) / 1e3
        if is_main:
            print(f"LLM: {llm_params:.1f}B total | {trainable_llm:.1f}M trainable (LoRA)")
            print(f"Encoder: {enc_params:.0f}K | DiffusionHead: {head_params:.0f}K")
    if world_size > 1:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[local_rank])
        diffusion_head = torch.nn.parallel.DistributedDataParallel(diffusion_head, device_ids=[local_rank])
        if args.use_lora:
            llm = torch.nn.parallel.DistributedDataParallel(llm, device_ids=[local_rank])
        if is_main:
            print(f"DDP wrapping done. World size: {world_size}")

    # ---- Optimizer ----
    params = list(encoder.parameters()) + list(diffusion_head.parameters())
    if args.fast:
        params += list(fast_lm.parameters())
    elif args.use_lora:
        params += [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    from transformers import get_scheduler
    scheduler = get_scheduler(
        "cosine", optimizer,
        num_warmup_steps=int(args.epochs * len(loader) * 0.05),
        num_training_steps=args.epochs * len(loader),
    )

    # ---- Training loop (DDPM diffusion loss, unified float32) ----
    if is_main:
        print(f"\nTraining: {args.epochs} epochs x {len(loader)} steps/batch={args.batch_size}")

    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if not args.fast:
            llm.train() if args.use_lora else None
        else:
            fast_lm.train()
        encoder.train()
        diffusion_head.train()
        total_loss = 0

        for batch in loader:
            agent_feat, text_prompts, target_tensor, images = batch

            B = agent_feat.shape[0]
            agent_feat = agent_feat.to(device=device, dtype=dtype)
            target_tensor = target_tensor.to(device=device, dtype=dtype)

            if args.fast:
                hidden = fast_lm(encoder(agent_feat))
            else:
                n_patches = 0
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
            pos = agent_feat[:, :n_active, :2]
            cos_h = agent_feat[:, :n_active, 2:3]
            sin_h = agent_feat[:, :n_active, 3:4]
            target_local = target_flat - pos.repeat(1, 1, 8)
            tx = target_local[:, :, 0::2]
            ty = target_local[:, :, 1::2]
            local_x = tx * cos_h + ty * sin_h
            local_y = -tx * sin_h + ty * cos_h
            target_flat = torch.stack([local_x, local_y], dim=-1).reshape(B, n_active, -1)

            goal_global = agent_feat[:, :n_active, 12:14]
            goal_local_x = goal_global[:, :, 0:1] * cos_h + goal_global[:, :, 1:2] * sin_h
            goal_local_y = -goal_global[:, :, 0:1] * sin_h + goal_global[:, :, 1:2] * cos_h
            goal_feat = torch.cat([goal_local_x, goal_local_y], dim=-1)
            loss, _, _, _ = compute_diffusion_loss(diffusion_head, target_flat, hidden, goal_features=goal_feat)

            optimizer.zero_grad()
            loss.backward()
            grad_params = list(encoder.parameters()) + list(diffusion_head.parameters())
            if args.fast:
                grad_params += list(fast_lm.parameters())
            elif args.use_lora:
                grad_params += [p for p in llm.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(grad_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if is_main and global_step % max(1, len(loader) // 5) == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  E{epoch+1} [{global_step:5d}] loss={loss.item():.4f} lr={lr_now:.2e}")

        avg_loss = total_loss / len(loader)
        if is_main:
            print(f"  Epoch {epoch+1} avg_loss={avg_loss:.4f}")

    # ---- Save (rank 0 only) ----
    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        enc_sd = encoder.module.state_dict() if world_size > 1 else encoder.state_dict()
        head_sd = diffusion_head.module.state_dict() if world_size > 1 else diffusion_head.state_dict()
        torch.save(enc_sd, os.path.join(args.save_dir, "encoder.pt"))
        torch.save(head_sd, os.path.join(args.save_dir, "diffusion_head.pt"))
        if args.fast:
            lm_sd = fast_lm.state_dict()
            torch.save(lm_sd, os.path.join(args.save_dir, "fast_lm.pt"))
        if args.use_lora:
            llm_mod = llm.module if world_size > 1 else llm
            llm_mod.save_pretrained(os.path.join(args.save_dir, "lora_adapter"))
        print(f"Saved to {args.save_dir}/")

    if world_size > 1:
        torch.distributed.destroy_process_group()


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
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: encoder + lightweight MLP + diffusion head (no LLM, trains in minutes)")
    args = parser.parse_args()

    if args.data:
        args.data_mix = f"{args.data}:1.0"
    elif args.data_mix is None:
        args.data_mix = "data/trajectories:1.0"

    train(args)
