"""
VLA training script for OpenVLA-7B backbone.
Loads collected JSONL data. Freezes the Llama2-7B backbone,
trains MLP encoder + action head + optional LoRA adapter.

Usage:
    # Small test model (local, 135M, no LoRA):
    python src/vla/train.py --data data/trajectories/ --model models/SmolLM2-135M-Instruct --epochs 2 --max-samples 100

    # OpenVLA-7B (superPOD, with LoRA):
    python src/vla/train.py --data data/trajectories/ --model openvla/openvla-7b --use-lora --quantize --batch-size 1
"""
import json
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vla.model_loader import AgentFeatureEncoder, WaypointActionHead


class VLADataset(Dataset):
    def __init__(self, data_dir, max_samples=None):
        self.samples = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(data_dir, fname)) as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]
        agents_feat = torch.tensor(record["features"]["agents"], dtype=torch.float32)
        text_prompt = record["text_prompt"]
        target = record["target_action"]
        target_wps = []
        for aid in sorted(target.keys(), key=int):
            target_wps.append(torch.tensor(target[aid], dtype=torch.float32))
        target_tensor = torch.stack(target_wps)
        return agents_feat, text_prompt, target_tensor


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device} | Dtype: {dtype}")

    # ---- Load tokenizer ----
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load OpenVLA-7B, extract Llama2 backbone ----
    print(f"Loading OpenVLA-7B: {args.model}")
    from transformers import AutoModel, BitsAndBytesConfig

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if args.quantize and device.type == "cuda":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )

    full_model = AutoModel.from_pretrained(args.model, **load_kwargs)

    # Extract Llama2-7B language model
    if hasattr(full_model, "language_model"):
        llm = full_model.language_model
    elif hasattr(full_model, "model"):
        llm = full_model.model
    else:
        raise RuntimeError("Cannot find language_model in OpenVLA checkpoint")
    del full_model

    # ---- LoRA on Llama2 backbone ----
    if args.use_lora:
        print("Applying LoRA to Llama2-7B backbone...")
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
    llm.to(device)

    # ---- MLP encoder + action head ----
    encoder = AgentFeatureEncoder(input_dim=55, hidden_dim=512, output_dim=hidden_dim)
    encoder = encoder.to(device=device, dtype=dtype)
    action_head = WaypointActionHead(hidden_dim=hidden_dim, chunk_size=8)
    action_head = action_head.to(device=device, dtype=dtype)

    total_model_params = sum(p.numel() for p in llm.parameters()) / 1e9
    trainable_params = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in encoder.parameters()) / 1e3
    head_params = sum(p.numel() for p in action_head.parameters()) / 1e3
    print(f"Model: {total_model_params:.1f}B params")
    print(f"LLM trainable (LoRA): {trainable_params/1e6:.1f}M")
    print(f"Encoder: {encoder_params:.0f}K | ActionHead: {head_params:.0f}K")

    # ---- Data ----
    dataset = VLADataset(args.data, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ---- Optimizer ----
    params = list(encoder.parameters()) + list(action_head.parameters())
    if args.use_lora:
        params += [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = get_scheduler(
        "cosine", optimizer,
        num_warmup_steps=5,
        num_training_steps=args.epochs * len(loader),
    )

    # ---- Training ----
    print(f"\nTraining: {args.epochs} epochs x {len(loader)} steps, lr={args.lr}")
    for epoch in range(args.epochs):
        llm.train()
        total_loss = 0
        for step, (agent_feat, text_prompt, target_tensor) in enumerate(loader):
            agent_feat = agent_feat.to(device=device, dtype=dtype)
            target_tensor = target_tensor.to(device=device, dtype=dtype)

            agent_embeds = encoder(agent_feat)

            encoded = tokenizer(
                list(text_prompt), return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            ).to(device)
            text_embeds = llm.get_input_embeddings()(encoded["input_ids"])

            combined = torch.cat([agent_embeds, text_embeds], dim=1)
            outputs = llm(inputs_embeds=combined, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, :agent_embeds.shape[1], :]
            pred = action_head(hidden)

            # MSE between predicted and expert waypoints
            target_flat = target_tensor.reshape(
                agent_feat.shape[0], target_tensor.shape[1], -1
            ).to(device=device, dtype=dtype)
            loss = F.mse_loss(pred[:, :target_tensor.shape[1], :], target_flat)

            optimizer.zero_grad()
            loss.backward()
            params_all = list(encoder.parameters()) + list(action_head.parameters())
            if args.use_lora:
                params_all += [p for p in llm.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params_all, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % max(1, len(loader) // 10) == 0:
                print(f"  Epoch {epoch+1} [{step:4d}/{len(loader)}] loss={loss.item():.4f}")

        print(f"  Epoch {epoch+1} avg_loss={total_loss/len(loader):.4f}")

    # ---- Save ----
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder.pt"))
    torch.save(action_head.state_dict(), os.path.join(args.save_dir, "action_head.pt"))
    if args.use_lora:
        llm.save_pretrained(os.path.join(args.save_dir, "lora_adapter"))
    print(f"Saved to {args.save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="openvla/openvla-7b",
                        help="OpenVLA-7B model ID or local path")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="weights/")
    parser.add_argument("--use-lora", action="store_true", help="Apply LoRA to Llama2 backbone")
    parser.add_argument("--quantize", action="store_true", help="4-bit quantization (for superPOD)")
    parser.add_argument("--lora-rank", type=int, default=64)
    args = parser.parse_args()
    train(args)
