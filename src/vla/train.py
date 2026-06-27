"""
VLA training script for superPOD.
Trains MLP encoder + action head + optional LoRA on LLM backbone.

Usage:
    # Minimal test (135M model, encoder+head only)
    python src/vla/train.py --data data/trajectories/ --model SmolLM2-135M-Instruct

    # Full training (7B model + LoRA)
    python src/vla/train.py --data data/trajectories/ --model Qwen2.5-7B-Instruct --use-lora
"""
import json
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_scheduler,
)
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
        return agents_feat, text_prompt, target_tensor, len(target)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load LLM backbone ----
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    load_kwargs = {"dtype": dtype}
    if args.quantize and device.type == "cuda":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model = model.to(device)

    # ---- LoRA on LLM backbone ----
    if args.use_lora:
        print("Applying LoRA to LLM backbone...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   LoRA trainable params: {model_params/1e6:.1f}M")
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    hidden_dim = model.config.hidden_size
    encoder = AgentFeatureEncoder(input_dim=55, hidden_dim=512, output_dim=hidden_dim).to(device)
    action_head = WaypointActionHead(hidden_dim=hidden_dim, chunk_size=8).to(device)

    total_trainable = (
        sum(p.numel() for p in encoder.parameters())
        + sum(p.numel() for p in action_head.parameters())
    )
    print(f"Model backbone: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    print(f"Encoder+Head trainable: {total_trainable/1e3:.1f}K params")

    # ---- Data ----
    dataset = VLADataset(args.data, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ---- Optimizer ----
    params = list(encoder.parameters()) + list(action_head.parameters())
    if args.use_lora:
        params += [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    scheduler = get_scheduler(
        "cosine", optimizer,
        num_warmup_steps=10,
        num_training_steps=args.epochs * len(loader),
    )

    # ---- Training loop ----
    print(f"\nTraining: {args.epochs} epochs, {len(loader)} steps/epoch, lr={args.lr}")
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for step, (agent_feat, text_prompt, target_tensor, n_agents) in enumerate(loader):
            agent_feat = agent_feat.to(device)
            target_tensor = target_tensor.to(device)

            agent_embeds = encoder(agent_feat)

            encoded = tokenizer(
                list(text_prompt), return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            ).to(device)
            text_embeds = model.get_input_embeddings()(encoded["input_ids"])

            combined = torch.cat([agent_embeds, text_embeds], dim=1)
            outputs = model(inputs_embeds=combined, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, :agent_embeds.shape[1], :]
            pred = action_head(hidden)

            target_flat = target_tensor.reshape(agent_feat.shape[0], target_tensor.shape[1], -1)
            loss = F.mse_loss(pred[:, :target_tensor.shape[1], :], target_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % max(1, len(loader) // 10) == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} Step {step:4d} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} avg_loss: {avg_loss:.4f}")

    # ---- Save ----
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder.pt"))
    torch.save(action_head.state_dict(), os.path.join(args.save_dir, "action_head.pt"))
    if args.use_lora:
        model.save_pretrained(os.path.join(args.save_dir, "lora_adapter"))
    print(f"\nSaved to {args.save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID or local path (7B recommended)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (smaller for 7B)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="weights/")
    parser.add_argument("--use-lora", action="store_true", help="Apply LoRA to LLM backbone")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization (for 7B+ models)")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    args = parser.parse_args()
    train(args)
