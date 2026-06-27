"""
Minimal VLA training script for superPOD.
Loads collected JSONL data, trains MLP encoder + action head on frozen LLM backbone.

Usage:
    python src/vla/train.py --data src/data/trajectories/ --model models/SmolLM2-135M-Instruct/
    python src/vla/train.py --data src/data/trajectories/ --model HuggingFaceTB/SmolLM2-1.7B-Instruct --use-lora
"""
import json
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vla.model_loader import AgentFeatureEncoder, WaypointActionHead


class VLADataset(Dataset):
    def __init__(self, data_dir, max_samples=None):
        self.samples = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith('.jsonl'):
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
        agents_feat = torch.tensor(record['features']['agents'], dtype=torch.float32)
        text_prompt = record['text_prompt']
        target = record['target_action']

        target_wps = []
        for aid in sorted(target.keys(), key=int):
            target_wps.append(torch.tensor(target[aid], dtype=torch.float32))
        target_tensor = torch.stack(target_wps)
        return agents_feat, text_prompt, target_tensor, len(target)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16 if device.type == 'cuda' else torch.float32)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.hidden_size
    encoder = AgentFeatureEncoder(input_dim=55, hidden_dim=512, output_dim=hidden_dim).to(device)
    action_head = WaypointActionHead(hidden_dim=hidden_dim, chunk_size=8).to(device)

    total_trainable = sum(p.numel() for p in list(encoder.parameters()) + list(action_head.parameters()))
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    print(f"Trainable: {total_trainable/1e3:.1f}K params (Encoder + ActionHead)")

    # Data
    dataset = VLADataset(args.data, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(action_head.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    num_epochs = args.epochs
    scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=10,
                              num_training_steps=num_epochs * len(loader))

    # Training loop
    print(f"\nTraining: {num_epochs} epochs, {len(loader)} steps/epoch, lr={args.lr}")
    for epoch in range(num_epochs):
        total_loss = 0
        for step, (agent_feat, text_prompt, target_tensor, n_agents) in enumerate(loader):
            agent_feat = agent_feat.to(device)
            target_tensor = target_tensor.to(device)

            # Forward
            agent_embeds = encoder(agent_feat)

            # Tokenize batch
            encoded = tokenizer(list(text_prompt), return_tensors='pt', padding=True,
                                truncation=True, max_length=512).to(device)
            text_embeds = model.get_input_embeddings()(encoded['input_ids'])

            combined = torch.cat([agent_embeds, text_embeds], dim=1)

            outputs = model(inputs_embeds=combined, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, :agent_embeds.shape[1], :]
            pred = action_head(hidden)

            # Reshape target: [B, agents, 8, 2] -> [B, agents, 16]
            target_flat = target_tensor.reshape(agent_feat.shape[0], target_tensor.shape[1], -1)
            loss = F.mse_loss(pred[:, :target_tensor.shape[1], :], target_flat)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(action_head.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} Step {step:4d} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} avg_loss: {avg_loss:.4f}")

    # Save
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(save_dir, 'encoder.pt'))
    torch.save(action_head.state_dict(), os.path.join(save_dir, 'action_head.pt'))
    print(f"\nSaved to {save_dir}/")

    # Verify
    print("Verifying saved weights...")
    encoder2 = AgentFeatureEncoder(input_dim=55, hidden_dim=512, output_dim=hidden_dim)
    encoder2.load_state_dict(torch.load(os.path.join(save_dir, 'encoder.pt')))
    print("Weights verified OK")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to training data directory (JSONL files)')
    parser.add_argument('--model', type=str, default='HuggingFaceTB/SmolLM2-1.7B-Instruct',
                        help='HuggingFace model ID or local path')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to load (for testing)')
    parser.add_argument('--save-dir', type=str, default='weights/', help='Directory to save trained weights')
    args = parser.parse_args()
    train(args)
