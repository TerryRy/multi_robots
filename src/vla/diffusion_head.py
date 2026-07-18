import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        freqs = freqs.to(t.dtype)
        emb = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cond_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, cond):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        cond_proj = self.cond_proj(cond)
        x = x + self.cross_attn(self.norm2(x), cond_proj, cond_proj)[0]
        x = x + self.ff(self.norm3(x))
        return x


class DiffusionActionHead(nn.Module):
    def __init__(self, hidden_dim=4096, chunk_size=8, cond_dim=512, d_model=512,
                 n_blocks=4, n_heads=8, T=1000):
        super().__init__()
        self.chunk_size = chunk_size
        self.T = T
        self.goal_dim = 2

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(128),
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.input_proj = nn.Linear(chunk_size * 2, d_model)
        self.cond_proj = nn.Linear(hidden_dim + self.goal_dim, d_model)
        self.time_proj = nn.Linear(d_model, d_model)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads)
            for _ in range(n_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, chunk_size * 2),
        )
        nn.init.xavier_uniform_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, x_t, t, condition, goal_features=None):
        B, N, _ = x_t.shape
        dtype = next(self.parameters()).dtype

        t_emb = self.time_embed(t.to(dtype))
        t_emb = self.time_proj(t_emb[:, None, :].expand(-1, N, -1))

        x = self.input_proj(x_t.to(dtype))
        c_input = condition.to(dtype)
        if goal_features is not None:
            c_input = torch.cat([c_input, goal_features.to(dtype)], dim=-1)
        c = self.cond_proj(c_input)
        h = x + t_emb

        for block in self.blocks:
            h = block(h, c)

        return self.output_proj(h)


def _cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    beta = torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], max=0.02)
    return beta.float()


def compute_diffusion_loss(model, x_0, condition, T=1000, goal_features=None):
    B = x_0.shape[0]
    device = x_0.device

    if torch.isnan(x_0).any() or torch.isinf(x_0).any():
        print("WARNING: x_0 contains NaN/Inf, skipping")
        return torch.tensor(0.0, device=device, requires_grad=True), None, None, None

    t = torch.randint(0, T // 2, (B,), device=device)
    noise = torch.randn_like(x_0)

    beta = _cosine_beta_schedule(T).to(device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    sqrt_alpha_bar = alpha_bar[t].sqrt().view(B, 1, 1)
    sqrt_one_minus = (1 - alpha_bar[t]).sqrt().view(B, 1, 1)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise

    noise_pred = model(x_t, t.float(), condition, goal_features=goal_features)

    pred_x0 = (x_t - sqrt_one_minus * noise_pred) / (sqrt_alpha_bar + 1e-8)

    if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
        print("WARNING: noise_pred contains NaN/Inf, skipping")
        return torch.tensor(0.0, device=device, requires_grad=True), None, None, None

    return F.mse_loss(noise_pred, noise), noise_pred, x_t, t


@torch.no_grad()
def ddim_sample(model, condition, ddim_steps=50, T=1000, eta=0.0, goal_features=None):
    B, N, _ = condition.shape
    device = condition.device

    max_t = T // 2
    step_ratio = max_t // ddim_steps
    timesteps = list(range(0, max_t, step_ratio))
    timesteps.reverse()

    beta = _cosine_beta_schedule(T).to(device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    x = torch.randn(B, N, model.chunk_size * 2, device=device)

    for i in range(len(timesteps) - 1):
        t_val = timesteps[i]
        t_next_val = timesteps[i + 1]
        t = torch.full((B,), t_val, device=device, dtype=torch.long)

        noise_pred = model(x, t, condition, goal_features=goal_features)

        alpha_bar_t = alpha_bar[t_val]
        alpha_bar_next = alpha_bar[max(t_next_val, 0)]

        pred_x0 = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()

        if eta > 0 and t_next_val > 0:
            sigma = eta * ((1 - alpha_bar_next) / (1 - alpha_bar_t) *
                           (1 - alpha_bar_t / alpha_bar_next)).sqrt()
            noise = torch.randn_like(x)
        else:
            sigma = 0.0
            noise = 0.0

        c1 = alpha_bar_next.sqrt()
        c2 = (1 - alpha_bar_next - sigma ** 2).sqrt()
        x = c1 * pred_x0 + c2 * noise_pred + sigma * noise

    return x
