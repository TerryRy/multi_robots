# VLA 模块设计文档

> 多机器人调度模拟器的 Vision-Language-Action 接口层  
> 基于 OpenVLA-7B (Llama2-7B + LoRA) + DDPM Diffusion Action Head

---

## 1. 架构总览

```
59 维特征 × N agents → MLP Encoder → Llama2-7B (LoRA) → DiffusionActionHead → 8 waypoints × N agents
文本 prompt → Tokenizer ──────────────────────────────────────────────┘
视觉 token ─ 训练时已移除 (n_patches=0), 推理时同样跳过
```

每个 agent 输出 8 个 waypoint（约 0.13 秒路径），WaypointTracker 做线性插值转为 (vx, vy) 速度指令。

---

## 2. 模型组件

### 2.1 AgentFeatureEncoder

```
59 → Linear(59,512) → ReLU → Linear(512,512) → ReLU → Linear(512,4096) → LayerNorm(4096)
```

~240 万参数，全参数训练。将 59 维结构化特征映射到 Llama2-7B 的 4096 维隐空间。

### 2.2 Llama2-7B (LoRA)

- **冻结**：7B 主体参数
- **微调**：LoRA rank=128 on `q_proj`, `v_proj`（~6700 万可训练参数）
- 作用：多 agent cross-attention。N 个 agent 的 token 在 LLM 内部互相 attend，学习「谁先过、谁让行、排队」
- Gradient checkpointing 开启（省显存 ~40%，慢 ~30%）

### 2.3 DiffusionActionHead

```
输入: hidden(4096) + goal_feat(2) → cat → Linear(4098→512) → 4×CrossAttentionBlock → Linear(512→16)
```

~1700 万参数，全参数训练。  
4 个 CrossAttentionBlock（self-attn + cross-attn + FFN），condition 为 LLM hidden + goal 方向。  
输出 8 对 (Δx, Δy) 局部帧 waypoint。  
最终输出层用正态初始化（std=0.001），避免零坍缩。

### 2.4 总可训练参数

| 模块 | 参数量 | 训练方式 |
|------|--------|----------|
| AgentFeatureEncoder | ~2.4M | 全参数 |
| diffusion_head | ~17M | 全参数 |
| Llama2-7B LoRA (rank=128) | ~67M | LoRA |
| SigLIP + Projector + Tokenizer | ~370M | 冻结 |
| **合计** | **~86M** | |

---

## 3. 输入特征规范（59 维/agent）

| 索引 | 维度 | 字段 | 说明 |
|------|------|------|------|
| 0-1 | 2 | position | (x, y) 世界坐标 |
| 2-3 | 2 | heading | (cosθ, sinθ) |
| 4 | 1 | speed | 当前速率 (m/s) |
| 5-10 | 6 | state one-hot | IDLE/LOADING/QUEUING/HALT/CRUISE/PREQUEUE |
| 11 | 1 | carrying | 是否载货 (0/1) |
| 12-13 | 2 | goal offset | (dx, dy) 全局坐标 → 训练时转为局部帧 |
| 14-15 | 2 | goal heading | (cosφ, sinφ) 全局 |
| 16-35 | 20 | neighbors ×4 | (dist, cos rel_angle, sin rel_angle, cos hd_diff, sin hd_diff) |
| 36-51 | 16 | LiDAR | 512 ray → 16 bin min 降采样 |
| 52-54 | 3 | nearest obstacle | (dist, cos angle, sin angle) |
| 55-58 | 4 | nearest port | (dist, cos angle, sin angle, is_loading) |

**坐标帧一致性**：训练和推理中，goal_feat 和 target waypoint 统一转为局部帧。

```
local_x =  dx_global * cos(θ) + dy_global * sin(θ)
local_y = -dx_global * sin(θ) + dy_global * cos(θ)
```

推理时逆旋转：`gx = px + lx*cosθ − ly*sinθ`

---

## 4. 数据采集

### 4.1 JSONL 格式

```json
{
  "text_prompt": "... Agent_0 pos=...",
  "features": {"agents": [[...]], "num_agents": N, ...},
  "target_action": {"0": [(x,y)*8], "1": [(x,y)*8]},
  "image_path": "session_xxx/frame_xxx.png",
  "agent_count": N
}
```

target_action 为专家真实走过的 8 个全局坐标。features 为 59 维（旧 55 维数据自动 padding）。

### 4.2 采集命令

```bash
# pure expert（默认）
sbatch collect_data.sh <stage>

# DAgger（需手动传 --mix）
sbatch collect_data.sh <stage> "--mix expert:0.4,policy:0.4,random:0.2 --vla-model openvla/openvla-7b --vla-device cuda --vla-checkpoint weights/stage_N"
```

### 4.3 Stage 参数

| Stage | Agents | Ports | Map | Time |
|-------|--------|-------|-----|------|
| 1 | 2 | 2+2 | 20×12 | 10 min |
| 2 | 4 | 4+4 | 30×16 | 10 min |
| 3 | 5 | 5+5 | 40×20 | 12 min |
| 4 | 7 | 7+7 | 45×20 | 15 min |

stride=4，VA 采集时记录 expert 轨迹。

### 4.4 碰撞过滤

训练加载数据时自动过滤碰撞帧（`_filter_waypoint_direction`）：text_prompt 中 `Collisions: AA=... AO=...` 非零即丢弃。样本 < 20 或过滤后 < 10% 时跳过。

---

## 5. 训练

### 5.1 命令

```bash
# Stage 1: 从头训
sbatch train.sh 1

# Stage 2-4: 课程学习（加载前 stage checkpoint）
sbatch train.sh 2

# Fresh: 单数据集从头训
sbatch train.sh fresh --data data/trajectories/stage_3 --epochs 8

# Quick: 管线验证
sbatch train.sh quick
```

### 5.2 配置

| 参数 | 值 |
|------|-----|
| 模型 | openvla/openvla-7b |
| LoRA rank | 128 |
| 优化器 | AdamW, lr=3e-5, weight_decay=0.01 |
| Scheduler | Cosine, 5% warmup |
| Batch size | 4 |
| Epochs | 8 (Stage), 4 (fresh 默认) |
| Max grad norm | 1.0 |
| Gradient checkpointing | 开启 |

**加速优化**：
- 视觉 token 完全移除 (n_patches=0, 训练推理一致)：LLM input ~568→312 tokens, attention 快 ~70%
- alpha_bar schedule 预计算：training loop 外一次性提前算好
- Flash Attention 2 自动检测（未安装时回退标准 attention）

### 5.3 训练循环

```
for batch in DataLoader:
    1. visual_tokens = zeros(0)  (已移除)
    2. agent_embeds = Encoder(agent_feat)
    3. text_embeds = Tokenizer(text_prompts)
    4. combined = cat(visual_tokens + agent_embeds + text_embeds)
    5. LLM forward → hidden = hidden_at_agent_positions
    6. target_flat = 全局wp → 局部帧 (旋转 + 减位置)
    7. loss = DDPM_loss + dir_loss + wrong_dir + mag_penalty + smooth_penalty + collision_penalty
    8. backward + clip_grad + step
```

### 5.4 Checkpoint 结构

```
weights/stage_N/
├── encoder.pt           # AgentFeatureEncoder 权重
├── diffusion_head.pt    # DiffusionActionHead 权重
└── lora_adapter/        # PEFT LoRA adapter (adapter_config.json + model.safetensors)
```

加载时 `PeftModel.from_pretrained()` 后显式 `p.requires_grad = True` for lora params，避免 LoRA 冻结 bug。

---

## 6. 损失函数

预计算 `alpha_bar_schedule`（于 training loop 外），避免每 batch 重复构建。

### 6.1 主损失：DDPM Diffusion Loss (λ=1.0)

```
loss = MSE(noise_pred, noise)
```

- T=1000，cosine beta schedule
- t ∈ [0, T/2] 均匀采样
- 返回 loss, noise_pred, x_t, t 四元组（用于辅助损失计算 pred_x0）

### 6.2 方向损失 (λ=0.3)

```
wp_vec = pred_wp[:,:,-1,:] − pred_wp[:,:,0,:]
cos_sim = cosine(wp_vec, goal_feat)
dir_loss = (1 − cos_sim).clamp(min=0).mean()
```

要求 net displacement 朝向 goal。不附加 safety gate。

### 6.3 反向惩罚 (λ=0.2)

```
wrong_dir = relu(−cos_sim).mean()
```

当 waypoint 方向与 goal 完全相反时额外惩罚。

### 6.4 幅度惩罚 (λ=0.2)

```
mag = wp_vec.norm
penalty = relu(0.15 − mag).mean()
```

要求 8 步总位移 ≥ 0.15m。

### 6.5 平滑性惩罚 (λ=0.05)

```
diff = pred_wp[:,:,1:,:] − pred_wp[:,:,:-1,:]
penalty = (diff²).mean()
```

相邻 waypoint 之间位移差过大时惩罚，压制锯齿。

### 6.6 碰撞惩罚 (λ=0.05)

```
nearest_obs = agent_feat[:, :, 52]
penalty = relu(0.3 − nearest_obs).mean()
```

最近障碍物距离 < 0.3m 时惩罚。简化版：不查 LiDAR bins，不拘 is_mobile 限定。

---

## 7. 推理

### 7.1 推理调度

```
每 chunk_size=8 步 (约 0.13s) 或任意 agent waypoint 为空时:
  1. StateSerializer.serialize()
  2. LLM forward → hidden states
  3. DDIM 200 步去噪 → local waypoints
  4. cos/sin 旋转 → global waypoints
  5. WaypointTracker.set_waypoints()
```

### 7.2 WaypointTracker

纯算法，不做第二层 AI：
- 逐个 waypoint 跟随，到达阈值 0.03m 弹到下一 waypoint
- 速度 = min(cruise_speed, dist/dt*0.8, max_step/dt)
- dist < 0.3m 时额外减速：`min(speed, dist*4.0)`
- 所有 waypoint 耗尽后速度 = (0,0)

### 7.3 VLA 与 State Machine 职责划分

| Agent State | VLA 控制? | 行为 |
|-------------|-----------|------|
| CRUISE, PREQUEUE, QUEUING | ✓ | VLA 生成 waypoint 驱动机器人移动 |
| IDLE, LOADING, HALT | ✗ | 速度 = (0, 0) |

State machine 负责：任务分配/港口准入/队列管理/操作计时。VLA 负责移动路径。

### 7.4 DDIM 采样

```
eta=0 (确定性), ddim_steps=200
从 N(0,I) 出发, 200 步迭代去噪 → local waypoints
```

---

## 8. 推理命令

```bash
# 评估 Stage N
sbatch eval.sh <stage>

# fresh 推理
sbatch eval.sh fresh

# 自定义 checkpoint
sbatch eval.sh 3 --vla-checkpoint weights/stage_fresh
```

eval 运行：先 VLA 推理 N 分钟，后 Expert baseline N 分钟。quick/fresh 模式跳过 baseline。

VLA 推理 timeout=(N*60+60) 秒，Expert baseline timeout=(N*60*5+60) 秒（5x 冗余）。

---

## 9. CLI 参数

| 参数 | 说明 |
|------|------|
| `--vla` | 启用 VLA 模式 |
| `--vla-model ID` | OpenVLA model (默认 mock) |
| `--vla-device DEV` | cpu / cuda |
| `--vla-checkpoint DIR` | 加载 encoder.pt + diffusion_head.pt + lora_adapter/ |
| `--vla-collect` | 数据采集模式 |
| `--mix expert:policy:random` | DAgger mix 比例 |

---

## 10. 设计决策

| 决策 | 理由 |
|------|------|
| 保留 Llama2-7B 主干 (frozen + LoRA) | Open X-Embodiment 预训练的 state→action 映射内化；multi-head attention 天然支持多 agent 协调 |
| 移除视觉 token (SigLIP) | 59 维特征 + text prompt 覆盖所有空间信息；低质量 top-down 渲染图贡献极低 |
| 用 DDPM 替代 BC MSE | 支持多模态动作分布，避免平均化导致撞墙 |
| Action chunking (8 步) | RT-2/ACT/Diffusion Policy 成熟范式；WaypointTracker 纯算法避免 LLM 级误差累积 |
| Gradient checkpointing | 5-agent + LoRA 训练必须开，否则 OOM |
| Alpha_bar 预计算 | 避免每 batch 重复构造 1000 维的 cosine schedule |

---

## 11. 已知问题

| 严重度 | 描述 | 位置 |
|--------|------|------|
| Low | `_filter_waypoint_direction` 名称 misleading（实际过滤碰撞帧） | `train.py` |
| Low | `DataCollector.step()/should_collect()` 死代码 | `data_collector.py` |
| Low | 碰撞过滤用字符串匹配 `AA=0 AO=0` | `train.py` |
| Info | Feature indices 14-15 (goal_heading) 训练推理均未使用 | `state_serializer.py` |
| Info | 离散 action mode 的 bin 范围错误（未修复，unused path） | `data_collector.py` |
| Info | Flash Attention 2 自动检测但 superPOD 未安装 | `train.py` |
