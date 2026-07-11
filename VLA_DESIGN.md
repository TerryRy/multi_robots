# VLA 模块设计文档

## 目录

1. [架构总览](#1-架构总览)
2. [模型架构](#2-模型架构)
3. [输入特征规范 (59 维)](#3-输入特征规范-59-维)
4. [数据采集](#4-数据采集)
5. [训练流程](#5-训练流程)
6. [损失函数](#6-损失函数)
7. [推理流程](#7-推理流程)
8. [Checkpoint 管理](#8-checkpoint-管理)
9. [状态机集成](#9-状态机集成)
10. [CLI / 接口](#10-cli--接口)
11. [已知问题与设计决策](#11-已知问题与设计决策)

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     simulator.py step()                          │
│                                                                  │
│  1. 512-ray LiDAR 射线 → agent.observe()                        │
│                                                                  │
│  2. State Machine                                                │
│     agent.state_machine.next_state() → CRUISE/QUEUING/LOADING…   │
│                                                                  │
│  3. VLAController.step()                                         │
│     ├── StateSerializer.serialize(simulator)                     │
│     │     ├── agents: 59-dim per agent                           │
│     │     ├── ports: 位置 + 队列 + items                        │
│     │     ├── obstacles: 位置 + 尺寸                            │
│     │     ├── text_prompt  (自然语言摘要)                        │
│     │     └── features_dict (数值特征)                           │
│     │                                                            │
│     ├── SimulatorRenderer.render() → 224×224 RGB (可选)         │
│     │                                                            │
│     ├── OpenVLAPolicy.predict()                                  │
│     │     └── features → MLP Encoder → Llama2-7B(LoRA)          │
│     │         → DiffusionActionHead (DDIM)                       │
│     │         → 8 local waypoints → 8 global waypoints          │
│     │                                                            │
│     └── WaypointTracker × N agents                               │
│           └── 8 global waypoints → 线性插值 → (vx, vy) 每步    │
│                                                                  │
│  4. Box2D 物理步进 (speed)                                       │
└─────────────────────────────────────────────────────────────────┘
```

**设计原则**：VLA 负责高层 waypoint 规划，WaypointTracker 是纯算法执行器，不做第二层 AI。避免误差累积。

---

## 2. 模型架构

### 2.1 组件总览

```
Image (224×224)   59-dim features × 12 agents       Text Prompt
     │                     │                             │
     ▼                     ▼                             ▼
  SigLIP*             AgentFeatureEncoder           Tokenizer*
  (frozen)            (trainable, ~2.4M)           (frozen)
     │                     │                             │
     ▼                     ▼                             ▼
  Projector*           agent_embeds                text_embeds
  (frozen)          [1, N, 4096]              [1, T, 4096]
     │                     │                             │
     └─────────────────────┼─────────────────────────────┘
                           │ concat
                           ▼
                   ┌───────────────┐
                   │  Llama2-7B    │  ← frozen base + LoRA (rank=128, q_proj, v_proj, ~40M)
                   │  (LoRA)       │
                   └───────────────┘
                           │
                           ▼ agent-position hidden states
                     [1, N, 4096]
                           │
                           ▼
                   ┌──────────────────┐
                   │ DiffusionActionHead│  ← trainable, ~14M
                   │ (DDPM + cross-attn)│
                   └──────────────────┘
                           │
                           ▼
                   local-frame waypoints
                   {(Δx₁,Δy₁)…(Δx₈,Δy₈)} × N agents
                           │
                           ▼ cos/sin 旋转
                   global-frame waypoints
```

\* SigLIP + Projector + Tokenizer 在训练和推理中都加载但不更新权重。

### 2.2 AgentFeatureEncoder

```
入口: 59-dim → 出口: 4096-dim (匹配 Llama2-7B hidden_size)

  Linear(59, 512) → ReLU → Linear(512, 512) → ReLU → Linear(512, 4096) → LayerNorm(4096)
  ~2.4M params, 全参数训练
```

### 2.3 DiffusionActionHead

- 4 个 CrossAttentionBlock（self-attn + cross-attn + FFN）
- 条件输入：LLM hidden states [B,N,4096] + goal_features [B,N,2]
- 输出：chunk_size × 2 = 16 维 local-frame waypoint 序列
- ~14M params，全参数训练
- 最终输出层：`normal_(mean=0, std=0.001)` 初始化

### 2.4 Llama2-7B (LoRA)

- **冻结**：Llama2-7B 主干（注意力 + FFN + embedding）
- **微调**：LoRA rank=128 on attention layers (`q_proj`, `v_proj`)，~40M params
- 作用：multi-agent cross-attention learning。同一 batch 内 N 个 agent 的 token 在 LLM 内部互相 attend，学出协调策略。

### 2.5 可训练参数汇总

| 模块 | 参数量 | 状态 |
|------|--------|------|
| AgentFeatureEncoder | ~2.4M | 全参数训练 |
| DiffusionActionHead | ~14M | 全参数训练 |
| Llama2-7B LoRA (rank=128) | ~40M | LoRA 微调 |
| SigLIP + Projector + Tokenizer | ~370M + | 冻结 |
| Llama2-7B (except LoRA) | ~7B | 冻结 |
| **合计可训练** | **~56M** | |

---

## 3. 输入特征规范 (59 维)

每 agent 一个 59-dim 特征向量，构成 `[batch, N_agents, 59]` 的 tensor。

### 3.1 特征布局

| 索引 | 维度 | 字段 | 说明 | 数据来源 |
|------|------|------|------|----------|
| 0-1 | 2 | position | (x, y) 世界坐标 | Box2D body.position |
| 2-3 | 2 | heading | (cos θ, sin θ) | Box2D body.angle |
| 4 | 1 | speed | 当前速率 (m/s) | agent.speed |
| 5-10 | 6 | state one-hot | IDLE/LOADING/QUEUING/HALT/CRUISE/PREQUEUE | agent.state |
| 11 | 1 | carrying | 是否载货 (0/1) | agent.carrying_item |
| 12-13 | 2 | goal offset | (dx_goal, dy_goal) 全局坐标系 | agent.destination_location − position |
| 14-15 | 2 | goal heading | (cos φ, sin φ) 全局坐标系 | atan2(dy_goal, dx_goal) |
| 16-35 | 20 | neighbors ×4 | (dist, cos rel_angle, sin rel_angle, cos hd_diff, sin hd_diff) × 4 | sensor 4m 范围内最近 4 agent |
| 36-51 | 16 | LiDAR bins | 512-ray → 16-bin min downsampling | agent.ray_length_list |
| 52-54 | 3 | nearest obstacle | (dist, cos obs_angle, sin obs_angle) | ray_length_list 全局 min |
| 55-58 | 4 | nearest port | (dist, cos rel_angle, sin rel_angle, is_loading) | 环境内所有 port 计算最近 |

### 3.2 关键索引在代码中的使用

| 索引 | `state_serializer.py` (构建) | `train.py` (训练) | `model_loader.py` (推理) |
|------|------------------------------|-------------------|--------------------------|
| 0-1 | `pos[0], pos[1]` | `agent_feat[:,:,:2]` | `agent_features[0,i,0:2]` |
| 2 | `cos(heading_rad)` | `agent_feat[:,:,2:3]` | `agent_features[0,:,2:3]` |
| 3 | `sin(heading_rad)` | `agent_feat[:,:,3:4]` | `agent_features[0,:,3:4]` |
| 7 | QUEUING one-hot | `is_mobile` (collision loss) | — |
| 9 | CRUISE one-hot | `is_mobile` (collision loss) | — |
| 12-13 | `dx_goal, dy_goal` (global) | `agent_feat[:,:,12:14]` → local frame | `agent_features[0,:,12:14]` → local frame |
| 36-51 | 16-bin LiDAR | `agent_feat[:,:,36:52]` | — |
| 52 | nearest obs dist | `agent_feat[:,:,52]` | — |

### 3.3 坐标帧转换

训练和推理中的 goal 坐标帧统一为局部帧：

```
global (dx_goal, dy_goal)  →  旋转矩阵  →  local (dx_local, dy_local)

  local_x =  global_dx * cos(θ) + global_dy * sin(θ)
  local_y = -global_dx * sin(θ) + global_dy * cos(θ)
```

训练目标 waypoint 同样转换为局部帧后喂入 DDPM loss。

推理时 DDIM 输出局部帧 waypoint，再逆旋转回全局帧：
```
global_x = px + local_dx * cos(θ) − local_dy * sin(θ)
global_y = py + local_dx * sin(θ) + local_dy * cos(θ)
```

---

## 4. 数据采集

### 4.1 采集流程

```
python simulator.py --vla-collect --agent <N> --port <L> <U> -t <minutes>
```

采集模式 (`VLAController.collect_data=True`) 下：
- Agent 由手写规划器（默认 SimpleAStar + DullPlanner）控制
- 每 `chunk_size`（8）步，记录观测状态（text_prompt + features）
- 等待 `chunk_size × stride` 步后，提取 Agent 实际走过的全局坐标作为 expert waypoint
- 写入 JSONL 文件 + 可选 PNG 截图

### 4.2 数据格式 (JSONL)

每行一条记录：

```json
{
  "text_prompt": "Timestamp: 120.00s | Packages: 45 | Collisions: AA=0 AO=0\n  Agent_0: pos=...",
  "features": {
    "agents": [[3.0, 17.0, ...], ...],
    "num_agents": 2,
    "ports": [1.0, 17.0, 1.0, 0.0, 5.0, ...],
    "obstacles": [...]
  },
  "target_action": {
    "0": [(3.1, 16.8), (3.2, 16.5), ..., (3.9, 14.7)],
    "1": [(12.8, 7.2), (12.9, 7.0), ..., (13.5, 5.8)]
  },
  "image_path": "session_xxx/frame_xxx.png",
  "agent_count": 2
}
```

- `target_action`: 每个 agent 的 8 个全局坐标 waypoint（agent ID 为 key）
- 坐标帧：**全局坐标系**（Box2D 世界坐标）
- `features.agents`: 59-dim per-agent 特征（旧数据为 55-dim，训练时自动 padding 到 59）

### 4.3 Hindsight DAgger

`collect_data.sh` Stage 2-4 使用混合控制策略：

| Stage | mix | 含义 |
|-------|-----|------|
| 1 | 100% expert | 纯手写规划器 |
| 2 | 60% expert + 30% policy + 20% random | policy = Stage 1 VLA 模型 |
| 3 | 40% expert + 40% policy + 20% random | policy = Stage 2 VLA 模型 |
| 4 | 20% expert + 60% policy + 20% random | policy = Stage 3 VLA 模型 |

Expert 和 random 的控制在本地执行，policy 控制需要加载对应阶段的 checkpoint 做推理。

### 4.4 stride 参数

默认 stride=1（每 8 步采集 8 个连续 waypoint）。设置为 4 时，每 8×4=32 步采集 8 个 stride=4 的 waypoint（增大时间跨度，降低采样密度）。

---

## 5. 训练流程

### 5.1 数据集加载

```python
class VLADataset(Dataset):
    # 从 JSONL 文件加载
    # 支持 curriculum mixing: --data-mix "stage_1:0.2,stage_2:0.8"
    # 自动 padding: 55-dim → 59-dim (尾部补 4 个 0)
    # 碰撞过滤: 丢弃 text_prompt 中包含 Collisions: AA>0 或 AO>0 的帧
    # max_samples: 限制样本数
```

### 5.2 Curriculum 训练（4 阶段）

| Stage | 数据 | Epochs | Batch | LR | 初始化 |
|-------|------|--------|-------|-----|--------|
| 1 | stage_1 (2 agents) | 8 | 4 | 3e-5 | 从头训练 |
| 2 | 20% stage_1 + 80% stage_2 | 8 | 4 | 3e-5 | 加载 stage_1 |
| 3 | 15%+15%+70% | 8 | 4 | 3e-5 | 加载 stage_2 |
| 4 | 10%+10%+10%+70% | 8 | 4 | 3e-5 | 加载 stage_3 |

每阶段混合旧数据 10-20% 防止灾难性遗忘。

### 5.3 优化配置

- Optimizer: AdamW, lr=3e-5, weight_decay=0.01
- Scheduler: Cosine, 5% warmup
- Gradient clipping: max_norm=1.0
- DDP: `torchrun --nproc_per_node=1`（单卡）

### 5.4 训练循环

```
for batch in DataLoader:
    1. 视觉编码 (SigLIP, frozen, torch.no_grad)
    2. agent_embeds = Encoder(agent_feat)
    3. text_embeds = Tokenizer(text_prompts)
    4. combined = cat(visual_tokens + agent_embeds + text_embeds)
    5. LLM forward (with_grad for LoRA)
    6. hidden = LLM_hidden_at_agent_positions
    7. target_flat = global_waypoints → local_frame (旋转 + 减位置)
    8. DDPM_loss = compute_diffusion_loss(target_flat, hidden, goal_feat_local)
    9. collision_penalty = penalize low-LiDAR + near-obstacle + is_mobile
    10. loss = DDPM_loss + 0.05 * collision_penalty
    11. backward + clip_grad + step
```

---

## 6. 损失函数

### 6.1 主损失：DDPM Diffusion Loss

```
loss_diffusion = MSE(noise_pred, noise)
```

- T=1000，cosine beta schedule
- t ∈ [0, T/2] 随机采样（去噪前 500 步）
- 预测目标：加入目标 waypoint 的噪声，而非直接回归坐标
- 等价于估计动作分布的 score function，天然支持多模态

### 6.2 辅助损失：碰撞惩罚

```
goal_angle = atan2(goal_local_y, goal_local_x)
lidar_bin = ((goal_angle + π/2) / π × 16).clamp(0,15)
goal_lidar = lidar_feat[bin]                         # 目标方向上的 LiDAR 距离
nearest_obs = agent_feat[i, 52]                      # 最近障碍物距离
is_mobile = state[QUEUING] + state[CRUISE]           # 仅在移动时惩罚

collision_penalty = relu(0.5 - goal_lidar) + relu(0.5 - nearest_obs)
loss = diffusion_loss + 0.05 × collision_penalty × is_mobile
```

### 6.3 数据级过滤

`train.py` 的 `_filter_waypoint_direction` 在加载时丢弃碰撞帧：
- 检查 text_prompt 中的 `Collisions: AA=... AO=...` 行
- 如果 `AA=0 AO=0` 不成立 → 丢弃
- 保护机制：样本 < 20 条或过滤后 < 10% 时跳过

---

## 7. 推理流程

### 7.1 推理调度

```
VLAController.step(simulator):
  1. 检查是否需要推理
     - 距上次推理 ≥ chunk_size(8) 步 → 需要
     - 任一 agent 的 waypoint 队列为空 → 需要
  2. StateSerializer.serialize() → text_prompt + features
  3. SimulatorRenderer.render() → 224×224 图像
  4. OpenVLAPolicy.predict() → 各 agent 的 8 个全局 waypoint
  5. _dispatch_waypoints() → WaypointTracker[aid].set_waypoints()
  6. 对每个 agent (仅在 CRUISE/QUEUING/PREQUEUE 状态时):
       WaypointTracker.compute_velocity(position) → (vx, vy)
       设置 agent.linear_velocity
  7. 对 LOADING/IDLE/HALT 状态的 agent: linear_velocity = (0, 0)
```

### 7.2 WaypointTracker

```
每步: agent.position → waypoint_queue[0] 的距离 < 0.03m? → pop
      dist = distance to current target waypoint
      speed = min(cruise_speed, dist/dt × 0.8, max_step/dt)
      if dist < 0.3: speed = min(speed, dist × 4)  # 靠近目标减速
      (vx, vy) = normalize(direction) × speed
```

### 7.3 推理时的 DDIM 采样

```
eta = 0.0（确定性采样）
init: x₀ ~ N(0, I)  [1, N, 16]
for 50 steps (从 T/2=500 倒序):
  noise_pred = DiffusionHead(x_t, condition, goal_feat)
  去噪: x_{t-1} = c₁ * pred_x₀ + c₂ * noise_pred
output: [1, N, 16] → 每个 agent 的 8 个 local-frame waypoint
转全局: gx = px + local_dx*cos(θ) - local_dy*sin(θ)
```

### 7.4 VLA 与 State Machine 的职责划分

| Agent State | VLA 控制? | 说明 |
|-------------|-----------|------|
| IDLE | ✗ (vel=0) | 无任务，等待状态机分配 |
| CRUISE | ✓ | VLA 生成 waypoint 驱动机器人移动 |
| PREQUEUE | ✓ | 等待进入港口队列，微调位置 |
| QUEUING | ✓ | 移动到具体 slot 位置 |
| LOADING | ✗ (vel=0) | 港口操作中，必须静止 |
| HALT | ✗ (vel=0) | 紧急停止 |

State machine 负责：任务分配、港口准入、队列管理、操作计时。
VLA 负责：移动路径规划（仅在移动状态下）。

---

## 8. Checkpoint 管理

### 8.1 文件结构

```
weights/stage_1/
├── encoder.pt           # AgentFeatureEncoder 权重
├── diffusion_head.pt    # DiffusionActionHead 权重
└── lora_adapter/        # PEFT LoRA adapter
    ├── adapter_config.json
    └── adapter_model.safetensors
```

### 8.2 保存/加载接口

| 文件 | 保存 (`train.py`) | 加载 (`model_loader.py`) |
|------|-------------------|--------------------------|
| encoder.pt | `torch.save(encoder.state_dict())` | `torch.load(path); encoder.load_state_dict()` |
| diffusion_head.pt | `torch.save(diffusion_head.state_dict())` | `torch.load(path); diffusion_head.load_state_dict()` |
| lora_adapter/ | `llm.save_pretrained(dir)` | `PeftModel.from_pretrained(llm, dir)` |

encoder/diffusion_head 的 `input_dim` 和 `hidden_dim` 在训练和推理中保持一致（59 / 4096）。

### 8.3 模型加载流程

```python
# simulator.py
use_mock = cmd_args.vla_model is None  # --vla 不带 --vla-model → mock

# vla_controller.py
if use_mock:     → MockVLAPolicy (直线走向目标, 调试用)
else:            → load_openvla_policy(model_id, device)
                    → if checkpoint_dir: load_checkpoint(encoder+head+LoRA)
```

---

## 9. 状态机集成

### 9.1 VLA 模式下的状态转换

```
IDLE  → go_for_next_loading_task → CRUISE (task = go to loading port)
                                      │ VLA 控制移动到 loading port
                                      ▼
CRUISE → in_control_range → approaching → confirm_enter → QUEUING
                                      │ VLA 控制移动到 slot
                                      ▼
QUEUING → in_operation_zone → start_loading → LOADING
                                      │ vel = 0, port 操作 2 秒
                                      ▼
LOADING → operate_done → CRUISE (task = go to unloading port, carrying item)
                                      │ VLA 控制移动到 unloading port
                                      ▼
CRUISE → ... → LOADING → operate_done → CRUISE (task = go to loading port)
                                      │ 循环，每完成一次 unload PPH+1
                                      ▼
```

### 9.2 与普通模式的差异

普通模式 (`__update_agent_state`):
```python
agent.observe() → agent.plan() → agent.act()
# agent.plan() 调用 global planner + local planner
```

VLA 模式:
```python
agent.observe()
agent.state_machine.next_state()  # 仅状态转换, 不调 planner
VLAController 控制 velocity        # VLA 替代 planner
```

---

## 10. CLI / 接口

### 10.1 命令行参数

```
--vla              启用 VLA 模式 (默认用 MockVLAPolicy)
--vla-model ID     加载 OpenVLA-7B (如 openvla/openvla-7b)
--vla-device DEV   cpu / cuda
--vla-checkpoint DIR 加载训练好的 checkpoint (encoder.pt + diffusion_head.pt + lora_adapter/)
--vla-collect      数据采集模式 (手写 planner 驱动, 记录 expert 轨迹)
--mix RATIOS       采集模式下的混合控制比例 (expert:policy:random)
--vla-stride N     采集模式下的 waypoint 步距 (default 1)
```

### 10.2 SLURM 脚本

```bash
sbatch collect_data.sh <stage>        # 1-4, 数据采集
sbatch train.sh <stage>               # 1-4, 课程训练
sbatch eval.sh <stage> [minutes]      # 1-4, 模型评估
sbatch train.sh quick                 # OpenVLA-7B 快速验证 (5 epochs, 200 samples, ~30min)
sbatch eval.sh quick                  # 快速推理验证 (仅 VLA, 不比 expert)
```

---

## 11. 已知问题与设计决策

### 11.1 已知问题

| ID | 严重度 | 描述 | 位置 |
|----|--------|------|------|
| KN-1 | Low | `_filter_waypoint_direction` 名称误导——实际过滤碰撞帧，非 waypoint 方向 | `train.py:112` |
| KN-2 | Low | `DataCollector.step()` / `should_collect()` 未被调用，死代码 | `data_collector.py:66-70` |
| KN-3 | Low | 碰撞过滤用字符串匹配 `"AA=0 AO=0"`，格式化变动会失效 | `train.py:118` |
| KN-4 | Low | `_nearest_port` 未找到 port 时返回角度 0（forward），应返回 sentinel | `state_serializer.py:231` |
| KN-5 | Low | `_pending_state` 用第一个 agent 的 history 长度做全部 agent 的代理检查 | `vla_controller.py:119` |
| KN-6 | Info | Feat indices 14-15（goal_heading cos/sin 全局帧）从未在训练或推理中使用，冗余 | `state_serializer.py:297` |
| KN-7 | Info | 离散 action mode (`action_mode="discrete"`) 的 bin 范围错误：对全局坐标做 [−0.6,0.6] 裁剪 | `data_collector.py:87` |

### 11.2 设计决策

| 决策 | 理由 |
|------|------|
| 保留 SigLIP 视觉编码器 | 利用 OpenVLA 预训练的多模态能力；top-down 渲染图虽简但保留空间拓扑 |
| MLP 编码器替代视觉微调 | 230K 参数直接从结构化特征学习，比渲染+ViT 更高效精确 |
| Action Chunking (8 步) | ACT/Diffusion Policy 成熟范式；纯算法 WaypointTracker 避免 LLM 级误差累积 |
| DDPM 替代 BC MSE | 支持多模态动作分布（两条路径 → 不会取平均走中间撞墙） |
| LoRA rank=128 | 覆盖足够表达能力 + 单卡 24GB 可训练；q_proj/v_proj 覆盖关键注意力层 |
| 碰撞 loss 为辅 (λ=0.05) | 不影响主训练方向，仅做 mild push；太大会干扰 waypoint 精度 |
| queuing state 允许 VLA 控制 | 进入 slot 需要微调位置，不能完全停止 |

### 11.3 向后兼容

- 旧 55-dim 数据自动 padding 到 59-dim（尾部补 4 个 0）
- Feature dim 从数据自动检测 (`VLADataset._feature_dim`)
- 碰撞过滤有保护（样本数 < 20 或过滤后 < 10% 时跳过）
