# Dorabot Minions — 多机器人调度 VLA 模拟器

基于 Box2D 物理引擎的多机器人物流分拣模拟器，已从 Python 2.7 升级至 Python 3.10，集成 **OpenVLA-7B** 架构用于端到端多机器人控制研究。

**原始项目**: Dorabot Inc. (Shenzhen) — 许可见 LICENSE。

---

## 项目结构

```
dorabot_minions-master/
│
├── README.md                         ← 本文档
├── AGENTS.md                         ← OpenCode agent 参考
├── USER_MANUAL.md                    ← 原始用户手册（命令行、键鼠控制）
├── setup_superpod.sh                 ← superPOD 一键环境搭建
├── .gitignore
│
├── models/                           ← （本地）下载的模型权重目录
│   └── SmolLM2-135M-Instruct/        ← 轻量测试模型（270MB）
│
├── src/
│   ├── simulator.py                  ← 主入口
│   ├── control.py                    ← 交互式控制面板（cmd2）
│   ├── server.py                     ← 中央服务器（任务分配）
│   ├── visualisation.py              ← Pygame 可视化
│   ├── interaction_handler.py        ← 命令行 & 交互处理
│   ├── config.json                   ← 全局配置
│   │
│   ├── agents/                       ← 机器人 Agent 实现
│   │   ├── agent.py                  ← Agent 基类
│   │   ├── agent_state_machine.py    ← 高质量免货状态机
│   │   ├── naive_agent.py            ← NaiveAgent 实现
│   │   └── sensor.py                 ← 传感器模拟
│   │
│   ├── global_planners/              ← 全局路径规划器
│   │   ├── global_planner.py         ← 基类
│   │   ├── sample_global_planner.py  ← SimpleAStar
│   │   ├── layered_astar_planner.py  ← LayeredAStar
│   │   ├── rrtstar_planner.py        ← RRTStar
│   │   ├── rrtstar_helper.py         ← RRT 辅助函数
│   │   ├── multiagent_planner_local_entry.py
│   │   └── user/                     ← 用户自定义全局规划器
│   │
│   ├── local_planners/               ← 局部避障规划器
│   │   ├── local_planner.py          ← 基类
│   │   ├── dull_local_planner.py     ← DullPlanner
│   │   ├── virtual_force_planner.py  ← 虚拟力场法
│   │   ├── rvo_planner.py            ← RVO
│   │   ├── hrvo_planner.py           ← HRVO
│   │   ├── DD_planner.py             ← DD Planner
│   │   ├── flc_local_planner.py      ← FLC
│   │   └── user/                     ← 用户自定义局部规划器
│   │       └── simple_local_planner.py
│   │
│   ├── multiagent_global_planners/   ← 多智能体协同全局规划
│   │   ├── multiagent_planner.py     ← 基类
│   │   ├── marrtstar_planner.py      ← MARRTStar
│   │   ├── inash_planner.py          ← iNashRRT
│   │   ├── lane_planner.py
│   │   └── user/
│   │       └── simple_multi_planner.py
│   │
│   ├── vla/                          ← ★ VLA/AI 接口层
│   │   ├── __init__.py
│   │   ├── state_serializer.py       ← 模拟器状态 → VLA 输入（文本+特征）
│   │   ├── waypoint_tracker.py       ← waypoint 序列 → 插值 → (vx, vy)
│   │   ├── model_loader.py           ← OpenVLA-7B 加载 + MockVLAPolicy
│   │   ├── vla_controller.py         ← 推理调度、任务分配同步
│   │   ├── data_collector.py         ← 专家轨迹采集
│   │   └── train.py                  ← 训练脚本（superPOD 用）
│   │
│   ├── task_managers/                ← 任务分配
│   ├── setup_environment/            ← 环境配置（端口、障碍物）
│   ├── representation/               ← 地图表示（GridMap, ContinuousSpace）
│   ├── template/                     ← 自定义规划器模板
│   ├── protocol/                     ← protobuf 数据协议
│   └── map/                          ← 预定义障碍物地图
│
├── uml/                              ← UML 图
└── venv/                             ← Python 虚拟环境（不纳入版本控制）
```

---

## VLA 架构设计

### 一句话总结

**用 OpenVLA-7B 的预训练 Llama2-7B 主干作为"协调脑"，抛弃其原始 SigLIP 视觉编码器，换成自定义 MLP 编码器处理 55 维结构化特征，输出 8-step action chunking 的 waypoint 序列。**

### 为什么这样设计

| 设计决策 | 理由 |
|---|---|
| 不用 SigLIP 视觉编码器 | SigLIP 在桌面操作 RGB 图上预训练，与 top-down 仓库视角不兼容 |
| 用 MLP 编码器处理结构化特征 | 比渲染图更精确（位置、LiDAR、邻居），训练效率高 |
| 保留 Llama2-7B 主干 | 已在 Open X-Embodiment（百万级机器人轨迹）上微调，内化"状态→动作"映射 |
| Waypoint tracker 用纯算法 | 不引入第二层 AI，避免误差累积 |
| Action chunking（8 步） | RT-2/ACT/Diffusion Policy 级别的成熟范式 |

### 数据流

```
┌──────────────────────────────────────────────────────────────┐
│                    simulator.py step()                        │
│                                                               │
│  1. Agent 传感器数据 → observe() → 状态机分配任务              │
│                                                               │
│  2. VLAController.step()                                      │
│     ├── StateSerializer.serialize()                           │
│     │     ├── agents: 位置, 角度, 状态, LiDAR(16bin), 邻居     │
│     │     ├── ports: 位置, 队列                           │
│     │     └── obstacles                                      │
│     │     → text_prompt + 55-dim feature vector × 12 agents   │
│     │                                                        │
│     ├── OpenVLAPolicy.predict()                               │
│     │     ├── MLP encoder: [batch, 12, 55] → [batch, 12, 4096]│
│     │     ├── Tokenizer: text → [batch, seq, 4096]        │
│     │     ├── Llama2-7B (frozen + LoRA): forward pass          │
│     │     └── Action head: [batch, 12, 4096] → [batch,12,16]   │
│     │     → {(Δx₁,Δy₁)...(Δx₈,Δy₈)} per agent                  │
│     │                                                        │
│     └── WaypointTracker × N agents                            │
│           └── 8 waypoints → 线性插值 → (vx, vy) 每步          │
│                                                               │
│  3. Box2D physics step(speed)                                  │
└──────────────────────────────────────────────────────────────┘
```

### VLA 输入特征结构（55 维 / agent）

| 特征 | 维度 | 说明 |
|---|---|---|
| 位置 | 2 | (x, y) |
| 朝向 | 2 | (cos θ, sin θ) |
| 速度 | 1 | m/s |
| 状态 one-hot | 6 | IDLE/LOADING/QUEUING/HALT/CRUISE/PREQUEUE |
| 载货 | 1 | 0/1 |
| 目标偏移 | 2 | (dx_goal, dy_goal) |
| 目标朝向 | 2 | (cos φ_goal, sin φ_goal) |
| 邻居（×4） | 20 | (dist, cos rel_angle, sin rel_angle, cos hd_diff, sin hd_diff) × 4 |
| LiDAR 降采样 | 16 | 512 rays → 16 bins |
| 最近障碍 | 3 | (dist, cos obs_angle, sin obs_angle) |

### 动作输出格式（8-step action chunking）

每个 agent 输出 8 个 waypoint，每个 waypoint 是全局坐标系下的 (x, y)：

```json
{
  "0": [(3.1, 16.8), (3.2, 16.5), (3.4, 16.2), (3.5, 15.9),
         (3.6, 15.6), (3.7, 15.3), (3.8, 15.0), (3.9, 14.7)],
  "1": [(12.8, 7.2), (12.9, 7.0), (13.0, 6.8), (13.1, 6.6),
         (13.2, 6.4), (13.3, 6.2), (13.4, 6.0), (13.5, 5.8)]
}
```

WaypointTracker 将 8 个 waypoint 在 8 个模拟步中线性插值执行（每步 ~0.1s 的 PID 平滑跟随）。

---

## 训练流程

### 阶段 1：数据采集

**本机（Mac）运行：**

```bash
cd src
python simulator.py --vla-collect --agent 3 --port 2 2 -t 60
```

在后台以手写算法控制机器人，同时周期性记录 `(状态, 专家 waypoint)` 对。

**数据格式（JSONL）：**

```json
{
  "text_prompt": "Agent_0: pos=(3.00,17.00) heading=30deg ...",
  "features": {
    "agents": [[3.0, 17.0, ...], ...],
    "num_agents": 3,
    "ports": [...],
    "obstacles": [...]
  },
  "target_action": {
    "0": [(3.0,17.0), (3.1,16.5), ...],
    "1": [(13.0,7.0), (13.1,6.8), ...]
  },
  "agent_count": 3
}
```

每 8 步采集一次，10 分钟约产生 6000 条记录。

### 阶段 2：模型架构

训练时只更新以下可训参数（冻结 Llama2-7B）：

| 模块 | 参数量 | 训练方式 |
|---|---|---|
| MLP 编码器 (55→512→4096) | ~230K | 全参数训练 |
| Action Head (4096→16) | ~65K | 全参数训练 |
| Llama2-7B LoRA (rank=64) | ~40M | LoRA 微调（`q_proj`, `v_proj`） |

总可训参数 ≈ 40M，单卡 24GB 可训练。

### 阶段 3：loss 函数

**MSE（均方误差）**——MSE 用于直接回归 waypoint (x, y) 坐标。

### 阶段 4：课程训练

| Phase | 数据 | 目标 |
|---|---|---|
| 1 组 | 2 agent, 2 port | 验证接口和数据流 |
| 2 组 | 3-4 agent, 3-4 port | **主要训练目标**。协调、避障 |
| 3 组 | 5-6 agent | 泛化训练 |
| 4 组 | 5 agent 主训练 | 调参优化 |
| 5 组（压力）| 10 agent | 纯评估，不用于训练 |

### 阶段 5：superPOD 训练

```bash
cd src
python vla/train.py \
  --data data/trajectories/ \
  --model ../models/openvla-7b/ \
  --use-lora --quantize \
  --epochs 5
```

输出权重到 `weights/encoder.pt` + `weights/action_head.pt` + `weights/lora_adapter/`。

---

## 环境准备

### Mac（本地开发 & 数据采集）

**Python 3.10.X（推荐）**，macOS 15/16 with Apple Silicon。

```bash
# 创建虚拟环境
/opt/homebrew/opt/python@3.10/bin/python3.10 -m venv venv --without-pip
source venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# 安装依赖
pip install box2d-py pygame networkx shapely "protobuf<3.21" cmd2 numpy matplotlib pandas

# VLA 模块（本地推理 & 训练）
pip install torch transformers accelerate peft bitsandbytes

# 下载测试模型（可选：本地验证管线）
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct \
  --local-dir models/SmolLM2-135M-Instruct/
```

### superPOD（训练）

```bash
bash setup_superpod.sh
source venv/bin/activate

# 下载 OpenVLA-7B
huggingface-cli download openvla/openvla-7b \
  --include '*.json' '*.safetensors' 'tokenizer*' \
  --local-dir models/openvla-7b/
```

### 依赖总览

| 包 | 用途 |
|---|---|
| `box2d-py` | 物理引擎（2D） |
| `pygame` | 可视化 |
| `networkx` | 图算法 |
| `shapely` | 计算几何 |
| `protobuf<3.21` | 进程间数据协议 |
| `cmd2` | 交互式控制面板 |
| `numpy`, `matplotlib`, `pandas` | 数据分析 |
| `torch`, `transformers` | 模型推理/训练 |
| `accelerate`, `peft`, `bitsandbytes` | 训练加速 +量化 |

---

## 运行方式

### 1. 普通模拟器（手写算法）

```bash
cd src
python simulator.py                        # GUI 模式（Pygame）
python simulator.py -t 1                   # Headless 模式，1 模拟分钟
python simulator.py --default -t 10        # 默认配置，10 分钟

# 自定义参数
python simulator.py --agent 5 --port 3 2 --size 20 20 --gp LayeredAStar --lp DullPlanner
```

**GUI 快捷键：** P=暂停, D=显示详情, G=网格, Q=退出

### 2. 交互式控制面板

```bash
cd src
python control.py                          # 启动 cmd2 面板

# 面板内命令：
(cmd) new --title test1 --agent 5 --port 3 2
(cmd) control --title test1 --agent 0 --gp RRTStar
(cmd) template --gp MyAlgo                # 生成自定义规划器模板
(cmd) q                                   # 退出
```

### 3. VLA Mock 模式（验证接口）

```bash
cd src
python simulator.py --vla --agent 2 --port 2 2 -t 1
```

使用 `MockVLAPolicy`（直线前行）验证 VLA 管线：状态序列化 → 模拟推理 → waypoint 分发 → 物理执行。

### 4. VLA 数据采集模式

```bash
cd src
# 基础采集
python simulator.py --vla-collect --agent 3 --port 2 2 -t 60

# 大规模采集（superPOD 训练用）
python simulator.py --vla-collect --default -t 600
```

数据输出到 `src/data/trajectories/session_xxx.jsonl`。

### 5. VLA 真实模型推理

```bash
cd src
# 待训练完成后，用训练好的权重运行
python simulator.py --vla -t 1
```

需先在 `model_loader.py` 中加载训练好的 `encoder.pt` + `action_head.pt` + `lora_adapter/`。

### 6. 训练

```bash
# 本地小模型验证（135M，跑通流程）
cd src
python vla/train.py \
  --data data/trajectories/ \
  --model ../models/SmolLM2-135M-Instruct/ \
  --epochs 2 --max-samples 100

# superPOD 正式训练（OpenVLA-7B + LoRA + 4bit）
python vla/train.py \
  --data data/trajectories/ \
  --model ../models/openvla-7b/ \
  --use-lora --quantize \
  --epochs 5
```

---

## 技术方案说明

### 为什么选择 OpenVLA-7B 而非通用 LLM

| | Qwen2.5-7B | OpenVLA-7B |
|---|---|---|
| 预训练 | 通用文本语料 | 通用文本 + Open X-Embodiment (1M+ 机器人轨迹) |
| 动作理解 | 需从头学 | 已内化"状态→动作"映射 |
| 多智能体协调 | 零知识 | 可从多指灵巧手规划迁移 |
| 收敛速度 | 慢 | 快 |

### 为什么不用原始视觉编码器

OpenVLA 的 SigLIP 编码器（370M 参数）在真实 RGB 图像（BridgeData V2 — 桌面抓取）上预训练。仓库 top-down 视图与此领域差距巨大，强行使用会产生领域偏移。替代的 MLP 编码器只用了 230K 参数，直接从结构化特征学习，更高效。

### Loss 函数选择

使用 MSE 直接回归 waypoint 坐标。稀疏环境 + 多最优解的特性由 Diffusion Policy 处理（可选开关），当前默认使用 BC 先快速验证管线。

---

## 引用

| 论文 | 说明 |
|---|---|
| Ng 1999 (PBRS) | 势能奖励不改变最优策略 |
| Andrychowicz 2017 (HER) | Hindsight Experience Replay — 稀疏奖励处理 |
| Chi 2023 (Diffusion Policy) | 扩散模型用于机器人动作生成 |
| Zheng 2024 (OpenVLA) | OpenVLA-7B 架构与预训练范式 |
| Lowe 2017 (MADDPG) | 多智能体 Centralized Critic |
