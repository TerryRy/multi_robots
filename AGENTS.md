# Dorabot Minions 多机器人模拟器

**Python 3** 物流分拣场景的多机器人调度模拟器。Box2D 物理、双层规划器（全局+局部）。

## 关键路径

所有命令从 `src/` 目录运行。

```bash
cd src
python simulator.py                    # 默认（手写规划器）
python simulator.py -t 60              # 无界面，运行 60 模拟秒
sbatch train.sh fresh                  # SLURM 训练
sbatch eval.sh fresh                   # SLURM 评估
```

## VLA（当前 wheel 分支）

**模型输出**：5 robot × 8 步 × (v_forward, ω) = `[B, 5, 16]`

- `v_forward` 范围 0-3（来自专家线速度模长），精确无误
- `omega` 范围 -185~+185（来自 heading 变化/dt），存储时 ×0.1 平衡维度（`wheel_kinematics.py` 的 `OMEGA_SCALE=0.1`）
- 推理时 `omega_raw = omega_scaled / 0.1`，`heading += omega_raw × dt`
- `body.angle` 优先使用 `agent.wheel_heading`（`simulator.py:269`）

### 采集 → 训练 → 评估 流程

```bash
sbatch collect_data.sh 3               # 采集 stage_3 数据（expert-only，5 agents）
sbatch train.sh fresh                   # 从零训 12 epoch（stage_3 数据，7B+LoRA）
sbatch eval.sh fresh                    # 评估 5 分钟 vs Expert 基线
```

路径隔离：`data/trajectories_wheel/` + `weights_wheel/`，不污染 dev 分支。

### 文件结构

| 文件 | 作用 |
|------|------|
| `src/vla/wheel_kinematics.py` | `motion_to_control()` 采集用，`control_to_motion()` 推理用 |
| `src/vla/vla_controller.py` | 采集/推理入口，8 步 (v,ω) 缓冲逐步执行 |
| `src/vla/train.py` | 纯扩散 BC（无辅助损失） |
| `src/vla/model_loader.py` | OpenVLA-7B+LoRA、Mock 轻量模型 |
| `src/vla/diffusion_head.py` | CrossAttention + DDPM/DDIM |
| `src/vla/state_serializer.py` | 59 维特征编码 |
| `src/vla/inspect_data.py` | 数据质量检查 |

### 关键参数

- `wheel_base=0.3`（来自 DD_planner）
- `OMEGA_SCALE=0.1`：存储 ω×0.1 以平衡 v(0~3) 和 ω(-185~+185) 的尺度
- `chunk_size=8`：模型一次输出 8 步，每步用一对，缓冲空则重新推理
- 推理间隔：`chunk_size × dt` ≈ 8/60 秒 ≈ 每 133 毫秒推理一次

## 架构

- **`src/simulator.py`** — 主入口；创建 Box2D 世界、环境、agent、规划器
- **`src/server.py`** — 中央服务器；任务分配（NaiveTaskManager）+ 多智能体全局规划器生命周期
- **`src/control.py`** — 交互式控制面板（cmd2）；通过多进程 Queue 与模拟器通信
- **`src/config.json`** — 环境配置（边长、速度、分辨率等）

规划器分层：

| 层 | 基类 | 位置 | 实现 |
|---|---|---|---|
| 全局 | `GlobalPlanner` | `global_planners/` | SimpleAStar, LayeredAStar, RRTStar |
| 全局（协同） | `MultiAgentPlanner` | `multiagent_global_planners/` | MARRTStar, INashRRT |
| 局部 | `LocalPlanner` | `local_planners/` | DullPlanner, VirtualForcePlanner, RVOPlanner, HRVOPlanner, DDPlanner |

自定义规划器放入 `user/` 子目录自动发现。

## SLURM 脚本

| 脚本 | 命令 | 耗时 |
|------|------|------|
| `collect_data.sh 3` | 采集 5 agents 数据 | 12 模拟分钟 ≈ 5 分钟 |
| `train.sh fresh` | 7B+LoRA 从头训 12 epoch | ~5 小时 |
| `train.sh <N>` | 增量训 stage N（加载 stage N-1） | ~3 小时/epoch |
| `eval.sh fresh` | VLA 评估 5 分钟 | ~15 分钟 |
| `inspect_data.sh 3` | 数据质量检查 | < 1 分钟 |

## 注意事项

- **Pygame 可视化**：使用内置基本图形，坐标取整为整数
- **模拟步进**：`dt = 1.0/steps_per_sec`（默认 60）
- **Box2D 非差速**：当前模拟器为全向驱动（`body.angle = atan2(vy, vx)`），wheel 分支通过 `wheel_heading` 模拟差速
- **macOS 段错误**：matplotlib 后端设为 `agg`（`matplotlibrc`）
- **IO 优化**：训练时 images 未使用（`n_patches=0`），已跳过加载
- **数据格式**：JSONL 中 `target_action` 为 `{(v, ω)}` 元组，JSONL 文件批量移到 `data/trajectories_wheel/stage_N/`
- **未锁版本**：OpenVLA 预期 `transformers==4.40.1`，但环境 `4.39.3`，通常可用但有兼容警告
- **梯度 checkpoint**：`use_cache=True` 不兼容，已自动设为 `False`
