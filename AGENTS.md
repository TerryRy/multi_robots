# Dorabot Minions 多机器人模拟器

**Python 3**（已从 Python 2.7 升级）物流分拣场景的多机器人调度模拟器。

## 快速开始

```bash
cd src
python simulator.py                    # 默认模拟器
python simulator.py -t 60              # 无界面模式，运行 60 分钟（模拟时间）
python control.py                       # 交互式控制面板（cmd2）
```

## 架构

- **`src/simulator.py`** — 主入口；创建 Box2D 物理世界、环境、智能体、规划器
- **`src/server.py`** — 中央服务器；负责任务分配（NaiveTaskManager）和多智能体全局规划器的生命周期
- **`src/control.py`** — 交互式控制面板（cmd2）；通过多进程 Queue 与模拟器通信

### 规划器架构（双层）

每个智能体拥有一组（全局规划器, 局部规划器）。

| 层 | 基类 | 目录 | 内置实现 |
|---|---|---|---|
| 全局（单智能体） | `GlobalPlanner` | `global_planners/` | `SimpleAStar`, `LayeredAStar`, `RRTStar` |
| 全局（协同） | `MultiAgentPlanner`（继承 GlobalPlanner） | `multiagent_global_planners/` | `MARRTStar`, `INashRRT` |
| 局部 | `LocalPlanner` | `local_planners/` | `DullPlanner`, `VirtualForcePlanner`, `FLCPlanner`, `RVOPlanner`, `HRVOPlanner`, `DDPlanner` |

模板位于 `src/template/`。自定义规划器放入 `user/` 子目录（例如 `global_planners/user/`），会被自动发现。

### 关键依赖

- Python 3（项目已从 Python 2.7 迁移）
- swig（>= 3.0.10 on macOS），pybox2d（源码安装），pygame，networkx，shapely，protobuf，cmd2，enum34

## 配置

- **`src/config.json`** — 环境尺寸、智能体/港口数量、速度、分辨率等
- 通过命令行标志 `--agent 5 --port 3 2 --size 20 20 --gp LayeredAStar --lp DullPlanner` 覆盖
- **`resolution: 1` 是推荐值**（README 强调）
- 港口/障碍物数量必须适配地图尺寸（宽度至少为 `max(ports)*3 + 2`，高度至少为 5）

## 自定义

- **全局规划器：** 子类化 `GlobalPlanner`（单智能体）或 `MultiAgentPlanner`（协同）；实现 `compute_path(self, position, goal_pose, environment, sensor_observation)` → `list of Point`
- **局部规划器：** 子类化 `LocalPlanner`，实现 `compute_plan(self, position, velocity, environment, sensor_observation, global_planner_path)` → `(vx, vy)` 元组
- **任务分配：** 子类化 `TaskMananger`
- **智能体行为：** 修改/扩展 `agent_state_machine.py` 中的状态机
- 使用 `template` 命令通过控制面板生成脚手架文件

## 端口放置

在 `simulator.py` 的 `set_environment()` 中设置：
- `setup_loading_ports_on_row()` / `setup_loading_ports_on_horizontal()`
- `setup_unloading_ports_on_bottom()` / `setup_unloading_ports_in_mid()`

## 测试

- 未找到正式的测试框架或测试套件
- 只有 `src/gridmap_test.py`（一次性的手动测试）
- 无 lint/typecheck 配置

## 注意事项

- Pygame 可视化仅使用内置基本图形，坐标取整为整数
- 每模拟秒步进 = `1.0/steps_per_sec`（config.json 中默认 60）
- agent.angle 每步在 `simulator.step()` 中从 linear_velocity 重新计算（非差速轮模式）
- 如果遇到 macOS 上 `matplotlib`/`pygame` 导致的段错误：将 `matplotlibrc` 中的 `backend` 设为 `agg`
