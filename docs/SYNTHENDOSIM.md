# SynthEndoSim — 血管介入仿真引擎文档

> 版本 0.1.0 | 适用场景：CAS / TAVR / 冠脉介入 RL 训练、评估、行为克隆

---

## 目录

1. [架构概览](#1-架构概览)
2. [快速开始](#2-快速开始)
3. [核心接口](#3-核心接口)
4. [子系统详解](#4-子系统详解)
   - 4.1 [VesselTree — 血管树](#41-vesseltree--血管树)
   - 4.2 [Target — 目标选择](#42-target--目标选择)
   - 4.3 [Device — 介入器械](#43-device--介入器械)
   - 4.4 [Start — 起始策略](#44-start--起始策略)
   - 4.5 [InterimTarget — 中间路标](#45-interimtarget--中间路标)
   - 4.6 [Pathfinder — 路径距离](#46-pathfinder--路径距离)
   - 4.7 [Terminal — 成功条件](#47-terminal--成功条件)
   - 4.8 [Truncation — 失败条件](#48-truncation--失败条件)
   - 4.9 [Reward — 奖励函数](#49-reward--奖励函数)
   - 4.10 [Observation — 观测构建](#410-observation--观测构建)
   - 4.11 [Imaging — 荧光成像](#411-imaging--荧光成像)
   - 4.12 [Info — 指标收集](#412-info--指标收集)
   - 4.13 [Visualisation — 可视化](#413-visualisation--可视化)
   - 4.14 [Physics Backend — 物理后端](#414-physics-backend--物理后端)
5. [工具模块](#5-工具模块)
   - 5.1 [StateRecorder — Episode 录制与回放](#51-staterecorder--episode-录制与回放)
6. [插件扩展指南](#6-插件扩展指南)
7. [配置系统](#7-配置系统)
8. [训练示例](#8-训练示例)
9. [评估协议](#9-评估协议)
10. [FAQ](#10-faq)

---

## 1. 架构概览

SynthEndoSim 是一个**全插件化**的血管介入仿真框架。每个子系统都是可独立替换的插件对象，通过依赖注入接入主环境。

```
SynthEndoEnv (Gymnasium 标准接口)
│
├── VesselTree   ── 血管拓扑（分支、分叉点、插入点）
├── Target       ── 目标选择（固定/随机/分支末端）
├── Device       ── 器械参数（导丝/导管型号）
├── Start        ── 起始策略（固定/随机推进/课程）
├── InterimTarget── 中间路标（分叉点/等距/阶段）
├── Pathfinder   ── 距离计算（欧氏/流形/Dijkstra）
├── Terminal     ── 成功判定（到达目标/路径完成）
├── Truncation   ── 失败判定（超时/无进展/越界）
├── Reward       ── 奖励函数（可组合）
├── Observation  ── 观测构建（Tip/Tracking/图像）
├── Imaging      ── 荧光渲染（DRR/Pillow/None）
├── Info         ── 指标收集（Step/Episode统计）
├── Visualisation── 可视化渲染（Null/MPL/OpenCV/Video）
└── PhysicsBackend── 物理仿真（SOFA/Mock）
```

**设计原则：**
- 每个子系统都有抽象基类（`base.py`）+ 若干实现
- 注册表（`_REGISTRY`）支持通过字符串名称查找
- `env.set_plugin(**kwargs)` 支持运行时热替换
- 配置驱动（YAML）或直接注入对象，两种用法任选

---

## 2. 快速开始

### 安装

```bash
cd synthendosim
pip install -e .
# SOFA 后端另需安装（见 install_env.sh）
```

### 最简使用（无需 SOFA）

```python
import synthendosim as ses

env = ses.make_env(config_dict={
    "physics": {"backend": "mock"},
    "anatomy": {
        "insertion_point": [-23.0, -180.0, -15.0],
        "insertion_direction": [0.0, 1.0, 0.0],
        "target_point": [-31.7, 70.3, 11.6],
    },
    "episode": {"max_steps": 300},
})

obs, info = env.reset(seed=42)
for _ in range(300):
    action = env.action_space.sample()          # 随机策略
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

### 插件注入（研究模式）

```python
import synthendosim as ses
from synthendosim.vesseltree import AorticArch
from synthendosim.target import BranchEndTarget
from synthendosim.interimtarget import BranchingPointTarget
from synthendosim.pathfinder import ManifoldPathfinder
from synthendosim.reward import ManifoldDistanceDelta, TargetReached, StepPenalty

# 程序化主动脉弓（TypeI，随机seed）
tree = AorticArch(arch_type="I", seed=42)
tree.reset()

env = ses.make_env(
    config_dict={"physics": {"backend": "mock"}, "episode": {"max_steps": 300}},
    target=BranchEndTarget(threshold_mm=8.0, branches=["left_common_carotid"]),
    interimtarget=BranchingPointTarget(threshold_mm=12.0),
    pathfinder=ManifoldPathfinder(tree.centerline_coordinates, curvature_weight=2.0),
    reward=ManifoldDistanceDelta(weight=1.0) + TargetReached(bonus=10.0) + StepPenalty(-0.01),
)
```

---

## 3. 核心接口

### `SynthEndoEnv`

继承自 `gymnasium.Env`，标准 Gym 接口。

```python
env = SynthEndoEnv(cfg, **plugins)

# 动作空间：每个器械 2 维 [-1, 1]
# [translation_scale, rotation_scale]
# 实际速度 = action * [MAX_TRANSLATION_mm_s, MAX_ROTATION_rad_s]
env.action_space   # Box(-1, 1, (2 * n_devices,), float32)

# 观测空间：Dict 或 Box（由 ObsConfig 决定）
env.observation_space

obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
frame = env.render()   # (H, W, 3) uint8 RGB，由 renderer 插件提供
env.close()

# 热替换插件（两个 episode 之间）
env.set_plugin(
    terminal=TargetReachedTerminal(threshold_mm=5.0),
    renderer=VideoRenderer("run.mp4"),
)
```

### `make()` / `make_env()` / `from_mesh()`

```python
# 预设名称
env = ses.make("SynthEndoSim-AorticArch-CAS-v1", anatomy_mesh="data/aorta.obj")

# Dict 配置
env = ses.make_env(config_dict={...}, **plugins)

# 最短路径：直接从 mesh
env = ses.from_mesh(
    mesh="data/aorta.obj",
    insertion_point=(-23, -180, -15),
    target_point=(-31.7, 70.3, 11.6),
    backend="sofa",
)
```

### 向量化环境

```python
vec_env = ses.make_vec_env(n_envs=8, config_dict={...})
# 兼容 stable-baselines3 VecEnv 接口
obs = vec_env.reset()
obs, rewards, dones, infos = vec_env.step(actions)
```

---

## 4. 子系统详解

### 4.1 VesselTree — 血管树

血管拓扑系统。提供分支结构、分叉点、插入点和网格路径。

#### 接口

```python
from synthendosim.vesseltree import VesselTree, Insertion
from synthendosim.vesseltree.util.branch import Branch, BranchWithRadii, BranchingPoint

tree.reset(episode=0, seed=42)   # 可能随机化解剖形态
tree.branches                    # Tuple[BranchWithRadii, ...]
tree.branching_points            # List[BranchingPoint]
tree.centerline_coordinates      # np.ndarray (N, 3)
tree.insertion                   # Insertion(position, direction)
tree.bbox_low, tree.bbox_high    # 解剖边界框
tree["left_common_carotid"]      # 按名称访问分支
tree.nearest_branch(point)       # 最近分支
tree.at_tree_end(point)          # 是否到达血管末端
```

#### 实现

| 类 | 说明 |
|---|---|
| `AorticArch(arch_type, seed)` | 程序化主动脉弓，7 种解剖变体 (I/II/IV/V/VI/VII) |
| `AorticArchRandom(arch_types, scale_xy, ...)` | 每 N episode 随机切换弓型+几何变换，用于多解剖泛化训练 |
| `VesselTreeFromMesh(mesh_path, insertion_position, branches)` | 从 OBJ/STL 加载，可附加手动 Branch 列表 |

```python
from synthendosim.vesseltree import AorticArch, AorticArchRandom, VesselTreeFromMesh

# 程序化 Type I（最常见）
tree = AorticArch(arch_type="I", seed=42)
tree.reset()

# 多解剖随机训练
tree = AorticArchRandom(
    arch_types=["I", "II"],
    scale_xy=[0.9, 1.0, 1.1],
    episodes_between_change=1,
)

# 从临床 CT 提取的网格
from synthendosim.vesseltree.util.branch import BranchWithRadii
import numpy as np
branches = [
    BranchWithRadii("aorta",         aorta_coords,  aorta_radii),
    BranchWithRadii("left_common_carotid", lcca_coords, lcca_radii),
]
tree = VesselTreeFromMesh(
    mesh_path="data/patient_001.obj",
    insertion_position=(-23, -180, -15),
    insertion_direction=(0, 1, 0),
    branches=branches,
)
```

#### 主动脉弓解剖类型

```
Type I  : Aorta → BCT(RCCA+RSA) → LCCA → LSA           ~75%
Type II : Aorta → BCT(RCCA+RSA+LCCA) → LSA              ~10%（牛型弓）
Type IV : Aorta → RSA → CO(RCCA+LCCA) → LSA
Type V  : Aorta → BCT → LSA → RSA（异位右锁骨下）
Type VI : Aorta → BCT → CO(RSA+LSA)
Type VII: Aorta → RSA → RCCA → LCCA → LSA
```

---

### 4.2 Target — 目标选择

每个 episode 开始时选择导航目标点。

```python
from synthendosim.target import (
    FixedTarget, CenterlineRandomTarget,
    BranchEndTarget, BranchIndexTarget, ManualRandomTarget,
)
```

| 类 | 说明 | 典型用途 |
|---|---|---|
| `FixedTarget(threshold_mm)` | 使用 AnatomySpec.target_point | 单解剖训练 |
| `CenterlineRandomTarget(branches=["lcca"])` | 在指定分支中线上随机采样 | 泛化训练 |
| `BranchEndTarget(branches=["lcca","lsa"])` | 随机选一个分支的远端开口 | 模拟临床目标 |
| `BranchIndexTarget(branch="lcca", idx=30)` | 固定分支+索引（可复现） | 标准化评估 |
| `ManualRandomTarget(positions=[...])` | 从用户列表随机选 | 自定义目标集 |

```python
# 评估：固定在 LCCA 中段
target = BranchIndexTarget(branch="left_common_carotid", idx=25, threshold_mm=8.0)

# 训练：随机目标（LCCA 或 LSA 末端）
target = BranchEndTarget(
    threshold_mm=10.0,
    branches=["left_common_carotid", "left_subclavian"],
)
```

---

### 4.3 Device — 介入器械

```python
from synthendosim.device import JShapedGuidewire, StraightGuidewire, SimmonsCatheter

wire   = JShapedGuidewire()       # 0.035" J 型头导丝（CAS 标准）
simmons = SimmonsCatheter()       # Simmons/Sidewinder 导管（复杂弓型）
```

| 类 | 说明 |
|---|---|
| `JShapedGuidewire` | J 型头 0.035" 导丝 |
| `StraightGuidewire` | 直头导丝 |
| `HydrophilicGuidewire` | 亲水涂层（低摩擦） |
| `SimmonsCatheter` | Simmons/Sidewinder 导管 |
| `PigtailCatheter` | 猪尾导管（主动脉造影） |
| `SheathCatheter` | 引导鞘（同轴系统外层） |

---

### 4.4 Start — 起始策略

控制每个 episode 开始时器械的初始状态。

```python
from synthendosim.start import (
    InsertionPointStart, RandomAdvanceStart,
    CurriculumStart, MultiAnatomyStart,
    VesselEndStart, MaxLengthStart,
)
```

| 类 | 说明 |
|---|---|
| `InsertionPointStart()` | 固定在入路点（默认） |
| `RandomAdvanceStart(advance_range_mm=(0, 50))` | 随机预推进 0-50mm |
| `CurriculumStart(mode="progressive", curriculum_episodes=10000)` | 课程学习：早期 episode 从近目标点开始 |
| `MultiAnatomyStart(anatomy_list=[...])` | 每 episode 随机切换解剖体 |
| `VesselEndStart()` | 检测到达血管末端时重置（防止死循环） |
| `MaxLengthStart(max_length_mm=400)` | 超出最大插入长度时重置 |

```python
# 课程学习：第 0 集从目标附近开始，逐渐延长距离
start = CurriculumStart(mode="progressive", curriculum_episodes=50_000)

# 训练：随机初始位置提升样本效率
start = RandomAdvanceStart(advance_range_mm=(0, 80))
```

---

### 4.5 InterimTarget — 中间路标

将长程导航分解为有意义的子目标，提供密集奖励。

```python
from synthendosim.interimtarget import (
    NoInterimTarget,
    FixedWaypointTarget,
    CenterlineWaypointTarget,
    ProximityWindowTarget,
    StageTarget,
    BranchingPointTarget,
)
```

| 类 | 说明 |
|---|---|
| `NoInterimTarget()` | 无路标，直接导向终点（baseline） |
| `CenterlineWaypointTarget(n_waypoints=5)` | 中线上均匀采 N 个路标 |
| `BranchingPointTarget(branches=["bcct","lcca"])` | 以血管分叉点为路标（解剖意义最强） |
| `ProximityWindowTarget(n_waypoints=8, lookahead_mm=30)` | 滑窗：只显示在前方 30mm 内的下一个目标 |
| `StageTarget(stages=[("arch", pos1), ("lcca", pos2)])` | 命名阶段（CAS 多步骤建模） |

```python
# CAS 标准路径：弓 → BCT → LCCA → 目标
from synthendosim.interimtarget import BranchingPointTarget
waypoints = BranchingPointTarget(
    threshold_mm=12.0,
    branches=["brachiocephalic_trunk", "left_common_carotid"],
)
```

---

### 4.6 Pathfinder — 路径距离

计算导丝尖端到目标的剩余路径长度，用于奖励和观测。

```python
from synthendosim.pathfinder import (
    EuclideanPathfinder,    # 直线距离 O(1)
    ManifoldPathfinder,     # 曲率加权测地距离（推荐）
    DijkstraPathfinder,     # 完整图 Dijkstra（最精确，适合评估）
)
```

| 类 | 速度 | 精度 | 适用场景 |
|---|---|---|---|
| `EuclideanPathfinder` | 最快 | 低（弯曲血管误差大） | debug |
| `ManifoldPathfinder(cl, curvature_weight=2.0)` | 快 | 高 | 训练 |
| `DijkstraPathfinder(cl)` | 慢 | 最高 | 评估 |

---

### 4.7 Terminal — 成功条件

Episode 成功时触发（`terminated=True`）。

```python
from synthendosim.terminal import (
    TargetReachedTerminal,   # 主力
    AllTargetsReachedTerminal,
    PathRatioTerminal,
    NeverTerminal,
)
# 组合
from synthendosim.terminal.base import AnyTerminal, AllTerminal

t = TargetReachedTerminal(threshold_mm=5.0) | PathRatioTerminal(0.05)
```

---

### 4.8 Truncation — 失败条件

Episode 失败/超时时触发（`truncated=True`）。

```python
from synthendosim.truncation import (
    MaxStepsTruncation,      # 超步数
    SimErrorTruncation,      # 物理错误
    VesselEndTruncation,     # 越界
    NoProgressTruncation,    # 无进展
)
from synthendosim.truncation.base import AnyTruncation

trunc = AnyTruncation([
    MaxStepsTruncation(max_steps=300),
    NoProgressTruncation(patience=50, min_delta_mm=0.5),
    SimErrorTruncation(),
])
```

---

### 4.9 Reward — 奖励函数

所有奖励项实现 `RewardComponent`，支持 `+` 运算符组合。

```python
from synthendosim.reward import (
    ManifoldDistanceDelta,       # 测地距离缩短量（主力项）
    EuclideanDistanceDelta,      # 欧氏距离缩短量
    TargetReached,               # 到达目标奖励
    StepPenalty,                 # 步数惩罚
    WallCollisionPenalty,        # 碰壁惩罚
    LastActionPenalty,           # 大动作惩罚（平滑控制）
    InsertionLengthDeltaReward,  # 插入进展奖励
    CoaxialClearanceReward,      # 同轴系统间距奖励
    FailurePenalty,              # 失败惩罚
)

# 标准 CAS 奖励配置
reward = (
    ManifoldDistanceDelta(weight=1.0)
    + TargetReached(bonus=10.0)
    + StepPenalty(penalty=-0.01)
    + LastActionPenalty(penalty_factor=-0.005)
)

# 同轴系统（导丝 + 导管）
reward = (
    ManifoldDistanceDelta(weight=1.0)
    + TargetReached(bonus=10.0)
    + CoaxialClearanceReward(lower_clearance_mm=5, upper_clearance_mm=40)
    + StepPenalty(-0.01)
)
```

---

### 4.10 Observation — 观测构建

#### 基础观测（ObsConfig 控制）

| 键名 | 维度 | 说明 |
|---|---|---|
| `tip_3d` | (3,) | 导丝尖端位置，归一化 [0,1]³ |
| `target_3d` | (3,) | 目标位置，归一化 [0,1]³ |
| `insertion_length` | (1,) | 插入长度，归一化 [0,1] |
| `rotation` | (1,) | 旋转角度 |
| `path_remaining` | (1,) | 剩余路径，归一化 |
| `wire_mask` | (H,W) 或 (2,H,W) | DSA 图像 |

#### 扩展观测（直接使用）

```python
from synthendosim.observation.tracking import (
    Tracking2D,              # N 点 2D 导丝追踪
    Tracking3D,              # N 点 3D 导丝追踪
    LastAction,              # 上一步动作
    InsertionLengths,        # 各器械插入长度
    InsertionLengthRelative, # 相对插入（同轴系统）
)

t3d = Tracking3D(n_points=8)
obs = t3d.build(state, bbox_min=env._bbox_min, bbox_max=env._bbox_max)
# obs.shape = (8, 3)
```

#### 观测包装器

```python
from synthendosim.observation.wrappers import (
    Memory,              # 时序堆叠（适合 RNN 策略）
    RelativeToLastState, # Δobs（速度信息）
    RelativeToFirstRow,  # 相对 episode 起始的位移
    Normalize,           # 自定义归一化
    SelectiveMemory,     # 只对特定 key 做时序堆叠
    MemoryResetMode,
)
import gymnasium as gym

# 堆叠最近 4 步的 tracking3d 观测
inner_space = gym.spaces.Box(0., 1., (8, 3))
mem = Memory(inner_space, n_steps=4, reset_mode=MemoryResetMode.FILL)
obs_reset = mem.reset(tracking3d_obs)   # shape (4, 8, 3)
obs_step  = mem.update(tracking3d_obs)  # shape (4, 8, 3)

# Delta obs
rel = RelativeToLastState(inner_space)
delta_obs = rel.update(tracking3d_obs)  # 当前 - 上一步
```

---

### 4.11 Imaging — 荧光成像

| 类 | 说明 | 速度 |
|---|---|---|
| `BiplaneDSA` | Beer-Lambert 双平面 DSA，Poisson 噪声 | 中 |
| `MonoplaneDSA` | 单平面 DSA | 中 |
| `PillowImager` | 2D 线段绘制（纯 NumPy） | 最快 |
| `NullImager` | 不渲染图像（向量观测策略） | 零开销 |

```python
from synthendosim.imaging.pillow import PillowImager

# 快速 debug 用
img_fn = PillowImager(image_size=(128, 128), line_radius_px=1)

# 换回 make_env 的 imaging 配置
env = ses.make_env(config_dict={
    "imaging": {"kind": "pillow", "image_size": [128, 128]},
    ...
})
```

---

### 4.12 Info — 指标收集

```python
from synthendosim.info.builders import InfoBuilder, build_info
from synthendosim.info.wrappers import AverageEpisodesLastStep, AverageSteps, InfoCompose

# 自定义指标
def fluoroscopy_dose(state, prev_state):
    return {"dose_mGy": state.step * 0.05}

builder = InfoBuilder(
    target_threshold_mm=5.0,
    extra_metrics=[fluoroscopy_dose],
)

# 滚动统计包装器
composer = InfoCompose([
    AverageEpisodesLastStep("target_distance_mm", n_episodes=100),
    AverageEpisodesLastStep("episode_stats/success", n_episodes=100),
    AverageSteps("reward"),
])
# 在 env.step() 之后：
info = composer.update(info, terminated or truncated)
# info 新增: "avg100/target_distance_mm", "avg100/episode_stats/success", "mean_step/reward"
```

每一步 `info` 包含：

| 键 | 说明 |
|---|---|
| `step` | 当前步数 |
| `episode` | Episode 编号 |
| `tip_position` | 尖端坐标 (3,) |
| `target_distance_mm` | 到目标距离 mm |
| `path_remaining_mm` | 剩余路径 mm |
| `target_reached` | 是否达到目标 |
| `interim_target_position` | 当前路标坐标 |
| `interim_target_distance_mm` | 到路标距离 |
| `interim_advanced` | 本步是否推进了路标 |
| `waypoints_remaining` | 剩余路标数 |
| `episode_stats/*` | Episode 结束时附加（成功率、总reward等） |

---

### 4.13 Visualisation — 可视化

```python
from synthendosim.visualisation import (
    NullRenderer,        # 不渲染（训练）
    MatplotlibRenderer,  # MPL 图（Jupyter / 脚本）
    OpenCVRenderer,      # cv2 实时窗口
    VideoRenderer,       # 写入 MP4
)

# 训练时录制每个 episode
env = ses.make_env(..., renderer=VideoRenderer("runs/ep.mp4", fps=10, per_episode=True))

# 实时监控
env = ses.make_env(..., renderer=OpenCVRenderer(wait_ms=1))
```

---

### 4.14 Physics Backend — 物理后端

| 后端 | 说明 |
|---|---|
| `mock` | 纯 NumPy mock，无物理，用于测试和调试 |
| `sofa` | SOFA + BeamAdapter FEM，真实导丝形变 |

SOFA 后端需要安装 SOFA v23.06 + BeamAdapter 插件：

```bash
export SOFA_ROOT=/opt/sofa/SOFA_v23.06.00_Linux
export PYTHONPATH=$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH
conda activate sofa
```

---

## 5. 工具模块

### 5.1 StateRecorder — Episode 录制与回放

```python
from synthendosim.util import InterventionStateRecorder, MultiEpisodeRecorder

# 单 episode 录制
recorder = InterventionStateRecorder()
obs, info = env.reset(seed=42)
recorder.on_reset(obs, info, episode=1)
for _ in range(max_steps):
    action = policy(obs)
    obs, reward, term, trunc, info = env.step(action)
    recorder.on_step(action, obs, reward, term, trunc, info)
    if term or trunc:
        break
recorder.save("episode_001.pkl")

# 回放分析
ep = InterventionStateRecorder.load("episode_001.pkl")
print(f"Steps: {ep.n_steps}, Reward: {ep.total_reward:.2f}, Success: {ep.success}")
obs_list, actions, rewards = ep.to_sar()   # 转换为 (obs, action, reward) 供 BC 训练

# 批量收集演示数据
dataset = MultiEpisodeRecorder(save_dir="demos/", prefix="expert")
for ep_i in range(1000):
    obs, info = env.reset()
    dataset.on_reset(obs, info)
    for _ in range(max_steps):
        action = expert_policy(obs)
        obs, reward, term, trunc, info = env.step(action)
        dataset.on_step(action, obs, reward, term, trunc, info)
        if term or trunc:
            break

# 加载为 BC 数据集
all_obs, all_actions, all_rewards = dataset.to_bc_dataset()
```

---

## 6. 插件扩展指南

### 自定义 VesselTree

```python
from synthendosim.vesseltree.base import VesselTree, Insertion
from synthendosim.vesseltree import register
import numpy as np

class CoronaryTree(VesselTree):
    def reset(self, episode=0, seed=None):
        # 加载冠脉中线坐标
        self.branches = (...)
        self.branching_points = [...]
        self.centerline_coordinates = ...
        self.insertion = Insertion(
            position=np.array([-10, -150, 0]),
            direction=np.array([0, 1, 0]),
        )
        self.bbox_low  = ...
        self.bbox_high = ...
        self.mesh_path = "data/coronary.obj"

register("coronary", CoronaryTree)
tree = make_vessel_tree("coronary")
```

### 自定义 Terminal

```python
from synthendosim.terminal.base import Terminal
from synthendosim.terminal import register as register_terminal

class FluoroscopyDoseTerminal(Terminal):
    """成功条件：到达目标 AND 使用的透视剂量 < 阈值。"""
    def __init__(self, threshold_mm=5.0, max_dose_mGy=100.0):
        self.threshold_mm = threshold_mm
        self.max_dose_mGy = max_dose_mGy
        self._dose = 0.0

    def reset(self, anatomy=None, episode=0):
        self._dose = 0.0

    def __call__(self, state, prev_state=None):
        self._dose += 0.05   # 每步 0.05 mGy
        tip = state.devices[0].tip_position
        dist = float(np.linalg.norm(tip - state.target_position))
        return dist <= self.threshold_mm and self._dose <= self.max_dose_mGy

register_terminal("dose_limited", FluoroscopyDoseTerminal)
```

### 自定义 Reward

```python
from synthendosim.reward.components import RewardComponent
from synthendosim.reward import register as register_reward

class WireOscillationPenalty(RewardComponent):
    """惩罚导丝前后反复的振荡行为。"""
    def __init__(self, penalty=-0.02, window=5):
        self.penalty = penalty
        self._history = []

    def __call__(self, state, prev_state=None):
        if not state.devices:
            return 0.0
        length = state.devices[0].inserted_length
        self._history.append(length)
        if len(self._history) > self._window:
            self._history.pop(0)
        if len(self._history) < 3:
            return 0.0
        oscillations = sum(
            1 for a, b, c in zip(self._history, self._history[1:], self._history[2:])
            if (b - a) * (c - b) < 0  # 符号改变 = 反向
        )
        return self.penalty * oscillations

register_reward("oscillation", WireOscillationPenalty)
```

---

## 7. 配置系统

YAML 配置文件等价于 `config_dict`。完整字段：

```yaml
# synthendosim_config.yaml
physics:
  backend: sofa          # sofa | mock
  dt_simulation: 0.01    # s
  friction: 0.1

anatomy:
  mesh_path: data/aorta.obj
  insertion_point: [-23.0, -180.0, -15.0]
  insertion_direction: [0.0, 1.0, 0.0]
  target_point: [-31.7, 70.3, 11.6]
  centerline_path: null
  scale: [1.0, 1.0, 1.0]
  rotation_yzx_deg: [0.0, 0.0, 0.0]

device:
  kind: j_guidewire
  total_length: 450.0   # mm
  tip_length: 40.0
  tip_angle: 0.21
  diameter: 0.89

imaging:
  kind: biplane_dsa      # biplane_dsa | monoplane | pillow | none
  image_size: [128, 128]
  lao_deg: 30.0
  lat_deg: 120.0
  n0_photons: 15000
  mu_blood: 0.048
  mu_tissue: 0.020

reward:
  manifold_distance_weight: 1.0
  target_reached_bonus: 10.0
  step_penalty: -0.01
  wall_collision_penalty: -1.0
  target_threshold_mm: 5.0
  use_manifold: true
  waypoint_bonus: 1.0

obs:
  use_tip_3d: true
  use_target_3d: true
  use_path_remaining: true
  use_insertion_length: true
  use_rotation: true
  use_wire_mask: false   # true 时在 obs 中加入 DSA 图像

episode:
  max_steps: 300
```

```python
env = ses.make_env(config_path="synthendosim_config.yaml")
# 或局部覆盖
env = ses.make_env(config_path="synthendosim_config.yaml",
                   terminal=TargetReachedTerminal(5.0))
```

---

## 8. 训练示例

### SAC 训练（stable-baselines3）

```python
from stable_baselines3 import SAC
import synthendosim as ses
from synthendosim.interimtarget import CenterlineWaypointTarget
from synthendosim.pathfinder import ManifoldPathfinder
from synthendosim.vesseltree import AorticArch

tree = AorticArch("I", seed=42); tree.reset()

env = ses.make_env(
    config_dict={
        "physics": {"backend": "sofa"},
        "anatomy": {"mesh_path": "data/aorta_K1.obj",
                    "insertion_point": [-23, -180, -15],
                    "target_point": [-31.7, 70.3, 11.6]},
        "episode": {"max_steps": 300},
    },
    interimtarget=CenterlineWaypointTarget(n_waypoints=5),
    pathfinder=ManifoldPathfinder(tree.centerline_coordinates),
)

model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log="logs/")
model.learn(total_timesteps=200_000)
```

### 多解剖泛化训练

```python
from synthendosim.vesseltree import AorticArchRandom
from synthendosim.target import BranchEndTarget

# 环境工厂（每个 episode 随机解剖）
def make_generalization_env():
    return ses.make_env(
        config_dict={"physics": {"backend": "sofa"},
                     "episode": {"max_steps": 300}},
        target=BranchEndTarget(
            threshold_mm=8.0,
            branches=["left_common_carotid", "left_subclavian"],
        ),
        interimtarget=BranchingPointTarget(threshold_mm=12.0),
    )

vec_env = ses.make_vec_env(n_envs=8, env_fn=make_generalization_env)
model = SAC("MultiInputPolicy", vec_env, verbose=1)
model.learn(total_timesteps=500_000)
```

---

## 9. 评估协议

```python
from synthendosim.util import InterventionStateRecorder
from synthendosim.target import BranchIndexTarget
from synthendosim.pathfinder import DijkstraPathfinder
import numpy as np

# 标准评估：固定目标 + Dijkstra 距离
eval_env = ses.make_env(
    config_dict={...},
    target=BranchIndexTarget(branch="left_common_carotid", idx=25, threshold_mm=5.0),
    pathfinder=DijkstraPathfinder(tree.centerline_coordinates),
)

results = []
recorder = InterventionStateRecorder()

for ep in range(100):
    obs, info = eval_env.reset(seed=ep)
    recorder.on_reset(obs, info, episode=ep)
    ep_reward = 0.0
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = eval_env.step(action)
        recorder.on_step(action, obs, reward, term, trunc, info)
        ep_reward += reward
        if term or trunc:
            break
    results.append({
        "success": info.get("target_reached", False),
        "steps": info["step"],
        "final_dist_mm": info["target_distance_mm"],
    })
    recorder.save(f"eval_eps/ep_{ep:03d}.pkl")

success_rate = np.mean([r["success"] for r in results])
avg_steps    = np.mean([r["steps"]   for r in results])
print(f"Success Rate: {success_rate:.1%}  |  Avg Steps: {avg_steps:.0f}")
```

---

## 10. FAQ

**Q: 没有 SOFA 可以运行吗？**

A: 可以。使用 `backend="mock"` 可以完整运行所有模块（VesselTree、Target、Reward、Obs 等），只是导丝的物理形变是随机 mock 的。用于调试算法逻辑时完全足够。

**Q: 如何添加真实 CT 患者数据？**

A: 用 vmtk 或 Slicer 从 CT 提取中线（.csv）和网格（.obj），然后：
```python
import numpy as np
centerline = np.loadtxt("centerline.csv")
from synthendosim.vesseltree.util.branch import BranchWithRadii
branch = BranchWithRadii("aorta", centerline[:, :3], centerline[:, 3])
tree = VesselTreeFromMesh("mesh.obj", insertion_position=..., branches=[branch])
```

**Q: 如何训练 RNN 策略？**

A: 使用 `Tracking3D` + `Memory` wrapper 堆叠时序观测：
```python
from synthendosim.observation.tracking import Tracking3D
from synthendosim.observation.wrappers import Memory, MemoryResetMode
import gymnasium as gym

t3d = Tracking3D(n_points=8)
mem = Memory(t3d.space, n_steps=4, reset_mode=MemoryResetMode.FILL)
# obs shape: (4, 8, 3) → 输入 LSTM
```

**Q: 如何做行为克隆？**

A: 用 `MultiEpisodeRecorder` 收集专家演示（来自键盘控制或运动规划），然后用 `to_bc_dataset()` 生成训练数据。

**Q: 奖励太稀疏怎么办？**

A: 加中间路标：
```python
interimtarget=CenterlineWaypointTarget(n_waypoints=8, threshold_mm=10.0)
# 每到达一个路标 +waypoint_bonus（config.reward.waypoint_bonus）
```

**Q: 如何录制训练视频？**

A: 注入 `VideoRenderer`：
```python
from synthendosim.visualisation import VideoRenderer
env.set_plugin(renderer=VideoRenderer("training.mp4", fps=15, per_episode=True))
```
