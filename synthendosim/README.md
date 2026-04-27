# SynthEndoSim

> **Synthetic Endovascular Simulation Engine** — 从 CT 到物理机器人的血管介入导丝导航仿真框架

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![SOFA](https://img.shields.io/badge/SOFA-v23.06-green)](https://www.sofa-framework.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-orange)](https://gymnasium.farama.org/)

---

## 核心优势（对比 stEVE）

| 特性 | stEVE | **SynthEndoSim** |
|------|-------|-----------------|
| API 入口 | 10+ 类手动组合 | `ses.make()` / `ses.from_mesh()` 单行 |
| 配置方式 | Python 硬编码 | YAML + dict，完全可组合 |
| 荧光渲染 | 单平面（需 SOFA GL） | **双平面 Beer-Lambert DSA（纯 NumPy）** |
| 奖励函数 | 固定几种 | 可组合，支持 `+` 运算符 |
| 血管加载 | 写死解剖结构 | 任意 OBJ/STL，直接读 pipeline 输出 |
| 并行仿真 | 笨拙 multiprocessing | `make_vec_env(n)` 真正多进程 |
| 测试支持 | 需要 SOFA | **mock backend，无需安装 SOFA** |
| 物理引擎 | SOFA + BeamAdapter | SOFA + BeamAdapter（可扩展） |

---

## 安装

```bash
# 1. 安装 SOFA v23.06（参考 ../simulation/install_env.sh）
# 2. 安装 SynthEndoSim
pip install -e .

# 3. 可选：RL 支持
pip install -e ".[rl]"
```

---

## 快速开始

### Mock 模式（无需 SOFA）

```python
import synthendosim as ses

env = ses.from_mesh(
    mesh="",                              # 空字符串 = mock 模式
    insertion_point=(-23, -180, -15),
    target_point=(-31.7, 70.3, 11.6),
    backend="mock",
)

obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

### 真实 SOFA 仿真

```python
import synthendosim as ses

env = ses.from_mesh(
    mesh="data/aorta_K1.obj",
    insertion_point=(-23.0, -180.0, -15.0),
    target_point=(-31.7, 70.3, 11.6),
    backend="sofa",
    max_steps=300,
)
```

### 预设环境

```python
env = ses.make(
    "SynthEndoSim-AorticArch-CAS-v1",
    anatomy_mesh="data/aorta_K1.obj",
)
```

### YAML 配置

```python
env = ses.make_env(config_path="my_config.yaml")
```

```yaml
# my_config.yaml
name: MyExp-v1
anatomy:
  mesh_path: data/aorta_K1.obj
  insertion_point: [-23.0, -180.0, -15.0]
  target_point: [-31.7, 70.3, 11.6]
reward:
  manifold_distance_weight: 2.0
  target_reached_bonus: 10.0
episode:
  max_steps: 300
```

### 自定义奖励（`+` 运算符）

```python
from synthendosim.reward import ManifoldDistanceDelta, TargetReached, StepPenalty

reward = (
    2.0 * ManifoldDistanceDelta()
    + TargetReached(bonus=10.0, threshold_mm=10.0)
    + StepPenalty(penalty=-0.01)
)
```

### 多解剖并行训练

```python
vec_env = ses.make_vec_env(
    n_envs=4,
    config_dict={"physics": {"backend": "sofa"}, ...},
    mesh_list=["aorta_K1.obj", "aorta_K2.obj", "aorta_K3.obj", "aorta_K4.obj"],
)
obs, info = vec_env.reset(seed=0)
```

---

## 架构

```
synthendosim/
├── __init__.py          # make() / from_mesh() / make_env() / make_vec_env()
├── core/
│   ├── env.py           # SynthEndoEnv — 主 Gymnasium 环境
│   ├── types.py         # SimState, Observation, DeviceState 等类型
│   └── registry.py      # 预设环境注册表
├── physics/
│   ├── backend.py       # PhysicsBackend 抽象接口
│   ├── sofa_backend.py  # SOFA + BeamAdapter 实现
│   └── mock_backend.py  # 纯 NumPy mock（测试用）
├── anatomy/
│   ├── loader.py        # OBJ/STL 直接加载，自动 bbox 检测
│   └── centerline.py    # ManifoldPathfinder（曲率加权测地距离）
├── imaging/
│   └── dsa.py           # BiplaneDSA / MonoplaneDSA（Beer-Lambert）
├── reward/
│   └── components.py    # ManifoldDistanceDelta / TargetReached / 可组合
├── observation/
│   └── builders.py      # ObservationBuilder（归一化 + 空间定义）
├── vec_env.py           # SynthEndoVecEnv（多进程并行）
├── config/
│   ├── schema.py        # EnvConfig dataclass + YAML 加载
│   └── defaults/        # base.yaml / aortic_arch_cas.yaml
└── examples/
    ├── quickstart_mock.py      # 无 SOFA 快速验证
    ├── train_sac.py            # SAC 训练
    └── train_ablation.py       # 多解剖消融实验
```

---

## 动作 / 观测空间

**动作空间** `Box(-1, 1, (2,))`
- `[0]` 平移速度（归一化，×10 mm/s）
- `[1]` 旋转速度（归一化，×π rad/s）

**观测空间** `Dict`（默认）
| 键 | Shape | 说明 |
|----|-------|------|
| `tip_3d` | (3,) | 导丝尖端位置（归一化至 [0,1]） |
| `target_3d` | (3,) | 目标位置（归一化） |
| `insertion_length` | (1,) | 已插入长度（归一化） |
| `rotation` | (1,) | 累计旋转（归一化） |
| `path_remaining` | (1,) | 剩余路径距离（归一化） |
| `wire_mask` | (2,H,W) | 双平面 DSA（`use_wire_mask=True` 时） |

---

## 物理后端说明

物理核心（SOFA + BeamAdapter FEM）与上层完全解耦，通过 `PhysicsBackend` 接口隔离。
未来可替换为 JAX 可微仿真，**无需修改任何上层代码**。

```python
# 切换后端只需一行
env = ses.from_mesh(mesh="...", backend="mock")   # 测试
env = ses.from_mesh(mesh="...", backend="sofa")   # 生产
```
