# 血管介入手术机器人仿真平台

> **SynthVasc** — 端到端血管介入导丝导航平台：从公开 CT 数据到物理机器人部署的完整闭环

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![SOFA](https://img.shields.io/badge/SOFA-v23.06-green)](https://www.sofa-framework.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 项目简介

本平台实现了从 CT 扫描到自主导丝导航的完整流水线，核心贡献包括：

1. **CT → 合成 DSA 自动生成**：全自动将任意对比增强 CT 转化为带完整 3D 标注的双平面合成荧光序列（≈200帧/例），无需真实患者 DSA 数据。
2. **stEVE 仿真环境**：基于 SOFA BeamAdapter 有限元引擎的物理精确导丝-血管交互仿真，支持 SAC/PPO/TD3 等强化学习算法训练数据采集。
3. **虚实域迁移**：带导丝保留损失 $\mathcal{L}_\text{wire}$ 的 CUT 无配对翻译，将合成 DSA 风格迁移至真实 guide3d 荧光图像分布。
4. **扩散策略部署**：在合成多解剖轨迹上离线训练 Diffusion Policy（ResNet-18 + DDIM-10），实现 <15ms/步 的实时推理，并支持零样本迁移至物理机器人平台。

---

## 整体架构

```
CT (.nrrd / .nii.gz)
  │
  ├─ pipeline/preprocess.py     重采样至 1.5mm 等向分辨率
  ├─ pipeline/segment.py        TotalSegmentator 血管分割（6类）
  ├─ pipeline/mesh.py           Marching Cubes → OBJ 网格
  ├─ pipeline/nav.py            提取插入点（股动脉）+ LCCA 目标
  ├─ pipeline/path.py           体素化 Dijkstra 路径规划
  │
  ├─ simulation/                ★ stEVE 仿真后端（本模块）
  │   ├─ eve/                   核心仿真库（SOFA BeamAdapter 封装）
  │   └─ examples/              训练 & 演示脚本
  │       ├─ train_demo.py                  快速验证训练（10k步）
  │       ├─ train_ablation_generalization.py  泛化消融实验（single/multi5/multi10）
  │       ├─ demo_exp7.py                   可视化演示
  │       └─ plot_ablation.py               生成论文图表
  │
  ├─ pipeline/synth_dsa_v2.py   Beer-Lambert 双平面 DSA 渲染
  ├─ pipeline/infer.py          推理接口（调用训练好的策略）
  │
  ├─ api/main.py                FastAPI 后端
  └─ frontend/                  React + Three.js 3D 可视化界面
```

---

## 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10 | conda 管理 |
| SOFA Framework | v23.06.00 | 有限元仿真引擎 |
| SofaPython3 插件 | 随 SOFA 发布 | SOFA Python 绑定 |
| CUDA（可选） | 11.8+ | GPU 训练加速（CPU 也可运行） |
| Node.js | 18+ | 前端构建 |

### 硬件推荐
- CPU：8 核以上（SOFA 仿真为单线程，多进程并行训练）
- 内存：32 GB+
- 磁盘：50 GB+（SOFA + conda 环境 + 数据）

---

## 安装步骤

### 1. 安装 SOFA v23.06

从 [SOFA 官网](https://github.com/sofa-framework/sofa/releases/tag/v23.06.00) 下载预编译包并解压：

```bash
# 示例：解压至 /opt/sofa/
tar -xzf SOFA_v23.06.00_Linux.tar.gz -C /opt/sofa/
export SOFA_ROOT=/opt/sofa/SOFA_v23.06.00_Linux
```

### 2. 克隆本仓库

```bash
git clone https://github.com/miracle-techlink/vascular-intervention-platform.git
cd vascular-intervention-platform
```

### 3. 创建仿真 conda 环境

```bash
conda create -n sofa python=3.10 -y
conda activate sofa

# 配置 SOFA 路径（每次激活环境后需执行，建议写入 ~/.bashrc）
export SOFA_ROOT=/opt/sofa/SOFA_v23.06.00_Linux
export PYTHONPATH=$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$SOFA_ROOT/lib:$SOFA_ROOT/plugins/SofaPython3/lib:$LD_LIBRARY_PATH

# 安装 Python 依赖
pip install stable-baselines3 gymnasium tensorboard \
            numpy pillow scipy scikit-image \
            pyvista meshio PyOpenGL pygame \
            matplotlib opencv-python pyyaml trimesh

# 安装 eve 仿真包（开发模式）
cd simulation
pip install -e .
cd ..
```

### 4. 创建 CT 处理环境（独立）

```bash
conda create -n dsa3d python=3.10 -y
conda activate dsa3d
pip install SimpleITK totalsegmentator trimesh scipy numpy
```

### 5. 安装前端

```bash
cd frontend
npm install    # 或 pnpm install
cd ..
```

---

## 快速开始

### 方式一：完整 Web 平台

```bash
# 终端1：启动后端
conda activate dsa3d
cd api && python main.py

# 终端2：启动前端
cd frontend && npm run dev
```

浏览器访问 `http://localhost:5173`，上传 CT 文件，点击运行即可。

---

### 方式二：仅运行仿真（无需 Web）

#### 最小验证（确认 SOFA 安装正常）

```bash
conda activate sofa
# 配置 SOFA 环境变量（见上）

cd simulation
python examples/single_jwire.py
```

若无报错并看到仿真输出，说明环境配置成功。

#### 快速训练演示（10k 步，约 15 分钟）

```bash
conda activate sofa
cd simulation
python examples/train_demo.py
# 模型保存至 models_demo/
```

#### 泛化消融实验（论文 E5）

对比单血管 vs 多血管训练的跨患者泛化能力：

```bash
conda activate sofa
cd simulation

# 三组并行训练（各 200k 步）
python examples/train_ablation_generalization.py --mode single  > logs/ablation_single.log 2>&1 &
python examples/train_ablation_generalization.py --mode multi5  > logs/ablation_multi5.log 2>&1 &
python examples/train_ablation_generalization.py --mode multi10 > logs/ablation_multi10.log 2>&1 &

# 等待完成后生成论文图表
python examples/plot_ablation.py
# 输出: models_ablation/figures/{fig_convergence,fig_final_sr,fig_per_seed}.pdf
```

#### 可视化演示

```bash
conda activate sofa
cd simulation
python examples/demo_exp7.py
```

---

## 目录结构说明

```
vascular-intervention-platform/
├── api/                        FastAPI 后端服务
│   ├── main.py                 API 路由（上传CT、触发流水线、轮询状态）
│   └── infer.py                策略推理接口
│
├── frontend/                   前端（React + Three.js）
│   ├── src/App.jsx             主界面
│   └── src/Viewer3D.jsx        3D 血管可视化
│
├── pipeline/                   CT 处理流水线
│   ├── preprocess.py           CT 重采样
│   ├── segment.py              TotalSegmentator 分割
│   ├── mesh.py                 网格生成
│   ├── nav.py                  导航点提取
│   ├── path.py                 Dijkstra 路径规划
│   ├── sofa_runner.py          SOFA 仿真调用器
│   └── synth_dsa_v2.py         Beer-Lambert DSA 渲染
│
├── simulation/                 ★ stEVE 仿真核心
│   ├── eve/                    仿真库
│   │   ├── env.py              Gymnasium 主环境
│   │   ├── intervention/       干预场景（设备、荧光、仿真、目标、血管树）
│   │   ├── observation/        观测空间定义
│   │   ├── reward/             奖励函数
│   │   ├── terminal/           终止条件
│   │   ├── truncation/         截断条件
│   │   └── visualisation/      可视化模块
│   ├── examples/               示例脚本
│   ├── setup.py                包安装配置
│   └── install_env.sh          一键环境安装脚本
│
├── data/                       示例数据（小文件）
├── .gitignore
├── build.sh                    完整构建脚本
├── start.sh                    一键启动脚本
└── README.md                   本文档
```

---

## 核心仿真接口

`eve` 库遵循 [Gymnasium](https://gymnasium.farama.org/) 接口，可与 stable-baselines3 直接配合：

```python
import eve
from stable_baselines3 import SAC

# 构建主动脉弓仿真环境
vessel_tree = eve.intervention.vesseltree.AorticArch(seed=42)
device      = eve.intervention.device.JShaped()
simulation  = eve.intervention.simulation.SofaBeamAdapter(friction=0.01)
fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
    simulation=simulation, vessel_tree=vessel_tree,
    image_frequency=7.5, image_rot_zx=[0, 0])
target = eve.intervention.target.CenterlineRandom(
    vessel_tree=vessel_tree, fluoroscopy=fluoroscopy,
    threshold=5.0, branches=["lcca"])
intervention = eve.intervention.MonoPlaneStatic(
    vessel_tree=vessel_tree, devices=[device],
    simulation=simulation, fluoroscopy=fluoroscopy,
    target=target, normalize_action=True)

obs    = eve.observation.TipState(intervention=intervention)
reward = eve.reward.Combination([
    eve.reward.TargetReached(intervention=intervention, factor=100.0),
    eve.reward.TipToTargetDistDelta(factor=1.0, intervention=intervention),
    eve.reward.FailurePenalty(intervention=intervention, factor=-50.0),
])
terminal   = eve.terminal.TargetReached(intervention=intervention)
truncation = eve.truncation.Combination([
    eve.truncation.MaxSteps(300),
    eve.truncation.VesselEnd(intervention=intervention),
])

env   = eve.Env(intervention=intervention, observation=obs,
                reward=reward, terminal=terminal, truncation=truncation)
model = SAC("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=50_000)
```

---

## 已验证性能

| 指标 | 数值 |
|------|------|
| KiTS 主动脉弓导航成功率（SAC, multi-10） | ~60% |
| 推理延迟（Diffusion Policy, DDIM-10） | <15 ms/步 |
| 合成 DSA 帧数（11例 KiTS AVT） | ~2,200 帧 |
| SOFA 仿真速度 | ~1× 实时 |
| 单次 CT 处理耗时（CPU） | 10–15 分钟 |

---

## 论文引用

如使用本平台，请引用：

```bibtex
@article{synthvasc2026,
  title   = {SynthVasc: Synthetic DSA Generation, Sim-to-Real Adaptation,
             and Diffusion Policy Learning for Endovascular Guidewire Navigation},
  author  = {Vascular Intervention Robot Group, Tongji University},
  journal = {RA-L + ICRA},
  year    = {2026}
}
```

同时请引用 stEVE 仿真环境基础：

```bibtex
@inproceedings{karstensen2025steve,
  title     = {stEVE: Open-source simulation for endovascular interventions},
  author    = {Karstensen, L. and others},
  booktitle = {Comput. Biol. Med.},
  year      = {2025}
}
```

---

## 常见问题

**Q: 运行时提示 `No module named 'Sofa'`**

请确认已正确配置 SOFA 环境变量：
```bash
export SOFA_ROOT=/path/to/SOFA_v23.06.00_Linux
export PYTHONPATH=$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$SOFA_ROOT/lib:$SOFA_ROOT/plugins/SofaPython3/lib:$LD_LIBRARY_PATH
```

**Q: CUDA 报错 `no kernel image is available for execution on the device`（RTX 40/50 系列）**

请在创建模型时指定 `device="cpu"`：
```python
model = SAC("MlpPolicy", env, device="cpu")
```
SOFA 仿真本身为 CPU 运算，策略网络使用 CPU 即可满足需求。

**Q: 训练速度很慢**

stEVE 仿真约为 1× 实时速度（每步约 0.1–0.5秒），这是 SOFA 有限元仿真的固有限制。建议：
- 使用 `--mode single` 先验证流程（约 1–2小时/200k步）
- 多进程并行运行不同 mode 的实验
- 使用 `learning_starts=2000` 减少早期无效探索

**Q: TotalSegmentator 显存不足**

```bash
# 使用 fast 模式（精度略低，但仅需 4GB 显存）
TotalSegmentator -i input.nii.gz -o output/ --fast
```

---

## 致谢

- [stEVE](https://github.com/lkarstensen/stEVE)：开源血管介入仿真环境基础
- [SOFA Framework](https://www.sofa-framework.org/)：有限元物理仿真引擎
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)：CT 血管分割
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)：强化学习算法库
- [guide3d](https://github.com/airvlab/guide3d)：真实双平面 DSA 数据集
- Sim4EndoR 团队（姚天亮、齐鹏等，同济大学）：物理机器人平台支持

---

## 开源协议

MIT License © 2026 血管介入手术机器人课题组，同济大学
