#!/bin/bash
# stEVE 仿真环境安装脚本
# 使用前请确保已安装 SOFA v23.06 + SofaPython3 插件
# 参考: https://www.sofa-framework.org/download/

set -e

SOFA_ROOT="${SOFA_ROOT:-/opt/sofa/SOFA_v23.06.00_Linux}"

if [ ! -d "$SOFA_ROOT" ]; then
  echo "[错误] 未找到 SOFA 安装目录: $SOFA_ROOT"
  echo "  请设置环境变量 SOFA_ROOT 指向你的 SOFA 安装路径，例如："
  echo "  export SOFA_ROOT=/path/to/SOFA_v23.06.00_Linux"
  exit 1
fi

echo "=== 创建 conda 环境 sofa ==="
conda create -n sofa python=3.10 -y

echo "=== 激活环境 ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sofa

echo "=== 配置 SOFA 路径 ==="
export PYTHONPATH=$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$SOFA_ROOT/lib:$SOFA_ROOT/plugins/SofaPython3/lib:$LD_LIBRARY_PATH

echo "=== 安装 Python 依赖 ==="
pip install stable-baselines3 gymnasium tensorboard \
            numpy pillow scipy scikit-image \
            pyvista meshio PyOpenGL pygame \
            matplotlib opencv-python pyyaml \
            trimesh

echo "=== 安装 eve 仿真包 ==="
pip install -e .

echo ""
echo "✓ 安装完成！运行以下命令激活环境："
echo "  conda activate sofa"
echo "  export SOFA_ROOT=$SOFA_ROOT"
echo "  export PYTHONPATH=\$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:\$PYTHONPATH"
echo "  export LD_LIBRARY_PATH=\$SOFA_ROOT/lib:\$SOFA_ROOT/plugins/SofaPython3/lib:\$LD_LIBRARY_PATH"
