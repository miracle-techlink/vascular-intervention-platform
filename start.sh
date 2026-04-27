#!/bin/bash
# Start: single uvicorn process serves both API and frontend
# Usage: ./start.sh [port]
PORT=${1:-8000}
cd "$(dirname "$0")"

if [ ! -d "frontend/dist" ]; then
  echo "frontend not built, running build.sh first..."
  bash build.sh
fi

echo "➜ http://localhost:$PORT"
CUDA_VISIBLE_DEVICES="" /home/liuyue/miniforge3/envs/dsa3d/bin/uvicorn api.main:app \
  --host 0.0.0.0 --port "$PORT" --log-level info
