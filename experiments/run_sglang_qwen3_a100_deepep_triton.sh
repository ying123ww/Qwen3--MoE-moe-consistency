#!/usr/bin/env bash
set -euo pipefail

project_root=/public/home/yingxy2024/moe-consistency
sglang_dir="$project_root/experiments/sglang-main-sm80"
model_path="${MODEL_PATH:-$project_root/experiments/Qwen3-30B-A3B-2layer}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONPATH="$sglang_dir/python${PYTHONPATH:+:$PYTHONPATH}"

exec python -m sglang.launch_server \
  --model-path "$model_path" \
  --tp-size 4 \
  --ep-size 4 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --moe-runner-backend triton \
  --deepep-dispatcher-output-dtype bf16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-custom-all-reduce \
  --mem-fraction-static 0.20 \
  --host 127.0.0.1 \
  --port "${PORT:-31004}" \
  --trust-remote-code
