#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=/public/home/yingxy2024/.conda/envs/slime
SLIME_DIR=/public/home/yingxy2024/slime
MEGATRON_DIR=/public/home/yingxy2024/slime-deps/Megatron-LM
HF_MODEL=/storage/data/yingxy2024/Models/Qwen3-4B
TORCH_DIST_MODEL=/storage/data/yingxy2024/Models/Qwen3-4B_torch_dist
RUNTIME_CUDA=/tmp/slime-cuda-runtime

if [[ -f "${TORCH_DIST_MODEL}/latest_checkpointed_iteration.txt" ]]; then
    echo "Converted checkpoint already exists: ${TORCH_DIST_MODEL}"
    exit 0
fi

mkdir -p "${RUNTIME_CUDA}/bin" "${RUNTIME_CUDA}/lib64" "${TORCH_DIST_MODEL}"
ln -sfn "${ENV_DIR}/bin/nvcc" "${RUNTIME_CUDA}/bin/nvcc"
ln -sfn "${ENV_DIR}/targets/x86_64-linux/include" "${RUNTIME_CUDA}/include"
ln -sfn "${ENV_DIR}/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12" \
    "${RUNTIME_CUDA}/lib64/libnvrtc.so.12"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}"
export CUDA_HOME="${RUNTIME_CUDA}"
export CUDACXX="${ENV_DIR}/bin/nvcc"
export CUDNN_HOME="${ENV_DIR}/lib/python3.12/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="${ENV_DIR}/lib/python3.12/site-packages/torch/lib:${ENV_DIR}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${ENV_DIR}/lib/python3.12/site-packages/nvidia/cudnn/lib:${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

cd "${SLIME_DIR}"
source scripts/models/qwen3-4B.sh

python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_MODEL}" \
    --save "${TORCH_DIST_MODEL}"

