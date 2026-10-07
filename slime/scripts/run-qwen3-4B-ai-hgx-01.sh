#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=/public/home/yingxy2024/.conda/envs/slime
SLIME_DIR=/public/home/yingxy2024/slime
MEGATRON_DIR=/public/home/yingxy2024/slime-deps/Megatron-LM
HF_MODEL=/storage/data/yingxy2024/Models/Qwen3-4B
TORCH_DIST_MODEL=/storage/data/yingxy2024/Models/Qwen3-4B_torch_dist
TRAIN_DATA=/storage/data/yingxy2024/data/dapo-math-17k/dapo-math-17k.jsonl
EVAL_DATA=/storage/data/yingxy2024/data/aime-2024/aime-2024.jsonl
OUTPUT_DIR=/storage/data/yingxy2024/trained_models/Qwen3-4B-slime-grpo
RUNTIME_CUDA=/tmp/slime-cuda-runtime

for required in \
    "${HF_MODEL}/config.json" \
    "${TORCH_DIST_MODEL}/latest_checkpointed_iteration.txt" \
    "${TRAIN_DATA}" \
    "${EVAL_DATA}"; do
    [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

mkdir -p "${RUNTIME_CUDA}/bin" "${RUNTIME_CUDA}/lib64/stubs" "${OUTPUT_DIR}"
ln -sfn "${ENV_DIR}/bin/nvcc" "${RUNTIME_CUDA}/bin/nvcc"
ln -sfn "${ENV_DIR}/nvvm/bin/cicc" "${RUNTIME_CUDA}/bin/cicc"
ln -sfn "${ENV_DIR}/targets/x86_64-linux/include" "${RUNTIME_CUDA}/include"
ln -sfn "${ENV_DIR}/nvvm" "${RUNTIME_CUDA}/nvvm"
ln -sfn "${ENV_DIR}/targets/x86_64-linux/lib/libcudart.so" \
    "${RUNTIME_CUDA}/lib64/libcudart.so"
ln -sfn /lib64/libcuda.so "${RUNTIME_CUDA}/lib64/stubs/libcuda.so"
ln -sfn "${ENV_DIR}/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12" \
    "${RUNTIME_CUDA}/lib64/libnvrtc.so.12"

export PATH="${RUNTIME_CUDA}/nvvm/bin:${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}"
export CUDA_HOME="${RUNTIME_CUDA}"
export CUDACXX="${ENV_DIR}/bin/nvcc"
export CUDNN_HOME="${ENV_DIR}/lib/python3.12/site-packages/nvidia/cudnn"
export CPATH="${RUNTIME_CUDA}/include:${CPATH:-}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib/python3.12/site-packages/torch/lib:${ENV_DIR}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${ENV_DIR}/lib/python3.12/site-packages/nvidia/cudnn/lib:${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export RAY_DEDUP_LOGS=0
export RAY_worker_register_timeout_seconds=300
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

cd "${SLIME_DIR}"
source scripts/models/qwen3-4B.sh

CKPT_ARGS=(
    --hf-checkpoint "${HF_MODEL}"
    --ref-load "${TORCH_DIST_MODEL}"
    --load "${OUTPUT_DIR}"
    --save "${OUTPUT_DIR}"
    --save-interval 20
    --save-retain-interval 3000
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --num-rollout 3000
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --rollout-max-response-len 8192
    --rollout-temperature 1
    --global-batch-size 256
    --balance-data
)

EVAL_ARGS=(
    --eval-interval 5
    --eval-prompt-data aime24 "${EVAL_DATA}"
    --n-samples-per-eval-prompt 4
    --eval-max-response-len 8192
    --eval-top-p 1
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
    --sglang-disable-custom-all-reduce
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

ray stop --force >/dev/null 2>&1 || true
ray start --head \
    --node-ip-address 127.0.0.1 \
    --num-gpus 8 \
    --disable-usage-stats \
    --dashboard-host=127.0.0.1 \
    --dashboard-port=8265

RUNTIME_ENV_JSON="{\"env_vars\":{\"PATH\":\"${PATH}\",\"PYTHONPATH\":\"${PYTHONPATH}\",\"CUDA_HOME\":\"${CUDA_HOME}\",\"CUDACXX\":\"${CUDACXX}\",\"CUDNN_HOME\":\"${CUDNN_HOME}\",\"CPATH\":\"${CPATH}\",\"LD_LIBRARY_PATH\":\"${LD_LIBRARY_PATH}\",\"HF_HUB_OFFLINE\":\"1\",\"TRANSFORMERS_OFFLINE\":\"1\",\"PYTHONUNBUFFERED\":\"1\",\"PYTHONFAULTHANDLER\":\"1\",\"RAY_DEDUP_LOGS\":\"0\",\"RAY_worker_register_timeout_seconds\":\"300\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"1\"}}"

ray job submit --address=http://127.0.0.1:8265 \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python "${SLIME_DIR}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"
