#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

export TORCHDYNAMO_DISABLE=1            # ← PyTorch 2.x에서 dynamo 끄기
export VLLM_TORCH_COMPILE=none          # ← vLLM이 torch.compile 안 쓰게

rm -rf ~/.cache/torch/inductor ~/.cache/torch/extension 2>/dev/null || true

export VLLM_MAX_MODEL_LEN=6144
export MAX_NEW_TOKENS=2048

python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen3-1.7B" \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 6144 \
  --enforce-eager \
  --gpu-memory-utilization 0.7 \
  --trust-remote-code \
  --host 127.0.0.1 --port 8010
