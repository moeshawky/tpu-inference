#!/usr/bin/env bash
set -u
export PJRT_DEVICE=TPU
export TPU_SKIP_MDS_QUERY=1
export TPU_ACCELERATOR_TYPE=v5litepod-8
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_WORKER_ID=0
export TPU_WORKER_HOSTNAMES=localhost
export TPU_PROCESS_ADDRESSES=local
export TPU_CHIPS_PER_HOST_BOUNDS=2,4,1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export HF_HOME=/dev/shm/hf
export VLLM_TARGET_DEVICE=tpu
export VLLM_XLA_CACHE_PATH=/dev/shm/vllm_xla_cache_flash_tp4_bank3
export QWEN_BANK_DIR=/kaggle/input/datasets/moeshawky/qwen38-flash-expert-banks
export MOE_EXPERT_OFFLOAD=1
export MOE_EXPERT_OFFLOAD_SLOTS=32
export MOE_EXPERT_OFFLOAD_LAYERS=""
export MOE_EXPERT_OFFLOAD_HOST_MEMORY_GUARD=1
export MOE_EXPERT_OFFLOAD_HOST_MEMORY_RESERVE_GIB=12
export MOE_EXPERT_OFFLOAD_CPU_WORKING_SET_GIB=16
export MOE_EXPERT_OFFLOAD_DISK_BACKED=0
export MOE_EXPERT_OFFLOAD_STORE=0
mkdir -p /dev/shm/hf /dev/shm/vllm_xla_cache_flash_tp4_bank3
MODEL=/kaggle/input/models/manarmilad/qwen3.8-flash-next-uncensored/transformers/fp8/1
echo "[flash_tp4_bank3] $(date) start wall=$(date +%s) HEAD=$(git -C /kaggle/working/vllm-src rev-parse HEAD)"
echo "[bank] count $(ls $QWEN_BANK_DIR/layer_*.bank 2>/dev/null | wc -l)"
kaggle-backend run vllm-tpu -- vllm serve "$MODEL" \
  --tensor_parallel_size 4 \
  --dtype bfloat16 \
  --max_model_len 8192 \
  --max_num_seqs 32 \
  --max_num_batched_tokens 16384 \
  --gpu_memory_utilization 0.70 \
  --enable_prefix_caching \
  --mamba_cache_mode align \
  --reasoning_parser qwen3 \
  --enable_auto_tool_choice \
  --tool_call_parser qwen3_coder \
  --trust-remote-code \
  --language-model-only \
  --port 8000 \
  --served-model-name qwen38-flash
