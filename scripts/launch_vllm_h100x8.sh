#!/usr/bin/env bash
set -euo pipefail

: "${VLLM_API_KEY:=token-abc123}"
: "${MODEL:=Qwen/Qwen3-235B-A22B-Instruct-2507}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${TP:=8}"
: "${DTYPE:=auto}"
: "${MAX_MODEL_LEN:=4096}"
: "${GPU_MEM_UTIL:=0.90}"
: "${CHAT_TEMPLATE:=}"

CMD=(
  vllm serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --api-key "$VLLM_API_KEY"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --generation-config vllm
)

if [[ -n "$CHAT_TEMPLATE" ]]; then
  CMD+=(--chat-template "$CHAT_TEMPLATE")
fi

echo "[launch_vllm] ${CMD[*]}"
exec "${CMD[@]}"
