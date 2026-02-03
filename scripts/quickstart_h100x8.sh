#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

CONFIG=${CONFIG:-configs/h100x8.yaml}
SHARDS=${SHARDS:-8}
BASE_URL=${TEACHER_BASE_URL:-http://localhost:8000/v1}

if ! command -v uv >/dev/null 2>&1; then
  echo "[quickstart] ERROR: uv is not installed. Install from https://astral.sh/uv/"
  exit 1
fi

if [[ ! -d .venv ]]; then
  uv venv .venv
fi

uv pip install --python .venv/bin/python -e .
if [[ -z "${NO_METRICX:=}" ]]; then
  METRICX_PYTHON=.venv/bin/python ./scripts/install_metricx_official.sh
fi

if [[ -z "${VLLM_API_KEY:=}" ]]; then
  echo "[quickstart] ERROR: VLLM_API_KEY is not set"
  exit 1
fi

if [[ -z "${SKIP_VLLM:=}" ]]; then
  echo "[quickstart] launching vLLM in background"
  mkdir -p runs
  nohup ./scripts/launch_vllm_h100x8.sh > runs/vllm.log 2>&1 &
else
  if [[ -z "${TEACHER_BASE_URL:=}" ]]; then
    echo "[quickstart] ERROR: SKIP_VLLM=1 requires TEACHER_BASE_URL"
    exit 1
  fi
fi

echo "[quickstart] waiting for vLLM to become ready"
for i in $(seq 1 120); do
  if ! command -v curl >/dev/null 2>&1; then
    echo "[quickstart] curl not found; skipping readiness check"
    break
  fi
  if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" ${BASE_URL}/models >/dev/null; then
    echo "[quickstart] vLLM ready"
    break
  fi
  sleep 10
done

echo "[quickstart] running sharded pipeline"
CONFIG="$CONFIG" SHARDS="$SHARDS" ./scripts/run_sharded_pipeline.sh "$CONFIG"
