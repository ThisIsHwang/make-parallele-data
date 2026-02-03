#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

REPO_DIR=${REPO_DIR:-third_party/metricx}
PY_BIN=${METRICX_PYTHON:-}

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "[metricx] cloning repo"
  git clone https://github.com/google-research/metricx "$REPO_DIR"
else
  echo "[metricx] updating repo"
  (cd "$REPO_DIR" && git pull --rebase)
fi

if [[ -z "$PY_BIN" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PY_BIN=.venv/bin/python
  else
    PY_BIN=python3
  fi
fi

if [[ -f "$REPO_DIR/requirements.txt" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "[metricx] ERROR: uv is required. Install from https://astral.sh/uv/"
    exit 1
  fi
  uv pip install --python "$PY_BIN" -r "$REPO_DIR/requirements.txt"
else
  echo "[metricx] requirements.txt not found in $REPO_DIR" >&2
fi

cat <<'MSG'
[metricx] NOTE:
- Install a CUDA-enabled torch build if you want GPU scoring.
- MetricX-24 XXL is large; ensure 충분한 GPU 메모리.
- requirements.txt pins transformers==4.30.2 and datasets==2.13.1 (may downgrade your env).
MSG
