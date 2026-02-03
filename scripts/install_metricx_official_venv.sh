#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

REPO_DIR=${REPO_DIR:-third_party/metricx}
VENV_DIR=${VENV_DIR:-.metricx-venv}

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "[metricx] cloning repo"
  git clone https://github.com/google-research/metricx "$REPO_DIR"
else
  echo "[metricx] updating repo"
  (cd "$REPO_DIR" && git pull --rebase)
fi

if [[ ! -d "$VENV_DIR" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "[metricx] ERROR: uv is required. Install from https://astral.sh/uv/"
    exit 1
  fi
  uv python install 3.11
  uv venv --python 3.11 "$VENV_DIR"
fi

if [[ -f "$REPO_DIR/requirements.txt" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "[metricx] ERROR: uv is required. Install from https://astral.sh/uv/"
    exit 1
  fi
  uv pip install --python "$VENV_DIR/bin/python" -r "$REPO_DIR/requirements.txt"
else
  echo "[metricx] requirements.txt not found in $REPO_DIR" >&2
fi

cat <<'MSG'
[metricx] NOTE:
- This installs MetricX deps into a separate venv.
- To use it, set metricx.backend: official_cli and python_bin: ./.metricx-venv/bin/python
- Official requirements pin transformers==4.30.2 and datasets==2.13.1.
MSG
