#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git curl

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

uv venv .venv
uv pip install --python .venv/bin/python -e .
# MetricX official deps
METRICX_PYTHON=.venv/bin/python ./scripts/install_metricx_official.sh

# vLLM installation is environment-specific for Qwen3-235B. Follow vLLM docs if needed.
