#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git curl

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
# For MetricX HF backend:
# pip install -e .[metricx]

# vLLM installation is environment-specific for Qwen3-235B. Follow vLLM docs if needed.
