#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-third_party/metricx}

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "[metricx] cloning repo"
  git clone https://github.com/google-research/metricx "$REPO_DIR"
else
  echo "[metricx] updating repo"
  (cd "$REPO_DIR" && git pull --rebase)
fi

pip install -U pip

if [[ -f "$REPO_DIR/requirements.txt" ]]; then
  pip install -r "$REPO_DIR/requirements.txt"
else
  echo "[metricx] requirements.txt not found in $REPO_DIR" >&2
fi

cat <<'MSG'
[metricx] NOTE:
- Install a CUDA-enabled torch build if you want GPU scoring.
- MetricX-24 XXL is large; ensure 충분한 GPU 메모리.
- requirements.txt pins transformers==4.30.2 and datasets==2.13.1 (may downgrade your env).
MSG
