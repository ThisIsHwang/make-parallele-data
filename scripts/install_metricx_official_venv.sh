#!/usr/bin/env bash
set -euo pipefail

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
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install -U pip

if [[ -f "$REPO_DIR/requirements.txt" ]]; then
  pip install -r "$REPO_DIR/requirements.txt"
else
  echo "[metricx] requirements.txt not found in $REPO_DIR" >&2
fi

cat <<'MSG'
[metricx] NOTE:
- This installs MetricX deps into a separate venv.
- To use it, set metricx.backend: official_cli and python_bin: ./.metricx-venv/bin/python
- Official requirements pin transformers==4.30.2 and datasets==2.13.1.
MSG
