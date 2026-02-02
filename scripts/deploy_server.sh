#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_HOST:?set TARGET_HOST}"
: "${TARGET_PATH:?set TARGET_PATH}"
: "${TARGET_USER:=}"

REMOTE="$TARGET_HOST"
if [[ -n "$TARGET_USER" ]]; then
  REMOTE="${TARGET_USER}@${TARGET_HOST}"
fi

RSYNC_OPTS=(
  -az
  --delete
  --exclude '.venv'
  --exclude '__pycache__'
  --exclude '.git'
  --exclude 'runs'
  --exclude '*.pyc'
)

echo "[deploy] rsync to ${REMOTE}:${TARGET_PATH}"
rsync "${RSYNC_OPTS[@]}" ./ "${REMOTE}:${TARGET_PATH}"
