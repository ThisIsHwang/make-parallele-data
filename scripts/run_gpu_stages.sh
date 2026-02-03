#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/h100x8.yaml}
SHARDS=${SHARDS:-8}
RESUME_FLAG=${RESUME_FLAG:---resume}
BIN=synth_parallel
if [[ -x .venv/bin/synth_parallel ]]; then
  BIN=.venv/bin/synth_parallel
fi

PREFILTER=${PREFILTER:-1}
SCORE=${SCORE:-1}

run_sharded() {
  local stage=$1
  echo "[gpu] stage=${stage} (sharded ${SHARDS})"
  for i in $(seq 0 $((SHARDS-1))); do
    $BIN run --config "$CONFIG" --stage "$stage" \
      --shard-id "$i" --num-shards "$SHARDS" $RESUME_FLAG &
  done
  wait
}

if [[ "$PREFILTER" == "1" ]]; then
  run_sharded prefilter_score
fi

if [[ "$SCORE" == "1" ]]; then
  run_sharded score_select_best
fi

