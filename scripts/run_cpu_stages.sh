#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/h100x8.yaml}
SHARDS=${SHARDS:-8}
RESUME_FLAG=${RESUME_FLAG:---resume}
BIN=synth_parallel
if [[ -x .venv/bin/synth_parallel ]]; then
  BIN=.venv/bin/synth_parallel
fi

SAMPLE=${SAMPLE:-1}
SELECT=${SELECT:-1}
GENERATE=${GENERATE:-1}
FILTER=${FILTER:-1}
EXPORT=${EXPORT:-1}

run_stage() {
  local stage=$1
  echo "[cpu] stage=${stage}"
  $BIN run --config "$CONFIG" --stage "$stage" $RESUME_FLAG
}

run_sharded() {
  local stage=$1
  echo "[cpu] stage=${stage} (sharded ${SHARDS})"
  for i in $(seq 0 $((SHARDS-1))); do
    $BIN run --config "$CONFIG" --stage "$stage" \
      --shard-id "$i" --num-shards "$SHARDS" $RESUME_FLAG &
  done
  wait
}

if [[ "$SAMPLE" == "1" ]]; then
  run_stage sample_sources
fi

if [[ "$SELECT" == "1" ]]; then
  run_stage select_sources
fi

if [[ "$GENERATE" == "1" ]]; then
  run_sharded generate_128
fi

if [[ "$FILTER" == "1" ]]; then
  run_sharded format_filter
fi

if [[ "$EXPORT" == "1" ]]; then
  run_stage export
fi

