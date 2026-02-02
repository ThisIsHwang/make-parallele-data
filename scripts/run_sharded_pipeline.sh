#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/h100x8.yaml}
SHARDS=${SHARDS:-8}
RESUME_FLAG=${RESUME_FLAG:---resume}
BIN=synth_parallel
if [[ -x .venv/bin/synth_parallel ]]; then
  BIN=.venv/bin/synth_parallel
fi

run_stage() {
  local stage=$1
  echo "[pipeline] stage=${stage}"
  $BIN run --config "$CONFIG" --stage "$stage"
}

run_sharded() {
  local stage=$1
  echo "[pipeline] stage=${stage} (sharded ${SHARDS})"
  for i in $(seq 0 $((SHARDS-1))); do
    $BIN run --config "$CONFIG" --stage "$stage" \
      --shard-id "$i" --num-shards "$SHARDS" $RESUME_FLAG &
  done
  wait
}

run_stage sample_sources
run_sharded prefilter_score
run_stage select_sources
run_sharded generate_128
run_sharded score_select_best
run_sharded format_filter
run_stage export

echo "[pipeline] done"
