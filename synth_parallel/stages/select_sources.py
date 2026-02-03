from __future__ import annotations

import heapq
import time
from typing import Any, Dict, List, Optional

from synth_parallel.utils.io import read_jsonl, write_jsonl
from synth_parallel.utils.logging import setup_logger, update_stats
from synth_parallel.utils.paths import resolve_inputs


def _push_heap(heap: List, item: Dict[str, Any], key: float, limit: int) -> None:
    if len(heap) < limit:
        heapq.heappush(heap, (key, item))
    else:
        if key > heap[0][0]:
            heapq.heapreplace(heap, (key, item))


def run(
    cfg: Dict[str, Any],
    run_dir: str,
    limit: Optional[int] = None,
) -> str:
    start = time.time()
    logger = setup_logger("synth_parallel", cfg["run"]["log_level"])
    input_path = f"{run_dir}/prefilter_candidates.jsonl"
    output_path = f"{run_dir}/selected_sources.jsonl"

    target_total = cfg["data"]["target_examples_total"]
    per_bucket = cfg.get("selection", {}).get("per_bucket", False)
    buckets = cfg["bucketing"]["boundaries"]
    num_buckets = len(buckets) - 1
    bucket_quota = max(1, target_total // num_buckets)

    if per_bucket:
        heaps = {i: [] for i in range(num_buckets)}
    else:
        heap: List = []

    seen = 0
    log_every = cfg["run"].get("log_every", 10000)
    for path in resolve_inputs(input_path):
        for rec in read_jsonl(path):
            if limit and seen >= limit:
                break
            seen += 1
            improvement = float(rec.get("improvement", 0.0))
            if per_bucket:
                bucket_id = int(rec.get("length_bucket_id", 0))
                _push_heap(heaps[bucket_id], rec, improvement, bucket_quota)
            else:
                _push_heap(heap, rec, improvement, target_total)
            if log_every and seen % log_every == 0:
                logger.info("select_sources progress: seen=%s", seen)
        if limit and seen >= limit:
            break

    if per_bucket:
        selected = []
        for heap in heaps.values():
            selected.extend([item for _, item in heap])
    else:
        selected = [item for _, item in heap]

    # Sort by improvement descending
    selected.sort(key=lambda x: float(x.get("improvement", 0.0)), reverse=True)
    selected = selected[:target_total]

    write_jsonl(output_path, selected, append=False)
    logger.info("select_sources wrote=%s output=%s", len(selected), output_path)

    update_stats(
        run_dir,
        "select_sources",
        {
            "seen": seen,
            "selected": len(selected),
            "duration_s": round(time.time() - start, 3),
        },
    )
    return output_path
