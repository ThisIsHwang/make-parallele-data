from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from synth_parallel.metricx import MetricXScorer
from synth_parallel.utils.io import read_jsonl, write_jsonl_one
from synth_parallel.utils.logging import setup_logger, update_stats
from synth_parallel.utils.paths import shard_path, resolve_inputs


def run(
    cfg: Dict[str, Any],
    run_dir: str,
    limit: Optional[int] = None,
    shard_id: int = 0,
    num_shards: int = 1,
    resume: bool = False,
    overwrite: bool = False,
) -> str:
    start = time.time()
    logger = setup_logger("synth_parallel", cfg["run"]["log_level"])
    sources_path = f"{run_dir}/selected_sources.jsonl"
    candidates_path = shard_path(f"{run_dir}/candidates_128.jsonl", shard_id, num_shards)
    output_path = shard_path(f"{run_dir}/selected_best.jsonl", shard_id, num_shards)

    if overwrite:
        open(output_path, "wb").close()

    store_top_k = cfg["final_generation"].get("store_top_k", 1)

    sources_map = {}
    for rec in read_jsonl(sources_path):
        sources_map[rec["source_id"]] = rec

    metric_cfg = cfg
    if num_shards > 1:
        metric_cfg = {**cfg, "metricx": {**cfg["metricx"]}}
        base_db = metric_cfg["metricx"].get("cache_db")
        if base_db:
            metric_cfg["metricx"]["cache_db"] = shard_path(base_db, shard_id, num_shards)
    scorer = MetricXScorer(metric_cfg)

    processed = 0
    log_every = cfg["run"].get("log_every", 10000)
    for path in resolve_inputs(candidates_path):
        for rec in read_jsonl(path):
            if limit and processed >= limit:
                break
            sid = rec["source_id"]
            source = sources_map.get(sid)
            if not source:
                continue
            source_text = source["source_text"]
            hyps = rec.get("translations", [])
            if not hyps:
                continue
            scores = scorer.score_batch([source_text] * len(hyps), hyps)
            ranked = sorted(zip(scores, hyps), key=lambda x: float(x[0]))
            best_score, best_text = ranked[0]
            record = {
                "source_id": sid,
                "source_text": source_text,
                "target_text": best_text,
                "metricx_qe_score_best": float(best_score),
                "madlad": source.get("madlad"),
                "length_bucket_id": source.get("length_bucket_id"),
                "segment_type": source.get("segment_type"),
            }
            if store_top_k and store_top_k > 1:
                record["top_k"] = [
                    {"score": float(score), "translation": text}
                    for score, text in ranked[:store_top_k]
                ]
            write_jsonl_one(output_path, record, append=True)
            processed += 1
            if log_every and processed % log_every == 0:
                logger.info(
                    "score_select_best progress: processed=%s shard=%s/%s",
                    processed,
                    shard_id,
                    num_shards,
                )
        if limit and processed >= limit:
            break

    update_stats(
        run_dir,
        "score_select_best",
        {
            "processed": processed,
            "duration_s": round(time.time() - start, 3),
        },
    )
    return output_path
