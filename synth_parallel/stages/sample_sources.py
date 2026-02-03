from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from synth_parallel.data.madlad import iter_segments, resolve_lang
from synth_parallel.sampling.bucketer import BucketSampler, LengthMeasurer, find_bucket
from synth_parallel.utils.hash import hash_record
from synth_parallel.utils.io import write_jsonl
from synth_parallel.utils.logging import setup_logger, update_stats
from synth_parallel.utils.text import approx_token_len


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
    _, resolved_lang = resolve_lang(cfg)
    total_target = cfg["data"]["sample_pool_size"]
    blob_cfg = cfg["final_generation"]["blob"]

    blob_enabled = bool(blob_cfg.get("enabled", False))
    blob_ratio = blob_cfg.get("blob_ratio", 0.0) if blob_enabled else 0.0
    blob_target = int(total_target * blob_ratio)
    sentence_target = total_target - blob_target

    boundaries = cfg["bucketing"]["boundaries"]
    per_bucket = cfg["bucketing"]["per_bucket_quota"]
    spillover = cfg["bucketing"].get("spillover_reservoir", 200000)
    measurer = LengthMeasurer(
        cfg["bucketing"]["measure"], cfg["bucketing"].get("tokenizer_name")
    )

    sentence_sampler = BucketSampler(
        boundaries=boundaries,
        total_target=sentence_target,
        per_bucket_quota=None if per_bucket == "auto" else int(per_bucket),
        spillover_capacity=spillover,
        seed=cfg["run"]["seed"],
    )
    blob_sampler = BucketSampler(
        boundaries=boundaries,
        total_target=blob_target,
        per_bucket_quota=None if per_bucket == "auto" else int(per_bucket),
        spillover_capacity=spillover,
        seed=cfg["run"]["seed"],
    )

    output_path = f"{run_dir}/sampled_sources.jsonl"
    if resume and not overwrite and os.path.exists(output_path):
        logger.info("sample_sources resume: using existing %s", output_path)
        return output_path
    if overwrite:
        open(output_path, "wb").close()

    seen = 0
    log_every = cfg["run"].get("log_every", 10000)
    for seg in iter_segments(cfg, limit=limit, include_blob=blob_enabled):
        text = seg["source_text"]
        length = measurer(text)
        bucket_id = find_bucket(boundaries, length)
        source_id = hash_record(seg["doc_id"], seg["segment_index"], seg.get("segment_type"), text)
        record = {
            "source_id": source_id,
            "source_text": text,
            "length": length,
            "length_bucket_id": bucket_id,
            "segment_type": seg.get("segment_type", "sentence"),
            "madlad": {
                "lang": resolved_lang,
                "split": cfg["data"]["madlad_split"],
                "doc_id": seg["doc_id"],
                "segment_index": seg["segment_index"],
                "segment_span": seg.get("segment_span"),
            },
        }

        if record["segment_type"] == "blob":
            blob_sampler.add(bucket_id, record)
        else:
            sentence_sampler.add(bucket_id, record)
        seen += 1
        if log_every and seen % log_every == 0:
            logger.info("sample_sources progress: seen=%s", seen)

    records = sentence_sampler.finalize() + blob_sampler.finalize()

    write_jsonl(output_path, records, append=False)
    logger.info("sample_sources wrote=%s output=%s", len(records), output_path)

    update_stats(
        run_dir,
        "sample_sources",
        {
            "seen": seen,
            "sampled": len(records),
            "sentence_target": sentence_target,
            "blob_target": blob_target,
            "duration_s": round(time.time() - start, 3),
        },
    )
    return output_path
