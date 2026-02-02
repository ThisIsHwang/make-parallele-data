from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from synth_parallel.metricx import MetricXScorer
from synth_parallel.teacher.client import AsyncTeacherClient, GenerationParams
from synth_parallel.teacher.prompts import build_translation_messages
from synth_parallel.utils.io import read_jsonl, write_jsonl
from synth_parallel.utils.logging import update_stats
from synth_parallel.utils.shard import in_shard
from synth_parallel.utils.paths import shard_path


async def _generate_batch(
    client: AsyncTeacherClient,
    batch: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    tasks = []
    for item in batch:
        meta = {
            "source_lang": cfg["data"]["src_lang"],
            "target_lang": cfg["data"]["tgt_lang"],
            "src_lang_code": cfg["data"]["src_lang"],
            "tgt_lang_code": cfg["data"]["tgt_lang"],
            "text": item["source_text"],
        }
        messages = build_translation_messages(meta)
        gen_cfg = cfg["teacher"]["generation"]
        greedy_params = GenerationParams(
            temperature=gen_cfg["greedy_temperature"],
            top_p=gen_cfg["top_p"],
            max_tokens=gen_cfg["max_tokens"],
            n=1,
            seed=gen_cfg.get("seed"),
        )
        sample_params = GenerationParams(
            temperature=gen_cfg["sample_temperature"],
            top_p=gen_cfg["top_p"],
            max_tokens=gen_cfg["max_tokens"],
            n=1,
            seed=gen_cfg.get("seed"),
        )
        tasks.append(client.generate(messages, greedy_params))
        tasks.append(client.generate(messages, sample_params))

    outputs = await asyncio.gather(*tasks)
    results = []
    for i, item in enumerate(batch):
        greedy_text = outputs[2 * i][0]
        sample_text = outputs[2 * i + 1][0]
        results.append({"greedy_text": greedy_text, "sample_text": sample_text})
    return results


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
    input_path = f"{run_dir}/sampled_sources.jsonl"
    output_path = shard_path(f"{run_dir}/prefilter_candidates.jsonl", shard_id, num_shards)

    processed = set()
    if resume and not overwrite:
        try:
            for rec in read_jsonl(output_path):
                processed.add(rec["source_id"])
        except FileNotFoundError:
            pass
    elif overwrite:
        open(output_path, "wb").close()

    client = AsyncTeacherClient(cfg)
    metric_cfg = cfg
    if num_shards > 1:
        metric_cfg = {**cfg, "metricx": {**cfg["metricx"]}}
        base_db = metric_cfg["metricx"].get("cache_db")
        if base_db:
            metric_cfg["metricx"]["cache_db"] = shard_path(base_db, shard_id, num_shards)
    scorer = MetricXScorer(metric_cfg)

    batch: List[Dict[str, Any]] = []
    total = 0
    kept = 0
    batch_size = max(1, cfg["teacher"]["max_concurrency"])

    async def _process_batch(batch_items: List[Dict[str, Any]]):
        nonlocal kept
        outputs = await _generate_batch(client, batch_items, cfg)
        sources = [item["source_text"] for item in batch_items]
        greedy = [out["greedy_text"] for out in outputs]
        sample = [out["sample_text"] for out in outputs]

        scores = scorer.score_batch(sources * 2, greedy + sample)
        greedy_scores = scores[: len(batch_items)]
        sample_scores = scores[len(batch_items) :]

        batch_records = []
        for item, g_text, s_text, g_score, s_score in zip(
            batch_items, greedy, sample, greedy_scores, sample_scores
        ):
            improvement = float(g_score - s_score)
            rec = {
                "source_id": item["source_id"],
                "source_text": item["source_text"],
                "length_bucket_id": item.get("length_bucket_id"),
                "segment_type": item.get("segment_type"),
                "madlad": item.get("madlad"),
                "score_greedy": float(g_score),
                "score_sample": float(s_score),
                "improvement": improvement,
            }
            if cfg["prefilter"].get("store_translations", False):
                rec["greedy_text"] = g_text
                rec["sample_text"] = s_text
            batch_records.append(rec)
            kept += 1
        if batch_records:
            write_jsonl(output_path, batch_records, append=True)

    async def _run_async():
        nonlocal total, batch
        for rec in read_jsonl(input_path):
            if limit and total >= limit:
                break
            if not in_shard(rec["source_id"], shard_id, num_shards):
                continue
            if rec["source_id"] in processed:
                continue
            batch.append(rec)
            total += 1
            if len(batch) >= batch_size:
                await _process_batch(batch)
                batch = []
        if batch:
            await _process_batch(batch)

    asyncio.run(_run_async())

    update_stats(
        run_dir,
        "prefilter_score",
        {
            "total": total,
            "processed": kept,
            "duration_s": round(time.time() - start, 3),
            "shard_id": shard_id,
            "num_shards": num_shards,
        },
    )
    return output_path
