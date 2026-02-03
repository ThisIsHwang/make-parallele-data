from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from synth_parallel.teacher.client import AsyncTeacherClient, GenerationParams
from synth_parallel.teacher.prompts import build_translation_messages
from synth_parallel.utils.io import read_jsonl, write_jsonl_one
from synth_parallel.utils.logging import setup_logger, update_stats
from synth_parallel.utils.shard import in_shard
from synth_parallel.utils.paths import shard_path


async def _generate_for_source(
    client: AsyncTeacherClient,
    cfg: Dict[str, Any],
    source_id: str,
    source_text: str,
    num_candidates: int,
    strategy: str,
) -> List[str]:
    meta = {
        "source_lang": cfg["data"]["src_lang"],
        "target_lang": cfg["data"]["tgt_lang"],
        "src_lang_code": cfg["data"]["src_lang"],
        "tgt_lang_code": cfg["data"]["tgt_lang"],
        "text": source_text,
    }
    messages = build_translation_messages(meta)
    gen_cfg = cfg["teacher"]["generation"]
    params = GenerationParams(
        temperature=gen_cfg["final_temperature"],
        top_p=gen_cfg["top_p"],
        max_tokens=gen_cfg["max_tokens"],
        n=num_candidates,
        seed=gen_cfg.get("seed"),
    )

    if strategy in ("auto", "n_parameter"):
        try:
            outputs = await client.generate(messages, params)
            if len(outputs) == num_candidates:
                return outputs
        except Exception:
            if strategy == "n_parameter":
                raise

    # Fallback: per-call
    outputs: List[str] = []
    single_params = GenerationParams(
        temperature=gen_cfg["final_temperature"],
        top_p=gen_cfg["top_p"],
        max_tokens=gen_cfg["max_tokens"],
        n=1,
        seed=gen_cfg.get("seed"),
    )
    for i in range(num_candidates):
        out = await client.generate(
            messages, single_params, request_id=f"{source_id}-cand-{i}"
        )
        outputs.append(out[0])
    return outputs


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
    input_path = f"{run_dir}/selected_sources.jsonl"
    output_path = shard_path(f"{run_dir}/candidates_128.jsonl", shard_id, num_shards)

    if overwrite:
        open(output_path, "wb").close()

    num_candidates = cfg["final_generation"]["num_candidates"]
    strategy = cfg["final_generation"].get("strategy", "auto")

    completed_counts: Dict[str, int] = {}
    if resume and not overwrite:
        try:
            for rec in read_jsonl(output_path):
                sid = rec["source_id"]
                completed_counts[sid] = completed_counts.get(sid, 0) + 1
        except FileNotFoundError:
            pass

    client = AsyncTeacherClient(cfg)
    write_lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()

    processed = 0
    log_every = cfg["run"].get("log_every", 10000)

    async def worker():
        nonlocal processed
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            sid = item["source_id"]
            texts = await _generate_for_source(
                client, cfg, item["source_id"], item["source_text"], num_candidates, strategy
            )
            async with write_lock:
                write_jsonl_one(
                    output_path,
                    {
                        "source_id": sid,
                        "translations": texts,
                    },
                    append=True,
                )
            processed += 1
            if log_every and processed % log_every == 0:
                logger.info(
                    "generate_128 progress: processed=%s shard=%s/%s",
                    processed,
                    shard_id,
                    num_shards,
                )
            queue.task_done()

    async def _run_async():
        workers = [
            asyncio.create_task(worker())
            for _ in range(max(1, cfg["teacher"]["max_concurrency"] // 4))
        ]
        queued = 0
        for rec in read_jsonl(input_path):
            if limit and queued >= limit:
                break
            if not in_shard(rec["source_id"], shard_id, num_shards):
                continue
            if completed_counts.get(rec["source_id"], 0) >= 1:
                continue
            await queue.put(rec)
            queued += 1
        for _ in workers:
            await queue.put(None)
        await queue.join()
        for w in workers:
            await w
        return queued

    queued = asyncio.run(_run_async())

    update_stats(
        run_dir,
        "generate_128",
        {
            "queued_sources": queued,
            "duration_s": round(time.time() - start, 3),
            "num_candidates": num_candidates,
        },
    )
    return output_path
