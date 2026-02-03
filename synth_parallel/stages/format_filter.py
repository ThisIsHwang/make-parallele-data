from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from synth_parallel.filters import LLMJudge, RuleBasedFilter
from synth_parallel.teacher.client import AsyncTeacherClient
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
    input_path = shard_path(f"{run_dir}/selected_best.jsonl", shard_id, num_shards)
    output_path = shard_path(f"{run_dir}/filtered.jsonl", shard_id, num_shards)
    rejected_path = shard_path(f"{run_dir}/rejected.jsonl", shard_id, num_shards)

    if overwrite:
        open(output_path, "wb").close()
        open(rejected_path, "wb").close()

    rule_filter = RuleBasedFilter(cfg)
    client = AsyncTeacherClient(cfg)
    judge = LLMJudge(cfg, client)
    write_lock = asyncio.Lock()

    async def _process_one(rec: Dict[str, Any]):
        source = rec["source_text"]
        target = rec["target_text"]
        if cfg["filters"].get("rule_based", True):
            passed, reason = rule_filter.check(source, target)
            if not passed:
                async with write_lock:
                    write_jsonl_one(
                        rejected_path,
                        {**rec, "reject_reason": reason},
                        append=True,
                    )
                return False

        if judge.config.enabled:
            ok, reason = await judge.check(
                cfg["data"]["src_lang"],
                cfg["data"]["tgt_lang"],
                source,
                target,
            )
            if not ok:
                async with write_lock:
                    write_jsonl_one(
                        rejected_path,
                        {**rec, "reject_reason": reason},
                        append=True,
                    )
                return False

        async with write_lock:
            write_jsonl_one(output_path, rec, append=True)
        return True

    log_every = cfg["run"].get("log_every", 10000)
    async def _run_async():
        tasks = []
        processed = 0
        for path in resolve_inputs(input_path):
            for rec in read_jsonl(path):
                if limit and processed >= limit:
                    break
                tasks.append(asyncio.create_task(_process_one(rec)))
                processed += 1
                if log_every and processed % log_every == 0:
                    logger.info(
                        "format_filter progress: processed=%s shard=%s/%s",
                        processed,
                        shard_id,
                        num_shards,
                    )
                if len(tasks) >= cfg["teacher"]["max_concurrency"]:
                    await asyncio.gather(*tasks)
                    tasks = []
            if limit and processed >= limit:
                break
        if tasks:
            await asyncio.gather(*tasks)
        return processed

    processed = asyncio.run(_run_async())

    update_stats(
        run_dir,
        "format_filter",
        {
            "processed": processed,
            "duration_s": round(time.time() - start, 3),
        },
    )
    return output_path
