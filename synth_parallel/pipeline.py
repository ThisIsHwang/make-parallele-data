from __future__ import annotations

from typing import Any, Dict, Optional

from synth_parallel.stages import (
    export,
    format_filter,
    generate_128,
    prefilter_score,
    sample_sources,
    score_select_best,
    select_sources,
)

STAGE_FUNCS = {
    "sample_sources": sample_sources.run,
    "prefilter_score": prefilter_score.run,
    "select_sources": select_sources.run,
    "generate_128": generate_128.run,
    "score_select_best": score_select_best.run,
    "format_filter": format_filter.run,
    "export": export.run,
}


def run_stage(
    cfg: Dict[str, Any],
    run_dir: str,
    stage: str,
    limit: Optional[int] = None,
    shard_id: int = 0,
    num_shards: int = 1,
    resume: bool = False,
    overwrite: bool = False,
) -> str:
    if stage not in STAGE_FUNCS:
        raise ValueError(f"Unknown stage: {stage}")
    func = STAGE_FUNCS[stage]
    if stage in ("prefilter_score", "generate_128", "score_select_best", "format_filter"):
        return func(
            cfg,
            run_dir,
            limit=limit,
            shard_id=shard_id,
            num_shards=num_shards,
            resume=resume,
            overwrite=overwrite,
        )
    if stage in ("sample_sources", "select_sources", "export"):
        return func(cfg, run_dir, limit=limit)
    return func(cfg, run_dir)  # type: ignore


def list_stages() -> list[str]:
    return list(STAGE_FUNCS.keys())
