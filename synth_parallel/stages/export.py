from __future__ import annotations

import time
from typing import Any, Dict, Optional

from synth_parallel.utils.io import maybe_write_parquet, read_jsonl, write_jsonl
from synth_parallel.utils.logging import update_stats
from synth_parallel.utils.paths import resolve_inputs
from synth_parallel.utils.versions import collect_versions


def run(
    cfg: Dict[str, Any],
    run_dir: str,
    limit: Optional[int] = None,
) -> str:
    start = time.time()
    input_path = f"{run_dir}/filtered.jsonl"
    selected_sources_path = f"{run_dir}/selected_sources.jsonl"

    selection_map = {}
    for rec in read_jsonl(selected_sources_path):
        selection_map[rec["source_id"]] = {
            "improvement": rec.get("improvement"),
            "score_greedy": rec.get("score_greedy"),
            "score_sample": rec.get("score_sample"),
        }

    pair_id = f"{cfg['data']['src_lang']}->{cfg['data']['tgt_lang']}"
    teacher_meta = {
        "backend": cfg["teacher"]["backend"],
        "base_url": cfg["teacher"]["base_url"],
        "model": cfg["teacher"]["model"],
        "sampling_params": cfg["teacher"]["generation"],
    }
    metricx_meta = {
        "checkpoint": cfg["metricx"]["checkpoint"],
        "backend": cfg["metricx"].get("backend", "hf"),
        "versions": collect_versions(),
    }
    filters_meta = {
        "rule_based": cfg["filters"].get("rule_based", True),
        "llm_judge": cfg["filters"]["llm_judge"].get("enabled", False),
    }

    outputs = []
    processed = 0
    for path in resolve_inputs(input_path):
        for rec in read_jsonl(path):
            if limit and processed >= limit:
                break
            source_id = rec["source_id"]
            selection = selection_map.get(source_id, {})
            out = {
                "pair_id": pair_id,
                "source_lang_code": cfg["data"]["src_lang"],
                "target_lang_code": cfg["data"]["tgt_lang"],
                "source_text": rec["source_text"],
                "target_text": rec["target_text"],
                "metricx_qe_score_best": rec["metricx_qe_score_best"],
                "selection": {
                    "source_id": source_id,
                    "length_bucket_id": rec.get("length_bucket_id"),
                    "segment_type": rec.get("segment_type"),
                    **selection,
                },
                "madlad": rec.get("madlad"),
                "teacher": teacher_meta,
                "metricx": metricx_meta,
                "filters": filters_meta,
            }
            outputs.append(out)
            processed += 1
        if limit and processed >= limit:
            break

    fmt = cfg["export"].get("format", "jsonl")
    if fmt == "parquet":
        output_path = f"{run_dir}/final.parquet"
        maybe_write_parquet(output_path, outputs)
    else:
        output_path = f"{run_dir}/final.jsonl"
        write_jsonl(output_path, outputs, append=False)

    update_stats(
        run_dir,
        "export",
        {
            "processed": processed,
            "duration_s": round(time.time() - start, 3),
            "format": fmt,
        },
    )
    return output_path
