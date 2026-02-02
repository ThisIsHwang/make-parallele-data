from __future__ import annotations

import argparse
import os
import sys

from synth_parallel.config import Config, load_config, resolve_paths, validate_config
from synth_parallel.pipeline import list_stages, run_stage
from synth_parallel.utils.logging import setup_logger


def _apply_dry_run(cfg: Config) -> None:
    cfg.data["data"]["sample_pool_size"] = min(1000, cfg.data["data"]["sample_pool_size"])
    cfg.data["data"]["target_examples_total"] = min(
        100, cfg.data["data"]["target_examples_total"]
    )
    cfg.data["final_generation"]["num_candidates"] = min(
        8, cfg.data["final_generation"]["num_candidates"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synth Parallel pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a pipeline stage")
    run_parser.add_argument("--config", required=True, help="Path to YAML config")
    run_parser.add_argument("--stage", required=True, choices=list_stages())
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--shard-id", type=int, default=0)
    run_parser.add_argument("--num-shards", type=int, default=1)

    args = parser.parse_args()

    cfg = resolve_paths(load_config(args.config), args.config)
    validate_config(cfg)
    if args.dry_run:
        _apply_dry_run(cfg)

    run_dir = cfg.data["run"]["out_dir"]
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger("synth_parallel", cfg.data["run"]["log_level"])
    logger.info("Running stage %s", args.stage)

    output = run_stage(
        cfg.data,
        run_dir,
        args.stage,
        limit=args.limit,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    logger.info("Stage %s completed: %s", args.stage, output)


if __name__ == "__main__":
    main()
