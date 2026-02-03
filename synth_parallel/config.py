from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "run": {
        "out_dir": "./runs/exp001",
        "seed": 1234,
        "log_level": "INFO",
        "log_every": 10000,
    },
    "data": {
        "madlad_dataset": "allenai/madlad-400",
        "madlad_split": "clean",
        "src_lang": "kor",
        "tgt_lang": "eng",
        "target_examples_total": 10000,
        "sample_pool_size": 1000000,
        "streaming": True,
        "hf_endpoint": None,
        "hf_timeout_s": 120,
        "hf_enable_hf_transfer": False,
    },
    "segmentation": {
        "mode": "auto",
        "min_chars": 20,
        "max_chars": 5000,
        "merge_short": True,
        "allow_html": False,
    },
    "bucketing": {
        "boundaries": [0, 10, 20, 40, 80, 120, 200, 400, 800, 999999],
        "measure": "approx_tokens",
        "per_bucket_quota": "auto",
        "spillover_reservoir": 200000,
        "tokenizer_name": None,
    },
    "teacher": {
        "backend": "vllm_openai_compatible",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "VLLM_API_KEY",
        "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "request_timeout_s": 120,
        "max_concurrency": 32,
        "retry": {
            "max_attempts": 6,
            "backoff_s": [1, 2, 4, 8, 16, 32],
        },
        "validation": {
            "min_chars": 1,
            "error_substrings": [
                "traceback",
                "internal server error",
                "runtimeerror",
                "exception",
                "chat template",
            ],
        },
        "generation": {
            "max_tokens": 512,
            "top_p": 1.0,
            "greedy_temperature": 0.0,
            "sample_temperature": 1.0,
            "final_temperature": 1.0,
            "seed": None,
        },
    },
    "prefilter": {
        "store_translations": False,
    },
    "selection": {
        "per_bucket": False,
    },
    "final_generation": {
        "num_candidates": 128,
        "store_top_k": 1,
        "strategy": "auto",
        "blob": {
            "enabled": True,
            "blob_ratio": 0.5,
            "blob_max_tokens": 512,
        },
    },
    "metricx": {
        "backend": "official_python",
        "checkpoint": "google/metricx-24-hybrid-xxl-v2p6",
        "tokenizer": "google/mt5-xl",
        "max_input_length": 1536,
        "batch_size": 64,
        "device": "cuda",
        "device_id": None,
        "cache_db": "./runs/exp001/metricx_cache.sqlite",
        "repo_path": "./third_party/metricx",
        "python_bin": "./.metricx-venv/bin/python",
        "prompt_template": "source: {source}\nhypothesis: {hypothesis}\nreference: {reference}\nscore:",
        "max_new_tokens": 8,
    },
    "filters": {
        "rule_based": True,
        "llm_judge": {
            "enabled": True,
            "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "temperature": 0.0,
            "max_tokens": 128,
            "fail_policy": "conservative",
        },
        "length_ratio": {
            "min": 0.25,
            "max": 4.0,
        },
        "min_chars": 1,
        "max_chars": 20000,
    },
    "export": {
        "format": "jsonl",
        "parquet_compression": "zstd",
    },
}


@dataclass
class Config:
    data: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_json(self) -> str:
        return json.dumps(self.data, ensure_ascii=False, indent=2)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    merged = _merge_dicts(DEFAULT_CONFIG, user_cfg)
    return Config(merged)


def resolve_paths(cfg: Config, config_path: str) -> Config:
    base_dir = os.path.dirname(os.path.abspath(config_path))
    resolved = copy.deepcopy(cfg.data)

    def _resolve(p: str) -> str:
        if p is None:
            return p
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))

    resolved["run"]["out_dir"] = _resolve(resolved["run"]["out_dir"])
    if "metricx" in resolved and "cache_db" in resolved["metricx"]:
        resolved["metricx"]["cache_db"] = _resolve(resolved["metricx"]["cache_db"])
    return Config(resolved)


def validate_config(cfg: Config) -> None:
    b = cfg.data["bucketing"]["boundaries"]
    if sorted(b) != b:
        raise ValueError("bucketing.boundaries must be sorted ascending")
    if cfg.data["final_generation"]["num_candidates"] <= 0:
        raise ValueError("final_generation.num_candidates must be > 0")
    if cfg.data["data"]["sample_pool_size"] <= 0:
        raise ValueError("data.sample_pool_size must be > 0")
    ratio = cfg.data["final_generation"]["blob"]["blob_ratio"]
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("final_generation.blob.blob_ratio must be between 0 and 1")
