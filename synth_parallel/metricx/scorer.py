from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from synth_parallel.utils.hash import stable_hash


class CacheDB:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL)"
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[float]:
        cur = self.conn.cursor()
        cur.execute("SELECT score FROM scores WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_many(self, items: Iterable[Tuple[str, float]]) -> None:
        cur = self.conn.cursor()
        cur.executemany("INSERT OR REPLACE INTO scores(key, score) VALUES (?, ?)", items)
        self.conn.commit()


@dataclass
class MetricXConfig:
    backend: str
    checkpoint: str
    tokenizer: Optional[str]
    max_input_length: Optional[int]
    batch_size: int
    device: str
    device_id: Optional[int]
    cache_db: str
    repo_path: Optional[str]
    python_bin: Optional[str]
    prompt_template: str
    max_new_tokens: int


class MetricXScorer:
    def __init__(self, cfg: Dict[str, Any]):
        mcfg = cfg["metricx"]
        self.config = MetricXConfig(
            backend=mcfg.get("backend", "hf"),
            checkpoint=mcfg["checkpoint"],
            tokenizer=mcfg.get("tokenizer"),
            max_input_length=mcfg.get("max_input_length"),
            batch_size=mcfg["batch_size"],
            device=mcfg["device"],
            device_id=mcfg.get("device_id"),
            cache_db=mcfg["cache_db"],
            repo_path=mcfg.get("repo_path"),
            python_bin=mcfg.get("python_bin"),
            prompt_template=mcfg["prompt_template"],
            max_new_tokens=mcfg.get("max_new_tokens", 8),
        )
        self.cache = CacheDB(self.config.cache_db)

        if self.config.backend == "dummy":
            self._backend = DummyMetricXBackend()
        elif self.config.backend == "hf":
            self._backend = HFMetricXBackend(self.config)
        elif self.config.backend == "official_cli":
            self._backend = OfficialMetricXCLIBackend(self.config)
        elif self.config.backend == "official_python":
            self._backend = OfficialMetricXPythonBackend(self.config)
        else:
            raise ValueError(f"Unknown metricx backend: {self.config.backend}")

    def score_batch(self, sources: List[str], hyps: List[str]) -> List[float]:
        if len(sources) != len(hyps):
            raise ValueError("sources and hyps must have same length")

        scores: List[Optional[float]] = [None] * len(sources)
        missing: List[Tuple[int, str]] = []
        for i, (src, hyp) in enumerate(zip(sources, hyps)):
            key = stable_hash(f"{src}||{hyp}")
            cached = self.cache.get(key)
            if cached is None:
                missing.append((i, key))
            else:
                scores[i] = cached

        if missing:
            batch_indices = [idx for idx, _ in missing]
            batch_sources = [sources[idx] for idx in batch_indices]
            batch_hyps = [hyps[idx] for idx in batch_indices]
            batch_scores = self._backend.score_batch(batch_sources, batch_hyps)
            if len(batch_scores) != len(batch_indices):
                raise RuntimeError("MetricX backend returned unexpected batch size")
            self.cache.set_many(
                [(missing[i][1], float(batch_scores[i])) for i in range(len(missing))]
            )
            for i, score in zip(batch_indices, batch_scores):
                scores[i] = float(score)

        return [float(s) for s in scores]  # type: ignore


class DummyMetricXBackend:
    def score_batch(self, sources: List[str], hyps: List[str]) -> List[float]:
        scores = []
        for src, hyp in zip(sources, hyps):
            # simple heuristic: lower is better
            diff = abs(len(src) - len(hyp))
            scores.append(float(diff))
        return scores


class HFMetricXBackend:
    def __init__(self, cfg: MetricXConfig):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install metricx extra: pip install .[metricx]") from exc

        self.torch = torch
        self.device = cfg.device
        if cfg.device == "cuda" and cfg.device_id is not None:
            device = f"cuda:{cfg.device_id}"
        else:
            device = cfg.device

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.checkpoint)
        self.model.to(device)
        self.model.eval()
        self.prompt_template = cfg.prompt_template
        self.max_new_tokens = cfg.max_new_tokens

    def _render_prompt(self, source: str, hyp: str) -> str:
        return self.prompt_template.format(source=source, hypothesis=hyp, reference="")

    def _parse_score(self, text: str) -> float:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            raise RuntimeError(f"Failed to parse score from: {text}")
        return float(match.group(0))

    def score_batch(self, sources: List[str], hyps: List[str]) -> List[float]:
        prompts = [self._render_prompt(src, hyp) for src, hyp in zip(sources, hyps)]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [self._parse_score(text) for text in decoded]


class OfficialMetricXCLIBackend:
    def __init__(self, cfg: MetricXConfig):
        self.checkpoint = cfg.checkpoint
        self.tokenizer = cfg.tokenizer or "google/mt5-xl"
        self.max_input_length = cfg.max_input_length or 1536
        self.batch_size = cfg.batch_size
        self.repo_path = cfg.repo_path or "./third_party/metricx"
        self.python_bin = cfg.python_bin or "python"

    def score_batch(self, sources: List[str], hyps: List[str]) -> List[float]:
        import json
        import os
        import shutil
        import subprocess
        import tempfile

        if not os.path.isdir(self.repo_path):
            raise RuntimeError(
                f"metricx repo not found at {self.repo_path}. "
                "Run scripts/install_metricx_official.sh"
            )
        if not (os.path.exists(self.python_bin) or shutil.which(self.python_bin)):
            raise RuntimeError(
                f"metricx python not found: {self.python_bin}. "
                "Check metricx.python_bin in config."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            output_path = os.path.join(tmpdir, "output.jsonl")

            with open(input_path, "w", encoding="utf-8") as f:
                for src, hyp in zip(sources, hyps):
                    f.write(
                        json.dumps(
                            {
                                "source": src,
                                "hypothesis": hyp,
                                "reference": "",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            env = os.environ.copy()
            env["PYTHONPATH"] = (
                f"{os.path.abspath(self.repo_path)}"
                + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            )

            cmd = [
                self.python_bin,
                "-m",
                "metricx24.predict",
                "--tokenizer",
                self.tokenizer,
                "--model_name_or_path",
                self.checkpoint,
                "--max_input_length",
                str(self.max_input_length),
                "--batch_size",
                str(self.batch_size),
                "--input_file",
                input_path,
                "--output_file",
                output_path,
                "--qe",
            ]

            subprocess.run(cmd, check=True, cwd=self.repo_path, env=env)

            scores: List[float] = []
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    scores.append(float(obj["prediction"]))

        if len(scores) != len(sources):
            raise RuntimeError("MetricX official CLI returned unexpected output size")
        return scores


class OfficialMetricXPythonBackend:
    def __init__(self, cfg: MetricXConfig):
        import sys
        import os

        self.checkpoint = cfg.checkpoint
        self.tokenizer_name = cfg.tokenizer or "google/mt5-xl"
        self.max_input_length = cfg.max_input_length or 1536
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.device_id = cfg.device_id
        self.repo_path = cfg.repo_path or "./third_party/metricx"

        if not os.path.isdir(self.repo_path):
            raise RuntimeError(
                f"metricx repo not found at {self.repo_path}. "
                "Run scripts/install_metricx_official.sh"
            )

        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)

        try:
            import torch
            import transformers
            from metricx24 import models
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "metricx24 import failed. Run scripts/install_metricx_official.sh"
            ) from exc

        if self.device == "cuda" and self.device_id is not None:
            device = f"cuda:{self.device_id}"
        else:
            device = self.device
        self.torch = torch
        self.device_obj = torch.device(device)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = models.MT5ForRegression.from_pretrained(
            self.checkpoint, torch_dtype="auto"
        )
        self.model.to(self.device_obj)
        self.model.eval()

    def _make_inputs(self, sources: List[str], hyps: List[str]):
        texts = [
            f"source: {src} candidate: {hyp}" for src, hyp in zip(sources, hyps)
        ]
        enc = self.tokenizer(
            texts,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
        )
        input_ids = []
        attention_masks = []
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
            if ids:
                input_ids.append(ids[:-1])
                attention_masks.append(mask[:-1])
            else:
                input_ids.append(ids)
                attention_masks.append(mask)
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding=True,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device_obj) for k, v in batch.items()}
        return batch

    def score_batch(self, sources: List[str], hyps: List[str]) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(sources), self.batch_size):
            batch_sources = sources[i : i + self.batch_size]
            batch_hyps = hyps[i : i + self.batch_size]
            inputs = self._make_inputs(batch_sources, batch_hyps)
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            preds = outputs.predictions.detach().cpu().tolist()
            scores.extend([float(p) for p in preds])
        return scores
