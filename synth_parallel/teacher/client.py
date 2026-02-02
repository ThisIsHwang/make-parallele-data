from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from openai import AsyncOpenAI

from synth_parallel.utils.hash import stable_hash
from synth_parallel.utils.retry import get_backoff_schedule, sleep_backoff


@dataclass
class GenerationParams:
    temperature: float
    top_p: float
    max_tokens: int
    n: int = 1
    seed: Optional[int] = None


class AsyncTeacherClient:
    def __init__(self, cfg: Dict[str, Any]):
        self.base_url = cfg["teacher"]["base_url"]
        self.model = cfg["teacher"]["model"]
        api_key_env = cfg["teacher"]["api_key_env"]
        api_key = os.environ.get(api_key_env, "")
        self.timeout_s = cfg["teacher"]["request_timeout_s"]
        self.max_concurrency = cfg["teacher"]["max_concurrency"]
        self.retry_cfg = cfg["teacher"]["retry"]
        self.validation_cfg = cfg["teacher"].get("validation", {})
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, timeout=self.timeout_s)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    def _idempotency_key(self, payload: Dict[str, Any], request_id: Optional[str]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if request_id:
            raw = f"{raw}||{request_id}"
        return stable_hash(raw, length=24)

    def _is_valid_text(self, text: Optional[str]) -> bool:
        if text is None:
            return False
        stripped = text.strip()
        min_chars = int(self.validation_cfg.get("min_chars", 1))
        if len(stripped) < min_chars:
            return False
        lowered = stripped.lower()
        for token in self.validation_cfg.get("error_substrings", []):
            if token and token.lower() in lowered:
                return False
        return True

    def _validate_outputs(self, outputs: List[str]) -> None:
        if not outputs:
            raise RuntimeError("Empty response from teacher API")
        if any(not self._is_valid_text(out) for out in outputs):
            raise RuntimeError("Invalid response content from teacher API")

    async def _request(self, payload: Dict[str, Any], request_id: Optional[str]) -> List[str]:
        key = self._idempotency_key(payload, request_id)
        backoff_schedule = get_backoff_schedule(
            self.retry_cfg.get("backoff_s", [1, 2, 4]),
            self.retry_cfg.get("max_attempts", 3),
        )
        attempts = 0
        while True:
            try:
                async with self._semaphore:
                    resp = await self.client.chat.completions.create(
                        **payload,
                        extra_headers={"Idempotency-Key": key},
                    )
                outputs = [choice.message.content for choice in resp.choices]
                self._validate_outputs(outputs)
                return outputs
            except Exception as exc:
                msg = str(exc).lower()
                if "chat template" in msg:
                    raise RuntimeError(
                        "vLLM chat template missing. Start vLLM with --chat-template /path/to/template.jinja"
                    ) from exc
                if attempts >= len(backoff_schedule):
                    raise
                await sleep_backoff(backoff_schedule[attempts])
                attempts += 1

    async def generate(
        self,
        messages: List[Dict[str, str]],
        params: GenerationParams,
        request_id: Optional[str] = None,
    ) -> List[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "n": params.n,
        }
        if params.seed is not None:
            payload["seed"] = params.seed
        return await self._request(payload, request_id=request_id)
