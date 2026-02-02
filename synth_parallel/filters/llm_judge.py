from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from synth_parallel.teacher.client import AsyncTeacherClient, GenerationParams
from synth_parallel.teacher.prompts import build_judge_messages


@dataclass
class JudgeConfig:
    enabled: bool
    model: str
    temperature: float
    max_tokens: int
    fail_policy: str


class LLMJudge:
    def __init__(self, cfg: Dict[str, Any], client: AsyncTeacherClient):
        judge_cfg = cfg["filters"]["llm_judge"]
        self.config = JudgeConfig(
            enabled=judge_cfg.get("enabled", True),
            model=judge_cfg.get("model"),
            temperature=judge_cfg.get("temperature", 0.0),
            max_tokens=judge_cfg.get("max_tokens", 128),
            fail_policy=judge_cfg.get("fail_policy", "conservative"),
        )
        self.client = client

    def _parse_json(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object in judge output")
        return json.loads(match.group(0))

    async def check(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        candidate_text: str,
    ) -> Tuple[bool, str]:
        messages = build_judge_messages(
            {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source_text": source_text,
                "candidate_text": candidate_text,
            }
        )
        params = GenerationParams(
            temperature=self.config.temperature,
            top_p=1.0,
            max_tokens=self.config.max_tokens,
            n=1,
        )
        try:
            outputs = await self.client.generate(messages, params)
            parsed = self._parse_json(outputs[0])
            return bool(parsed.get("pass", False)), str(parsed.get("reason_code", "unknown"))
        except Exception as exc:
            if self.config.fail_policy == "permissive":
                return True, "judge_failed_permissive"
            return False, "judge_failed"
