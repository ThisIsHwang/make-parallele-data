import types

import pytest

from synth_parallel.teacher.client import AsyncTeacherClient, GenerationParams


class DummyCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls < 2:
            raise RuntimeError("transient")
        msg = types.SimpleNamespace(content="ok")
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])


class DummyChat:
    def __init__(self):
        self.completions = DummyCompletions()


class DummyOpenAI:
    def __init__(self, **kwargs):
        self.chat = DummyChat()


@pytest.mark.asyncio
async def test_teacher_retry(monkeypatch):
    monkeypatch.setattr("synth_parallel.teacher.client.AsyncOpenAI", DummyOpenAI)

    cfg = {
        "teacher": {
            "base_url": "http://localhost:8000/v1",
            "model": "dummy",
            "api_key_env": "DUMMY_KEY",
            "request_timeout_s": 1,
            "max_concurrency": 2,
            "retry": {"max_attempts": 2, "backoff_s": [0]},
            "generation": {
                "max_tokens": 8,
                "top_p": 1.0,
                "greedy_temperature": 0.0,
                "sample_temperature": 1.0,
                "final_temperature": 1.0,
                "seed": None,
            },
        }
    }
    client = AsyncTeacherClient(cfg)
    params = GenerationParams(temperature=0.0, top_p=1.0, max_tokens=8)
    output = await client.generate([{"role": "user", "content": "hi"}], params)
    assert output == ["ok"]
