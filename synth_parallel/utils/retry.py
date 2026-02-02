from __future__ import annotations

import asyncio
from typing import Iterable


def get_backoff_schedule(backoff_s: Iterable[float], max_attempts: int) -> list[float]:
    schedule = list(backoff_s)
    if max_attempts <= 1:
        return []
    if len(schedule) >= max_attempts - 1:
        return schedule[: max_attempts - 1]
    while len(schedule) < max_attempts - 1:
        schedule.append(schedule[-1] if schedule else 1.0)
    return schedule


async def sleep_backoff(delay: float) -> None:
    await asyncio.sleep(delay)
