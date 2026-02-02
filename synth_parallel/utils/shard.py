from __future__ import annotations

import hashlib
from typing import Optional


def shard_for_key(key: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % num_shards


def in_shard(key: str, shard_id: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    return shard_for_key(key, num_shards) == shard_id
