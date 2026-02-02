from __future__ import annotations

import glob
import os
from typing import Iterable, List


def shard_path(path: str, shard_id: int, num_shards: int) -> str:
    if num_shards <= 1:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.shard{shard_id:02d}{ext}"


def existing_shards(path: str) -> List[str]:
    root, ext = os.path.splitext(path)
    pattern = f"{root}.shard*{ext}"
    return sorted(glob.glob(pattern))


def resolve_inputs(path: str) -> List[str]:
    if os.path.exists(path):
        return [path]
    shards = existing_shards(path)
    if shards:
        return shards
    raise FileNotFoundError(path)
