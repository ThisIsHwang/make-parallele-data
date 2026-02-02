from __future__ import annotations

import hashlib
from typing import Any


def stable_hash(text: str, length: int = 16) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return digest[:length]


def hash_record(*parts: Any, length: int = 16) -> str:
    joined = "||".join(str(p) for p in parts if p is not None)
    return stable_hash(joined, length=length)
