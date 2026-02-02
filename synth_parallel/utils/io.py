from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Iterator, Optional

import orjson


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]], append: bool = False) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    mode = "ab" if append else "wb"
    with open(path, mode) as f:
        for rec in records:
            f.write(orjson.dumps(rec))
            f.write(b"\n")


def write_jsonl_one(path: str, record: Dict[str, Any], append: bool = True) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    mode = "ab" if append else "wb"
    with open(path, mode) as f:
        f.write(orjson.dumps(record))
        f.write(b"\n")


def maybe_write_parquet(path: str, records: Iterable[Dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required for parquet output") from exc

    table = pa.Table.from_pylist(list(records))
    pq.write_table(table, path)


def read_parquet(path: str) -> Iterable[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required for parquet input") from exc

    table = pq.read_table(path)
    return table.to_pylist()


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
