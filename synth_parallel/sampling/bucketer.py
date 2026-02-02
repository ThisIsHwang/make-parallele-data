from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from synth_parallel.utils.text import approx_token_len


@dataclass
class BucketResult:
    records: List[Dict[str, Any]]
    bucket_id: int


class LengthMeasurer:
    def __init__(self, measure: str, tokenizer_name: Optional[str] = None):
        self.measure = measure
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None

    def _load_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # optional dependency

            if not self.tokenizer_name:
                raise ValueError("tokenizer_name is required for tokenizer measure")
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

    def __call__(self, text: str) -> int:
        if self.measure == "approx_tokens":
            return approx_token_len(text)
        if self.measure == "tokenizer":
            self._load_tokenizer()
            return len(self._tokenizer.encode(text))
        raise ValueError(f"Unknown measure: {self.measure}")


def find_bucket(boundaries: List[int], length: int) -> int:
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= length < boundaries[i + 1]:
            return i
    return len(boundaries) - 2


class Reservoir:
    def __init__(self, capacity: int, seed: int = 1234):
        self.capacity = capacity
        self.data: List[Dict[str, Any]] = []
        self.count = 0
        random.seed(seed)

    def add(self, item: Dict[str, Any]) -> None:
        self.count += 1
        if len(self.data) < self.capacity:
            self.data.append(item)
            return
        j = random.randint(0, self.count - 1)
        if j < self.capacity:
            self.data[j] = item


class BucketSampler:
    def __init__(
        self,
        boundaries: List[int],
        total_target: int,
        per_bucket_quota: Optional[int],
        spillover_capacity: int,
        seed: int = 1234,
    ):
        self.boundaries = boundaries
        self.num_buckets = len(boundaries) - 1
        if per_bucket_quota is None:
            per_bucket_quota = math.ceil(total_target / self.num_buckets)
        self.per_bucket_quota = per_bucket_quota
        self.total_target = total_target
        self.buckets = [Reservoir(per_bucket_quota, seed=seed) for _ in range(self.num_buckets)]
        self.spillover = Reservoir(spillover_capacity, seed=seed)

    def add(self, bucket_id: int, record: Dict[str, Any]) -> None:
        if bucket_id < 0 or bucket_id >= self.num_buckets:
            return
        bucket = self.buckets[bucket_id]
        if len(bucket.data) < bucket.capacity:
            bucket.add(record)
        else:
            self.spillover.add(record)

    def finalize(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for bucket in self.buckets:
            records.extend(bucket.data)
        if len(records) >= self.total_target:
            return records[: self.total_target]

        needed = self.total_target - len(records)
        if not self.spillover.data or needed <= 0:
            return records

        existing_ids = {rec.get("source_id") for rec in records}
        for rec in self.spillover.data:
            if rec.get("source_id") in existing_ids:
                continue
            records.append(rec)
            if len(records) >= self.total_target:
                break
        return records
