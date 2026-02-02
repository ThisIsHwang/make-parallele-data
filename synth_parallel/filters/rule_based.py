from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from synth_parallel.utils.text import has_forbidden_markers, word_overlap_ratio


@dataclass
class RuleBasedConfig:
    min_chars: int
    max_chars: int
    min_ratio: float
    max_ratio: float
    overlap_threshold: float


class RuleBasedFilter:
    def __init__(self, cfg: Dict[str, Any]):
        filt_cfg = cfg["filters"]
        ratio_cfg = filt_cfg.get("length_ratio", {"min": 0.25, "max": 4.0})
        self.config = RuleBasedConfig(
            min_chars=filt_cfg.get("min_chars", 1),
            max_chars=filt_cfg.get("max_chars", 20000),
            min_ratio=ratio_cfg.get("min", 0.25),
            max_ratio=ratio_cfg.get("max", 4.0),
            overlap_threshold=filt_cfg.get("overlap_threshold", 0.9),
        )

    def check(self, source: str, target: str) -> Tuple[bool, str]:
        if has_forbidden_markers(target):
            return False, "forbidden_marker"
        if len(target) < self.config.min_chars:
            return False, "too_short"
        if len(target) > self.config.max_chars:
            return False, "too_long"
        ratio = len(target) / max(1, len(source))
        if ratio < self.config.min_ratio:
            return False, "length_ratio_small"
        if ratio > self.config.max_ratio:
            return False, "length_ratio_large"
        if word_overlap_ratio(source, target) > self.config.overlap_threshold:
            return False, "high_overlap"
        return True, "pass"
