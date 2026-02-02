from __future__ import annotations

import re
from typing import Iterable, List

_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+")


def approx_token_len(text: str) -> int:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(1, len(tokens))


def split_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def split_sentences(text: str) -> List[str]:
    return [seg.strip() for seg in _SENT_SPLIT_RE.split(text) if seg.strip()]


def merge_short(lines: Iterable[str], min_chars: int) -> List[str]:
    merged: List[str] = []
    buf = ""
    for line in lines:
        if not buf:
            buf = line
            continue
        if len(buf) < min_chars:
            buf = buf + " " + line
        else:
            merged.append(buf)
            buf = line
    if buf:
        merged.append(buf)
    return merged


def has_forbidden_markers(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "assistant:",
        "user:",
        "system:",
        "here is the translation",
        "번역:",
        "i will translate",
        "<think>",
        "</think>",
    ]
    return any(m in lowered for m in markers)


def word_overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\w+", a.lower()))
    b_tokens = set(re.findall(r"\w+", b.lower()))
    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)
