from __future__ import annotations

import itertools
import re
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import os

from datasets import DownloadConfig, load_dataset

from synth_parallel.utils.text import approx_token_len, merge_short, split_lines, split_sentences

_HTML_RE = re.compile(r"<[^>]+>")


def load_madlad(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    hf_endpoint = data_cfg.get("hf_endpoint")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    hf_timeout = data_cfg.get("hf_timeout_s")
    if hf_timeout:
        os.environ["HF_HUB_TIMEOUT"] = str(hf_timeout)
        os.environ["HF_HUB_READ_TIMEOUT"] = str(hf_timeout)
    if data_cfg.get("hf_enable_hf_transfer"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    download_config = None
    if hf_timeout:
        download_config = DownloadConfig(timeout=hf_timeout)

    ds = load_dataset(
        data_cfg["madlad_dataset"],
        data_cfg["src_lang"],
        split=data_cfg["madlad_split"],
        streaming=data_cfg.get("streaming", True),
        download_config=download_config,
    )
    return ds


def _extract_doc_id(record: Dict[str, Any], fallback: int) -> str:
    for key in ("doc_id", "document_id", "document_key", "id", "record_id"):
        if key in record and record[key] is not None:
            return str(record[key])
    return f"doc_{fallback}"


def _extract_text_field(record: Dict[str, Any]) -> Any:
    if "text" in record:
        return record["text"]
    if "content" in record:
        return record["content"]
    return None


def _split_text(text: str, cfg: Dict[str, Any]) -> List[str]:
    lines = split_lines(text)
    split_mode = cfg["segmentation"]["mode"]
    if split_mode in ("auto", "newline"):
        segments: List[str] = []
        for line in lines:
            if len(line) > cfg["segmentation"]["max_chars"]:
                segments.extend(split_sentences(line))
            else:
                segments.append(line)
    else:
        segments = lines

    if cfg["segmentation"].get("merge_short", True):
        segments = merge_short(segments, cfg["segmentation"]["min_chars"])
    return segments


def _filter_segment(text: str, cfg: Dict[str, Any]) -> bool:
    if len(text) < cfg["segmentation"]["min_chars"]:
        return False
    if len(text) > cfg["segmentation"]["max_chars"]:
        return False
    if not cfg["segmentation"].get("allow_html", False) and _HTML_RE.search(text):
        return False
    return True


def segment_text_field(text_field: Any, cfg: Dict[str, Any]) -> List[str]:
    if isinstance(text_field, list):
        segments = [seg.strip() for seg in text_field if seg and seg.strip()]
    else:
        segments = _split_text(str(text_field), cfg)
    return [seg for seg in segments if _filter_segment(seg, cfg)]


def _build_blobs(
    segments: List[str],
    max_tokens: int,
    measure_fn,
) -> List[Tuple[str, Tuple[int, int]]]:
    blobs: List[Tuple[str, Tuple[int, int]]] = []
    buf: List[str] = []
    buf_tokens = 0
    start_idx = 0
    for i, seg in enumerate(segments):
        seg_tokens = measure_fn(seg)
        if buf and buf_tokens + seg_tokens > max_tokens:
            blobs.append((" ".join(buf), (start_idx, i - 1)))
            buf = [seg]
            buf_tokens = seg_tokens
            start_idx = i
        else:
            buf.append(seg)
            buf_tokens += seg_tokens
    if buf:
        blobs.append((" ".join(buf), (start_idx, len(segments) - 1)))
    return blobs


def iter_segments(
    cfg: Dict[str, Any],
    limit: Optional[int] = None,
    include_blob: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    ds = load_madlad(cfg)
    count = 0
    measure_fn = approx_token_len

    for idx, record in enumerate(ds):
        doc_id = _extract_doc_id(record, idx)
        text_field = _extract_text_field(record)
        if text_field is None:
            continue

        filtered_segments = segment_text_field(text_field, cfg)

        for seg_idx, seg in enumerate(filtered_segments):
            yield {
                "doc_id": doc_id,
                "segment_index": seg_idx,
                "segment_type": "sentence",
                "source_text": seg,
            }
            count += 1
            if limit and count >= limit:
                return

        if include_blob and filtered_segments:
            blobs = _build_blobs(
                filtered_segments,
                cfg["final_generation"]["blob"]["blob_max_tokens"],
                measure_fn,
            )
            for blob_text, (start_idx, end_idx) in blobs:
                if not _filter_segment(blob_text, cfg):
                    continue
                yield {
                    "doc_id": doc_id,
                    "segment_index": start_idx,
                    "segment_type": "blob",
                    "segment_span": [start_idx, end_idx],
                    "source_text": blob_text,
                }
                count += 1
                if limit and count >= limit:
                    return
