from __future__ import annotations

from typing import Dict

SYSTEM_TRANSLATOR = "You are a professional translator."

TRANSLATION_USER_TEMPLATE = (
    "Translate the following text from {source_lang} ({src_lang_code}) to {target_lang} ({tgt_lang_code}).\n"
    "- Preserve meaning, tone, and formatting.\n"
    "- Output only the translation, with no explanations.\n\n"
    "Text:\n{text}"
)

JUDGE_SYSTEM = "You are a strict translation quality judge."

JUDGE_USER_TEMPLATE = (
    "Evaluate whether the candidate translation is acceptable.\n"
    "Return ONLY a JSON object with keys: pass (boolean), reason_code (string), notes (string).\n\n"
    "Source language: {source_lang}\n"
    "Target language: {target_lang}\n"
    "Source text:\n{source_text}\n\n"
    "Candidate translation:\n{candidate_text}"
)


def build_translation_messages(meta: Dict[str, str]) -> list[dict]:
    user = TRANSLATION_USER_TEMPLATE.format(**meta)
    return [
        {"role": "system", "content": SYSTEM_TRANSLATOR},
        {"role": "user", "content": user},
    ]


def build_judge_messages(meta: Dict[str, str]) -> list[dict]:
    user = JUDGE_USER_TEMPLATE.format(**meta)
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
