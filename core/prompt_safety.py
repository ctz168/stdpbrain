"""Prompt safety helpers for coupling internal thought with final response.

Design goal:
- Keep "thinking guides response" behavior.
- Avoid leaking raw noisy monologue into final user-facing prompt.
"""

from __future__ import annotations

import re


_LOOP_BREAKER_MARKERS = (
    "忽略上述重复",
    "跳过无效循环",
    "重置语义空间",
    "<|im_start|>",
    "<|im_end|>",
    "[潜意识]",
    "[内心独白]",
)


def summarize_internal_thought(monologue: str, max_len: int = 60) -> str:
    """Return a safe thought summary for prompt guidance.

    Returns empty string when the monologue appears noisy/unsafe.
    """
    if not monologue:
        return ""

    text = re.sub(r"\s+", " ", monologue).strip()
    if len(text) < 4:
        return ""

    if any(marker in text for marker in _LOOP_BREAKER_MARKERS):
        return ""

    # If there are too many non-language symbols, treat as noisy output.
    symbol_count = len(re.findall(r"[^\u4e00-\u9fa5A-Za-z0-9\s，。！？,.:;、]", text))
    if symbol_count > max(4, len(text) // 8):
        return ""

    # Keep only the first semantic sentence/chunk.
    parts = re.split(r"(?<=[。！？!?])", text)
    summary = parts[0].strip() if parts else text
    if not summary:
        summary = text

    # Drop fragments that are mostly latin gibberish with few spaces.
    latin_words = re.findall(r"[A-Za-z]{4,}", summary)
    if len(latin_words) >= 4 and summary.count(" ") < 3:
        return ""

    return summary[:max_len].strip()


def build_guided_user_input(user_input: str, thought_summary: str) -> str:
    """Build user input with a *structured* thought hint.

    If summary is empty, returns user input unchanged.
    """
    if not thought_summary:
        return user_input

    return (
        f"[内部思路摘要]\n{thought_summary}\n\n"
        f"[任务]\n请基于用户问题给出准确、直接的回答。\n"
        f"[用户问题]\n{user_input}"
    )
