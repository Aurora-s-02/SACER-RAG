from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from .stem_utils import CHOICE_LABELS, normalize_space


def _norm(text: str) -> str:
    return normalize_space(text).casefold()


def audit_option_leakage(
    retrieval_input_text: str,
    options: Iterable[Dict[str, str]],
) -> Tuple[bool, str]:
    """Conservatively audit whether options appear in retrieval input."""
    text = retrieval_input_text or ""
    normalized_input = _norm(text)
    reasons: List[str] = []

    labels_found: List[str] = []
    for label in CHOICE_LABELS:
        if re.search(rf"(?m)^\s*{re.escape(label)}[\.\)]\s+\S+", text):
            labels_found.append(label)
    if labels_found:
        reasons.append("option_label_lines_present:" + ",".join(labels_found))

    option_text_found: List[str] = []
    for option in options or []:
        label = str(option.get("label", "")).strip()
        option_text = _norm(str(option.get("text", "")))
        # Very short option strings are too noisy for substring detection.
        if len(re.sub(r"\W+", "", option_text)) < 4:
            continue
        if option_text and option_text in normalized_input:
            option_text_found.append(label or "?")
    if option_text_found:
        reasons.append("option_text_present:" + ",".join(option_text_found))

    if reasons:
        return True, "; ".join(reasons)
    return False, "no_option_label_or_option_text_detected"

