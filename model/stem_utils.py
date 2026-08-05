from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Tuple


CHOICE_LABELS = ("A", "B", "C", "D")
OPTION_LINE_RE = re.compile(r"^\s*([A-D])[\.\)]\s*(.*?)\s*$")


def sha256_text(text: Any) -> str:
    """Return a stable SHA256 hash for text-like values."""
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def split_choice_question(question: str) -> Tuple[str, List[Dict[str, str]]]:
    """Split a formatted multiple-choice question into stem and options."""
    lines = str(question or "").splitlines()
    stem_lines: List[str] = []
    options: List[Dict[str, str]] = []
    in_options = False

    for line in lines:
        match = OPTION_LINE_RE.match(line)
        if match:
            in_options = True
            options.append({"label": match.group(1), "text": match.group(2).strip()})
        elif in_options and line.strip() and options:
            # Preserve wrapped option text if a source file ever contains it.
            options[-1]["text"] = normalize_space(options[-1]["text"] + " " + line)
        elif not in_options:
            stem_lines.append(line)

    stem = "\n".join(stem_lines).strip()
    if not stem and lines:
        stem = lines[0].strip()
    return stem, options


def format_choice_question(stem: str, options: List[Any]) -> str:
    """Match InfiniteChoiceLoader's formatted question shape."""
    formatted = str(stem or "").rstrip() + "\n"
    for index, option in enumerate(options or []):
        label = CHOICE_LABELS[index] if index < len(CHOICE_LABELS) else str(index)
        formatted += f"{label}. {option}\n"
    return formatted


def gold_label_from_options(answer: Any, options: List[Any]) -> str:
    """Convert InfiniteChoice raw answer/options into A-D where possible."""
    if isinstance(answer, list):
        answer_value = answer[0] if answer else ""
    else:
        answer_value = answer

    answer_text = str(answer_value or "").strip()
    if answer_text.upper() in CHOICE_LABELS and len(answer_text) == 1:
        return answer_text.upper()

    for index, option in enumerate(options or []):
        if str(option).strip() == answer_text and index < len(CHOICE_LABELS):
            return CHOICE_LABELS[index]
    return answer_text

