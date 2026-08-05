"""Public AFER anchor-localization interfaces."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


CONFIDENTIAL_IMPLEMENTATION_MESSAGE = (
    "The full implementation is available in the confidential reviewer package."
)
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: Any) -> List[str]:
    """Tokenize text for non-confidential runtime data preparation."""
    return [token.strip("'") for token in TOKEN_RE.findall(str(text or "").casefold())]


def localize_summary_anchors(
    question_stem: str,
    query_entities: Iterable[Any],
    anchor_index: Dict[str, Any],
    top_k: int = 3,
    method: str = "lexical_entity_overlap",
    runtime_view: Any = None,
    candidate_anchor_ids: Optional[Iterable[Any]] = None,
) -> List[Dict[str, Any]]:
    """Localize summary anchors for anchor-first evidence routing.

    Inputs, output types, and configuration names remain public. The complete
    ranking implementation is provided in the confidential reviewer package.
    """
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)
