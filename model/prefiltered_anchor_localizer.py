"""Public CACR prefiltered anchor-localization interfaces."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


CONFIDENTIAL_IMPLEMENTATION_MESSAGE = (
    "The full implementation is available in the confidential reviewer package."
)


def filter_anchor_index_for_candidate_anchors(
    anchor_index: Dict[str, Any], candidate_anchor_ids: Iterable[Any]
) -> Dict[str, Any]:
    """Construct the candidate-only anchor view used by CACR."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def localize_summary_anchors_prefiltered(
    *,
    question_stem: str,
    query_entities: Iterable[Any],
    anchor_index: Dict[str, Any],
    candidate_anchor_ids: Iterable[str],
    top_k: int = 3,
    method: str = "lexical_entity_overlap",
    config: Optional[Dict[str, Any]] = None,
    runtime_view: Any = None,
) -> Dict[str, Any]:
    """Localize anchors within a CACR-controlled candidate set."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)
