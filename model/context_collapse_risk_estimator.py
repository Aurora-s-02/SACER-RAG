"""Public CACR context-collapse risk interfaces."""

from __future__ import annotations

from typing import Any, Dict, Iterable


CONFIDENTIAL_IMPLEMENTATION_MESSAGE = (
    "The full implementation is available in the confidential reviewer package."
)


def decide_phase4_route_from_risk(
    signals: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Choose the CACR route from context-collapse risk signals."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def compute_context_collapse_risk_signals(
    *,
    question_stem: str,
    query_entities: Iterable[Any],
    original_chunk_ids: Iterable[Any],
    original_route: str,
    original_subroute: str,
    anchor_index: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the typed CACR risk signals used by controlled replacement."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def estimate_context_collapse_risk(**kwargs: Any) -> Dict[str, Any]:
    """Estimate context-collapse risk through the public compatibility entry."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)
