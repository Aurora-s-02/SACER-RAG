"""Public ARG-R reliability-estimation and routing interfaces."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


CONFIDENTIAL_IMPLEMENTATION_MESSAGE = (
    "The full implementation is available in the confidential reviewer package."
)


def compute_anchor_reliability_signals(
    selected_anchors: Iterable[Any],
    candidate_chunk_ids: Iterable[Any],
    original_chunk_ids: Iterable[Any],
    query_entities: Iterable[Any],
    anchor_index: Dict[str, Any],
    original_route: Any,
    original_subroute: Any,
    filter_debug: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute ARG-R reliability signals for contextual anchor evidence."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def decide_anchor_evidence_policy(
    signals: Dict[str, Any], config: Dict[str, Any]
) -> Tuple[str, str, float]:
    """Apply the standard ARG-R gate selected by the unified public policy."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def decide_anchor_evidence_policy_loose(
    signals: Dict[str, Any], config: Dict[str, Any]
) -> Tuple[str, str, float]:
    """Apply the loose ARG-R gate selected by the unified public policy."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def decide_arg_r_confidence(
    signals: Dict[str, Any], config: Dict[str, Any], *, variant: str
) -> Tuple[str, str, str, float]:
    """Return the ARG-R confidence band and hierarchical routing decision."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def build_conservative_preserved_chunks(
    anchor_candidate_chunk_ids: Iterable[Any],
    original_chunk_ids: Iterable[Any],
    signals: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Build the guarded medium-confidence evidence set used by CACR."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)
