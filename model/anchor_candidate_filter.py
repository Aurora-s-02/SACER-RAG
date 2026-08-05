"""Public AFER candidate-filtering interface."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


CONFIDENTIAL_IMPLEMENTATION_MESSAGE = (
    "The full implementation is available in the confidential reviewer package."
)


def filter_anchor_candidates(
    selected_anchor_ids: Iterable[Any],
    query_entities: Iterable[Any],
    anchor_index: Dict[str, Any],
    max_candidates: int = 20,
    min_candidates: int = 2,
    runtime_view: Any = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Filter contextual chunk candidates from selected summary anchors."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)
