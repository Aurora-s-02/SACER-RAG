"""Public CACR entity-anchor routing interface."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


CONFIDENTIAL_IMPLEMENTATION_MESSAGE = (
    "The full implementation is available in the confidential reviewer package."
)


def rank_entity_anchors(
    query_entities: Iterable[Any],
    original_chunk_ids: Iterable[Any],
    anchor_index: Dict[str, Any],
    top_k: int = 3,
    config: Optional[Dict[str, Any]] = None,
    runtime_view: Any = None,
) -> Dict[str, Any]:
    """Rank entity-linked anchors for CACR evidence replacement."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)
