from __future__ import annotations

from typing import Any, Dict, Iterable


def build_graph_signal(
    *,
    chunk_count: int,
    raw_chunk_ids: Iterable[Any] | None = None,
    raw_entities: Iterable[Any] | None = None,
) -> Dict[str, Any]:
    """Expose only the allowed graph signal.

    Raw graph chunk ids and entities are intentionally accepted only so callers
    cannot accidentally leak them into diagnostics or downstream evidence.
    """
    del raw_chunk_ids, raw_entities
    return {"graph_signal_has_valid_path": bool(max(0, int(chunk_count or 0)) > 0)}
