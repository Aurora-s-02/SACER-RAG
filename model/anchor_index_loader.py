from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


REQUIRED_TOP_LEVEL_FIELDS = [
    "metadata",
    "chunk_to_anchors",
    "anchor_to_chunks",
    "entity_to_anchor_to_chunks",
    "anchor_to_entities",
    "chunk_to_entities",
]


class LegacyCacheError(RuntimeError):
    """Raised when a cache predates the persisted anchor-link index."""


def get_anchor_index_path(cache_folder: Path, method: Optional[str] = None) -> Path:
    cache_folder = Path(cache_folder)
    if method:
        return cache_folder / f"summary_anchor_index_{method}.json"
    return cache_folder / "summary_anchor_index.json"


def validate_summary_anchor_index(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "missing_fields": REQUIRED_TOP_LEVEL_FIELDS[:],
            "reason": "payload_not_dict",
            "metadata": {},
            "num_anchors": 0,
            "num_leaf_chunks": 0,
        }

    missing_fields = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in payload]
    type_errors = []
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field in payload and not isinstance(payload[field], dict):
            type_errors.append(f"{field}_not_dict")

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    anchor_to_chunks = payload.get("anchor_to_chunks") if isinstance(payload.get("anchor_to_chunks"), dict) else {}
    chunk_to_anchors = payload.get("chunk_to_anchors") if isinstance(payload.get("chunk_to_anchors"), dict) else {}

    return {
        "valid": not missing_fields and not type_errors,
        "missing_fields": missing_fields,
        "type_errors": type_errors,
        "reason": ";".join(missing_fields + type_errors) if missing_fields or type_errors else None,
        "metadata": metadata,
        "num_anchors": len(anchor_to_chunks),
        "num_leaf_chunks": len(chunk_to_anchors),
        "schema_version": metadata.get("schema_version"),
    }


def load_summary_anchor_index(cache_folder: Path, method: Optional[str] = None) -> Dict[str, Any]:
    index_path = get_anchor_index_path(cache_folder, method=method)
    result: Dict[str, Any] = {
        "path": str(index_path.resolve()),
        "loaded": False,
        "valid": False,
        "payload": None,
        "validation": {
            "valid": False,
            "missing_fields": REQUIRED_TOP_LEVEL_FIELDS[:],
            "reason": "not_loaded",
        },
        "error": None,
    }

    if not index_path.exists():
        result["error"] = "missing_summary_anchor_index"
        result["validation"]["reason"] = "missing_summary_anchor_index"
        return result

    try:
        with index_path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
    except Exception as exc:
        result["error"] = f"load_error:{exc}"
        result["validation"]["reason"] = result["error"]
        return result

    validation = validate_summary_anchor_index(payload)
    result.update(
        {
            "loaded": True,
            "valid": bool(validation.get("valid")),
            "payload": payload,
            "validation": validation,
        }
    )
    return result


def load_required_summary_anchor_index(cache_folder: Path, method: Optional[str] = None) -> Dict[str, Any]:
    result = load_summary_anchor_index(cache_folder, method=method)
    if not result.get("loaded"):
        raise LegacyCacheError(
            "Cache is missing the persisted summary-anchor index. "
            f"Expected: {result['path']}. Rebuild it with `python main.py build-index`; "
            "SACER-RAG will not silently fall back to a different retrieval flow."
        )
    if not result.get("valid"):
        reason = (result.get("validation") or {}).get("reason") or result.get("error") or "unknown"
        raise LegacyCacheError(
            "Persisted summary-anchor index is invalid "
            f"({reason}) at {result['path']}. Rebuild the cache; fallback is disabled."
        )
    return result
