"""Public interfaces for persistent SACER-RAG index construction.

The public preprint release keeps the index API and configuration fingerprint
helpers. The confidential reviewer package contains the SACI construction
algorithm used to create entity-anchor-chunk mappings.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List


SCHEMA_VERSION = "sacer.summary_anchor_index.v1"
BUILDER_VERSION = "sacer-index-builder-public"
CONFIDENTIAL_IMPLEMENTATION_MESSAGE = (
    "The full implementation is available in the confidential reviewer package."
)


def index_config_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the non-secret configuration fields that identify an index."""
    retriever = config.get("retriever", {}).get("kwargs", {})
    llm = config.get("llm", {})
    return {
        "indexing": config.get("indexing", {}),
        "extractor": config.get("extractor", {}),
        "summary_model": config.get("indexing", {}).get("summary_model")
        or llm.get("llm_path"),
        "dense_embedder": retriever.get("embedder"),
        "dense_device": retriever.get("device"),
    }


def index_config_fingerprint(config: Dict[str, Any]) -> str:
    """Return a stable SHA256 fingerprint for index compatibility checks."""
    encoded = json.dumps(
        index_config_payload(config), ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SummaryBackend:
    """Public construction interface for hierarchical summary backends."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Store the public configuration without loading model resources."""
        self.config = dict(config or {})

    def split(self, text: str, size: int, overlap: int) -> Dict[str, Dict[str, Any]]:
        """Split source text for SACI construction in the confidential package."""
        raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)

    def summarize(
        self,
        children: Iterable[str],
        tree: Dict[str, Dict[str, Any]],
        max_words: int,
        *,
        first_level: bool,
        overlap: int = 0,
    ) -> str:
        """Summarize a hierarchy level in the confidential package."""
        raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def build_anchor_payload(
    tree: Dict[str, Any],
    entity_index: Dict[str, List[str]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the SACI entity-anchor-chunk payload.

    The signature and output type are public; the mapping and validation
    algorithm are included only in the confidential reviewer package.
    """
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def build_indexes(
    config: Dict[str, Any], dataset_name: str | None = None
) -> Dict[str, Any]:
    """Build and persist the complete SACI index for a configured dataset."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)
