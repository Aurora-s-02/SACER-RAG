"""Public AECF configuration, result types, and algorithm interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


CONFIDENTIAL_IMPLEMENTATION_MESSAGE = (
    "The full implementation is available in the confidential reviewer package."
)


@dataclass(frozen=True)
class AecfScoreWeights:
    """Names of the AECF scoring components without confidential values."""

    query_chunk_sim: float = None  # type: ignore[assignment]
    query_anchor_sim: float = None  # type: ignore[assignment]
    entity_coverage: float = None  # type: ignore[assignment]
    anchor_consistency: float = None  # type: ignore[assignment]
    local_coherence: float = None  # type: ignore[assignment]
    redundancy_penalty: float = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AecfConfig:
    """Public AECF configuration schema with confidential defaults omitted."""

    target_routes: str = None  # type: ignore[assignment]
    anchor_top_k: int = None  # type: ignore[assignment]
    chunks_per_anchor: int = None  # type: ignore[assignment]
    max_final_chunks: int = None  # type: ignore[assignment]
    min_entity_coverage: float = None  # type: ignore[assignment]
    min_anchor_margin: float = None  # type: ignore[assignment]
    preserve_on_uncertain: bool = None  # type: ignore[assignment]
    debug: bool = False
    score_mode: str = None  # type: ignore[assignment]
    min_final_chunks: int = None  # type: ignore[assignment]
    localization_method: str = None  # type: ignore[assignment]
    enable_early_anchor_guard: bool = None  # type: ignore[assignment]
    early_guard_use_anchor_margin: bool = None  # type: ignore[assignment]
    early_guard_min_anchor_margin: Optional[float] = None
    anchor_source: str = None  # type: ignore[assignment]
    candidate_collect_mode: str = None  # type: ignore[assignment]
    cache_tokenization: bool = None  # type: ignore[assignment]
    enable_redundancy_penalty: bool = None  # type: ignore[assignment]
    weights: AecfScoreWeights = field(default_factory=AecfScoreWeights)


@dataclass(frozen=True)
class AecfResult:
    """Typed AECF output returned by the confidential implementation."""

    triggered: bool
    replacement_used: bool
    final_chunk_ids: List[str]
    diagnostics: Dict[str, Any]


def build_aecf_candidates(
    *,
    question: str,
    query_entities: Iterable[Any],
    original_candidate_chunk_ids: Iterable[Any],
    original_route_name: str,
    anchor_index: Mapping[str, Any],
    chunk_texts: Optional[Mapping[str, Any]] = None,
    config: Optional[Any] = None,
    runtime_view: Any = None,
    candidate_anchors: Optional[Sequence[Mapping[str, Any]]] = None,
    localization_elapsed: float = 0.0,
) -> Dict[str, Any]:
    """Build the selective candidate context used by AECF."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def score_aecf_candidates(
    *,
    question: str,
    query_entities: Iterable[Any],
    candidate_anchors: Sequence[Mapping[str, Any]],
    candidate_chunks: Sequence[Mapping[str, Any]],
    config: Optional[Any] = None,
    timing: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Score AECF candidates using the confidential evidence model."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def apply_aecf_guard(
    *,
    candidate_anchors: Sequence[Mapping[str, Any]],
    scored_candidates: Sequence[Mapping[str, Any]],
    original_candidate_chunk_ids: Iterable[Any],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Apply the AECF preservation guard before contextual replacement."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def run_anchor_evidence_contextual_fallback(
    *,
    question: str,
    query_entities: Iterable[Any],
    original_candidate_chunk_ids: Iterable[Any],
    original_route_name: str,
    anchor_index: Optional[Mapping[str, Any]],
    chunk_texts: Optional[Mapping[str, Any]] = None,
    config: Optional[Any] = None,
    runtime_view: Any = None,
    precomputed_anchor_ids: Optional[Iterable[Any]] = None,
    precomputed_anchor_scores: Optional[Iterable[Any]] = None,
    precomputed_anchor_source: Optional[str] = None,
) -> AecfResult:
    """Run selective AECF reconstruction for low-confidence evidence."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)
