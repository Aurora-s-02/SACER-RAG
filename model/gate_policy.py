"""Unified public gate selection and confidential decision interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple


CONFIDENTIAL_IMPLEMENTATION_MESSAGE = (
    "The full implementation is available in the confidential reviewer package."
)

GATE_VARIANT_BY_DATASET_MODEL = {
    ("infinitechoice", "qwen"): "loose",
    ("infinitechoice", "llama"): "standard",
    ("infiniteqaloader", "qwen"): "standard",
    ("infiniteqaloader", "llama"): "standard",
    ("novelqa", "qwen"): "loose",
    ("novelqa", "llama"): "standard",
}


@dataclass(frozen=True)
class GatePolicy:
    """Resolved public gate identity for a dataset and model family."""

    canonical: str
    variant: str
    selection_key: str
    selection_reason: str


@dataclass(frozen=True)
class GateDecision:
    """Typed ARG-R gate result returned by the confidential implementation."""

    policy: str
    action: str
    reason: str
    original_graph_fallback_used: bool
    graph_evidence_used: bool
    reliability_score: float | None = None


def _norm(value: Any) -> str:
    """Normalize a public selection key."""
    return str(value or "").strip().casefold()


def _model_family(model_name: Any) -> str:
    """Map supported model identifiers to their public model family."""
    text = _norm(model_name)
    if "qwen" in text:
        return "qwen"
    if "llama" in text:
        return "llama"
    return text


def resolve_gate_policy(dataset_name: Any, model_name: Any) -> GatePolicy:
    """Select the standard or loose gate from dataset and model identity."""
    dataset = _norm(dataset_name)
    model_family = _model_family(model_name)
    key = (dataset, model_family)
    try:
        variant = GATE_VARIANT_BY_DATASET_MODEL[key]
    except KeyError as exc:
        raise ValueError(
            "No public gate selection is defined for "
            f"dataset_name={dataset_name!r}, model_name={model_name!r}"
        ) from exc
    return GatePolicy(
        canonical="gate",
        variant=variant,
        selection_key=f"{dataset}:{model_family}",
        selection_reason="automatic_dataset_model_selection",
    )


def decide_graphless_gate_action(
    *,
    variant: str,
    signals: Mapping[str, Any],
    graph_signal: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> GateDecision:
    """Apply ARG-R to the graphless branch of the full-model pipeline."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)


def build_graphless_preserved_chunks(
    *,
    anchor_candidate_chunk_ids: Iterable[Any],
    current_non_graph_chunk_ids: Iterable[Any],
    config: Mapping[str, Any] | None = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Build the guarded evidence set for a graphless routing decision."""
    raise NotImplementedError(CONFIDENTIAL_IMPLEMENTATION_MESSAGE)
