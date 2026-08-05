from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .anchor_schema import natural_sort_key
from .stem_utils import sha256_text


def normalize_retrieval_route(retrieval_type: Any) -> str:
    text = str(retrieval_type or "").strip()
    if text == "Global Search":
        return "dense_retrieval"
    if text == "Occurrence Rerank":
        return "occurrence_ranking"
    if text == "Evidence Contextual Fallback":
        return "evidence_contextual_fallback"
    if text.startswith("Local"):
        return "graph_local"
    if text.startswith("EntityAware Filter"):
        return "graph_local"
    return "unknown"


def normalize_retrieval_subroute(retrieval_type: Any) -> str:
    text = str(retrieval_type or "").strip()
    if text == "Global Search":
        return "dense_retrieval"
    if text == "Occurrence Rerank":
        return "occurrence_ranking"
    if text == "Evidence Contextual Fallback":
        return "evidence_contextual_fallback"
    if text.startswith("Local"):
        return "local_direct"
    if text.startswith("EntityAware Filter"):
        return "local_entityaware_filter"
    return "unknown"


def flatten_chunk_ids(chunk_ids: Any) -> List[str]:
    flattened: List[str] = []
    seen = set()

    def add_one(value: Any) -> None:
        if value is None:
            return
        text = str(value)
        if text not in seen:
            seen.add(text)
            flattened.append(text)

    if isinstance(chunk_ids, dict):
        for values in chunk_ids.values():
            if isinstance(values, list):
                for value in values:
                    add_one(value)
            else:
                add_one(values)
    elif isinstance(chunk_ids, list):
        for value in chunk_ids:
            add_one(value)
    else:
        add_one(chunk_ids)
    return flattened


def chunk_id_dict_from_list(chunk_ids: Iterable[Any], key: str = "anchor_bounded") -> Dict[str, List[str]]:
    unique = []
    seen = set()
    for chunk_id in chunk_ids:
        text = str(chunk_id)
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return {key: unique}


def count_chunk_ids(chunk_ids: Any) -> int:
    return len(flatten_chunk_ids(chunk_ids))


def safe_json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def parse_mc_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else text


def default_route1_debug(
    *,
    enabled: bool,
    retrieval_input_text: str,
    retrieval_input_mode: Optional[str] = None,
    anchor_localization_text: Optional[str] = None,
    anchor_index_path: Optional[str] = None,
) -> Dict[str, Any]:
    anchor_text = retrieval_input_text if anchor_localization_text is None else anchor_localization_text
    return {
        "route1_enabled": bool(enabled),
        "route1_phase": "phase2_anchor_first_local_retrieval_mvp",
        "retrieval_input_mode": retrieval_input_mode,
        "anchor_localization_text": anchor_text,
        "anchor_localization_hash": sha256_text(anchor_text),
        "anchor_index_path": anchor_index_path,
        "anchor_index_loaded": False,
        "anchor_index_valid": False,
        "anchor_selected": False,
        "selected_anchor_ids": [],
        "selected_anchor_scores": [],
        "anchor_candidate_chunk_ids": [],
        "anchor_candidate_count": 0,
        "anchor_reliability_gate_enabled": False,
        "anchor_evidence_policy_mode": "disabled",
        "anchor_reliability_policy": "anchor_only",
        "anchor_reliability_score": None,
        "anchor_reliability_signals": {},
        "anchor_core_chunk_ids": [],
        "preserved_original_chunk_ids": [],
        "final_policy_chunk_ids": [],
        "preservation_reason": None,
        "original_guard_count": None,
        "final_policy_reason": None,
        "phase3v2_gate_error": None,
        "phase2_structural_fallback_used": False,
        "phase2_structural_fallback_reason": None,
        "anchor_reliability_fallback_used": False,
        "anchor_reliability_fallback_reason": None,
        "anchor_fallback_used": bool(enabled),
        "anchor_fallback_reason": "anchor_retrieval_disabled" if not enabled else None,
        "original_retrieval_type": None,
        "original_retrieval_route": None,
        "original_retrieval_subroute": None,
        "original_chunk_ids": [],
        "final_retrieval_type": None,
        "final_retrieval_route": None,
        "final_retrieval_subroute": None,
        "final_chunk_ids": [],
        "final_evidence_chunk_count": 0,
        "retrieval_input_text": retrieval_input_text,
        "query_time_original": 0.0,
        "query_time_anchor": 0.0,
        "query_time_total": 0.0,
        "query_time_anchor_index_validation": 0.0,
        "query_time_anchor_localization": 0.0,
        "query_time_anchor_candidate_filter": 0.0,
        "query_time_anchor_reliability_gate": 0.0,
        "query_time_anchor_format": 0.0,
        "query_time_anchor_other": 0.0,
        "query_time_sidecar_load": 0.0,
        "query_time_phase4_risk_estimation": 0.0,
        "query_time_phase4_entity_anchor_routing": 0.0,
        "query_time_phase4_entity_anchor_prepare": 0.0,
        "query_time_phase4_entity_anchor_candidate_collect": 0.0,
        "query_time_phase4_entity_anchor_scoring": 0.0,
        "query_time_phase4_entity_anchor_sort": 0.0,
        "query_time_phase4_entity_anchor_details": 0.0,
        "query_time_phase4_prefiltered_anchor_localization": 0.0,
        "query_time_phase4_prefiltered_anchor_candidate_prepare": 0.0,
        "query_time_phase4_prefiltered_anchor_rerank": 0.0,
        "query_time_phase4_prefiltered_anchor_sort": 0.0,
        "query_time_phase4_controlled_replacement_decision": 0.0,
        "query_time_phase4_effective_anchor_localization": 0.0,
        "query_time_anchor_format_chunk_dict_build": 0.0,
        "query_time_anchor_format_retriever_format_res": 0.0,
        "query_time_anchor_format_total": 0.0,
        "final_evidence_hash": None,
        "final_evidence_char_len": 0,
        "evidence_contextual_fallback_triggered": False,
        "evidence_contextual_fallback_valid": False,
        "evidence_contextual_fallback_time": 0.0,
        "evidence_contextual_fallback_exception": "",
        "evidence_contextual_fallback_invalid_reason": "",
        "final_route_before_contextual_fallback": None,
        "final_route_after_contextual_fallback": None,
        "original_occurrence_chunk_ids": [],
        "original_occurrence_chunk_ids_source": "",
        "original_occurrence_chunk_ids_verified": False,
        "contextual_fallback_chunk_ids": [],
        "occurrence_to_contextual_chunk_overlap": None,
        "evidence_top_anchor_ids": [],
        "evidence_selected_anchor_ids": [],
        "evidence_candidate_source_counts": {},
        "phase4_enabled": False,
        "phase4_routing_mode": "disabled",
        "phase4_diagnostics_available": False,
        "phase4_error": None,
        "phase4_risk_score": None,
        "phase4_risk_bucket": None,
        "phase4_risk_flags": [],
        "phase4_risk_signals": {},
        "phase4_proposed_route_decision": None,
        "phase4_proposed_anchor_mode": None,
        "phase4_anchor_invocation_should_skip": False,
        "phase4_entity_anchor_confidence": None,
        "phase4_entity_anchor_candidate_ids": [],
        "phase4_entity_anchor_scores": [],
        "phase4_entity_anchor_score_details": {},
        "phase4_entity_anchor_error": None,
        "phase4_entity_anchor_scoring_mode": None,
        "phase4_entity_anchor_include_score_details": None,
        "phase4_entity_anchor_timing": {},
        "phase4_dense_anchor_recommended": False,
        "phase4_prefiltered_localization_enabled": False,
        "phase4_prefiltered_localization_available": False,
        "phase4_prefiltered_candidate_anchor_ids": [],
        "phase4_prefiltered_candidate_anchor_count": 0,
        "phase4_prefiltered_selected_anchor_ids": [],
        "phase4_prefiltered_selected_anchor_scores": [],
        "phase4_prefiltered_selected_anchors": [],
        "phase4_prefiltered_error": None,
        "phase4_prefiltered_top1_equal": None,
        "phase4_prefiltered_topk_overlap_count": None,
        "phase4_prefiltered_topk_jaccard": None,
        "phase4_prefiltered_selected_contains_dense_top1": None,
        "phase4_dense_selected_in_prefilter_candidates": None,
        "phase4_dense_selected_all_in_prefilter_candidates": None,
        "phase4_prefiltered_top3_set_recall": None,
        "phase4_prefiltered_top3_order_exact_match": None,
        "phase4_prefiltered_top3_set_exact_match": None,
        "phase4_controlled_replacement_enabled": False,
        "phase4_controlled_replacement_mode": "disabled",
        "phase4_controlled_replacement_used": False,
        "phase4_controlled_replacement_would_replace": False,
        "phase4_controlled_replacement_reason": None,
        "phase4_controlled_replacement_candidate_risk_bucket": None,
        "phase4_controlled_replacement_original_full_localization_skipped": False,
        "phase4_controlled_replacement_selected_anchor_ids": [],
        "phase4_controlled_replacement_estimated_saved_time": None,
        "phase4_guard_enabled": False,
        "phase4_guard_type": "none",
        "phase4_guard_pass": True,
        "phase4_guard_block_reason": None,
        "phase4_guard_selected_anchor_span": None,
        "phase4_guard_selected_anchor_max_adjacent_gap": None,
        "phase4_guard_required_candidate_topk": None,
        "phase4_guard_selected_anchor_all_in_required_topk": None,
        "phase4_guard_min_selected_anchor_score": None,
        "phase4_guard_observed_min_selected_anchor_score": None,
        "phase4_guard_min_score_margin": None,
        "phase4_guard_observed_score_margin": None,
        "phase4_replacement_blocked_by_guard": False,
    }


def natural_sorted(values: Iterable[Any]) -> List[str]:
    return sorted([str(value) for value in values], key=natural_sort_key)
