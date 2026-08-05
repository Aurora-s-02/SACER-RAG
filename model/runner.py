from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

from indexing import build_indexes
from indexing.dense import HashingEmbedder
from indexing.loader import load_index_bundle
from indexing.loader import inspect_index_state
from utils.config import resolve_path
from utils.io import write_json

from .anchor_evidence_contextual_fallback import AecfConfig, run_anchor_evidence_contextual_fallback
from .anchor_retrieval_adapter import AnchorRetrievalAdapter
from .answering import AnswerGenerator
from .dataset_loaders import (
    InfiniteChoiceLoader,
    InfiniteQALoader,
    NovelQALoader,
    resolve_book_ids,
)
from .gate_policy import resolve_gate_policy
from .graphless_retriever import GraphlessRetriever
from .nlp_loader import load_nlp
from .route1_debug_schema import chunk_id_dict_from_list, flatten_chunk_ids, normalize_retrieval_route
from .stem_utils import split_choice_question


def _load_dataset(name: str, path: Path):
    if name == "InfiniteChoice":
        return InfiniteChoiceLoader(str(path))
    if name == "InfiniteQALoader":
        return InfiniteQALoader(str(path))
    if name == "NovelQA":
        return NovelQALoader(str(path))
    raise ValueError(f"Unsupported dataset_name: {name}")


def _graph(edges: List[List[Any]]) -> nx.Graph:
    graph = nx.Graph()
    for left, right, weight in edges:
        graph.add_edge(str(left), str(right), weight=int(weight))
    return graph


def _adapter_config(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    sacer = dict(config.get("sacer", {}))
    retriever = config.get("retriever", {}).get("kwargs", {})
    merged = {**retriever, **sacer}
    requested_low_confidence_route = str(
        merged.get("arg_r_low_confidence_route", "aecf")
    ).casefold()
    if requested_low_confidence_route in {"graph", "graph_local", "graph-local"}:
        raise ValueError(
            "ARG-R graph_local low-confidence routing is deprecated in standalone SACER-RAG. "
            "Use `arg_r_low_confidence_route: aecf`."
        )
    if requested_low_confidence_route != "aecf":
        raise ValueError("sacer.arg_r_low_confidence_route must be `aecf`")
    external_policy = str(merged.get("anchor_evidence_policy", "gate")).casefold()
    if external_policy != "gate":
        raise ValueError(
            "sacer.anchor_evidence_policy only accepts the canonical value `gate`; "
            "`gated` and `loose_gated` are selected internally."
        )
    manually_selected = {
        key
        for key in (
            "phase3_gate_variant",
            "gate_variant",
            "phase3v2_gate_profile",
        )
        if key in merged
    }
    if manually_selected:
        raise ValueError(
            "Gate variants are selected from dataset_name + model_name, not config: "
            + ", ".join(sorted(manually_selected))
        )
    model_name = config.get("llm", {}).get("llm_name") or config.get("llm", {}).get("llm_path")
    gate = resolve_gate_policy(dataset_name, model_name)
    merged.update(
        {
            "phase3_gate_canonical": gate.canonical,
            "phase3_gate_variant": gate.variant,
            "phase3_gate_selection_key": gate.selection_key,
            "phase3_gate_selection_reason": gate.selection_reason,
            "anchor_evidence_policy": "gate",
            "arg_r_low_confidence_route": "aecf",
        }
    )
    return merged


def _aecf_config(config: Dict[str, Any]) -> AecfConfig:
    sacer = dict(config.get("sacer", {}))
    raw = dict(sacer.get("aecf", {}))
    allowed = set(AecfConfig.__dataclass_fields__)
    return AecfConfig(**{key: value for key, value in raw.items() if key in allowed})


def _apply_aecf(
    result: Dict[str, Any],
    *,
    question: str,
    adapter: AnchorRetrievalAdapter,
    anchor_payload: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    pre_aecf_ids = result.get("pre_aecf_chunk_ids") or result.get(
        "final_chunk_ids"
    ) or flatten_chunk_ids(result.get("chunk_ids"))
    if result.get("aecf_requested") is not True:
        result.update(
            {
                "aecf_triggered": False,
                "aecf_trigger_reason": None,
                "pre_aecf_chunk_ids": list(pre_aecf_ids),
                "post_aecf_chunk_ids": list(pre_aecf_ids),
            }
        )
        return result
    trigger_reason = str(result.get("aecf_trigger_reason") or "aecf_requested")
    aecf = run_anchor_evidence_contextual_fallback(
        question=question,
        query_entities=result.get("entities", []),
        original_candidate_chunk_ids=pre_aecf_ids,
        original_route_name="dense_retrieval",
        anchor_index=anchor_payload,
        chunk_texts=getattr(adapter.retriever, "cache_tree", {}),
        config=_aecf_config(config),
        runtime_view=adapter.phase2_runtime_view,
        precomputed_anchor_ids=result.get("selected_anchor_ids", []),
        precomputed_anchor_scores=result.get("selected_anchor_scores", []),
        precomputed_anchor_source="saci",
    )
    result.update(aecf.diagnostics)
    post_aecf_ids = list(aecf.final_chunk_ids or pre_aecf_ids)
    result.update(
        {
            "aecf_triggered": True,
            "aecf_trigger_reason": trigger_reason,
            "pre_aecf_chunk_ids": list(pre_aecf_ids),
            "post_aecf_chunk_ids": post_aecf_ids,
        }
    )
    if aecf.triggered and aecf.replacement_used:
        chunk_ids = chunk_id_dict_from_list(post_aecf_ids)
        result.update(
            {
                "chunks": adapter.retriever.format_res(chunk_ids),
                "chunk_ids": chunk_ids,
                "neighbor_nodes": chunk_ids,
                "keys": list(chunk_ids),
                "len_chunks": len(post_aecf_ids),
                "final_chunk_ids": post_aecf_ids,
                "final_retrieval_route": "aecf",
                "final_retrieval_subroute": f"{trigger_reason}_aecf",
                "final_evidence_chunk_count": len(post_aecf_ids),
            }
        )
    else:
        result["final_retrieval_route"] = "aecf"
        result["final_retrieval_subroute"] = f"{trigger_reason}_preserve"
        result["final_chunk_ids"] = post_aecf_ids
        result["final_evidence_chunk_count"] = len(post_aecf_ids)
    result["graph_evidence_used"] = False
    result["original_graph_fallback_used"] = False
    return result


def _stable_question_id(qa: Dict[str, Any], question_index: int) -> str:
    value = qa.get("question_id")
    if value is None:
        value = qa.get("id")
    if value is None:
        value = question_index
    return str(value)


def _resume_key(
    dataset_name: str,
    book_id: Any,
    question_id: Any,
) -> tuple[str, str, str]:
    return str(dataset_name), str(book_id), str(question_id)


def run_dataset(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    paths_by_name = dataset_cfg.get("paths", {})
    raw_path = paths_by_name.get(dataset_name) if isinstance(paths_by_name, dict) else None
    raw_path = raw_path or dataset_cfg.get("dataset_path")
    if not raw_path:
        raise ValueError(f"No dataset path configured for {dataset_name}")
    dataset_path = resolve_path(config, raw_path)
    dataset = _load_dataset(dataset_name, dataset_path)
    raw_original_path = dataset_cfg.get("original_dataset_path")
    original_dataset_path = (
        resolve_path(config, raw_original_path)
        if raw_original_path
        else None
    )
    # Reject deprecated ARG-R routing before creating outputs or rebuilding an
    # index, so an obsolete config cannot cause expensive side effects first.
    adapter_cfg = _adapter_config(config, dataset_name)
    paths = config.get("paths", {})
    index_root = resolve_path(config, paths.get("index_path", "outputs/index"))
    answer_root = resolve_path(config, paths.get("answer_path", "outputs/answers"))
    output_dir = answer_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = config.get("runtime", {})
    resume = bool(runtime.get("resume", True))
    max_books = runtime.get("max_books")
    book_ids = resolve_book_ids(
        dataset,
        dataset_name,
        dataset_cfg,
        original_dataset_path=original_dataset_path,
    )
    if max_books is not None:
        book_ids = book_ids[: int(max_books)]
    index_state_before = inspect_index_state(index_root, dataset_name, book_ids, config=config)
    index_action = "loaded_existing"
    index_build_summary = None
    if not index_state_before["complete"]:
        index_build_summary = build_indexes(config, dataset_name)
        action_counts = index_build_summary.get("action_counts", {})
        if action_counts.get("full_rebuild"):
            index_action = "full_rebuild"
        elif action_counts.get("incremental_repair"):
            index_action = "incremental_repair"
        else:
            index_action = "loaded_existing"
    index_state_after = inspect_index_state(index_root, dataset_name, book_ids, config=config)
    if not index_state_after["complete"]:
        raise RuntimeError(f"Index remains incomplete after build: {index_state_after}")

    generator = None
    nlp = None
    retriever_kwargs = dict(config.get("retriever", {}).get("kwargs", {}))
    processed = skipped = question_count = resumed_questions = 0
    for position in range(len(dataset)):
        if max_books is not None and position >= int(max_books):
            break
        piece = dataset[position]
        book_id = book_ids[position]
        output_path = output_dir / f"book_{book_id}.json"
        existing_records = []
        if resume and output_path.is_file():
            with output_path.open("r", encoding="utf-8") as infile:
                existing_records = json.load(infile)
            if not isinstance(existing_records, list):
                raise ValueError(f"Resume answer file must contain a list: {output_path}")
        completed_keys = {
            _resume_key(
                record.get("dataset_name"),
                record.get("book_id"),
                record.get("question_id"),
            )
            for record in existing_records
            if record.get("dataset_name") is not None
            and record.get("book_id") is not None
            and record.get("question_id") is not None
        }
        expected_keys = {
            _resume_key(
                dataset_name,
                book_id,
                _stable_question_id(qa, question_index),
            )
            for question_index, qa in enumerate(piece["qa"])
        }
        if expected_keys and expected_keys <= completed_keys:
            resumed_questions += len(expected_keys)
            skipped += 1
            continue
        if nlp is None:
            nlp = load_nlp(
                str(config.get("extractor", {}).get("language", "en")),
                str(config.get("extractor", {}).get("method", "Spacy")),
            )
        bundle = load_index_bundle(index_root / dataset_name / str(book_id))
        dense_metadata = bundle["manifest"].get("dense_index", {})
        embedder_instance = None
        if dense_metadata.get("backend") == "hashing":
            embedder_instance = HashingEmbedder(int(dense_metadata["dimension"]))
        retriever = GraphlessRetriever(
            bundle["tree"],
            _graph(bundle["graph_edges"]),
            bundle["entity_index"],
            bundle["appearance_count"],
            nlp,
            dense_embeddings=bundle["dense_embeddings"],
            dense_ids=bundle["dense_ids"],
            embedder_instance=embedder_instance,
            **retriever_kwargs,
        )
        adapter = AnchorRetrievalAdapter(retriever, bundle["anchor_index_result"], adapter_cfg)
        records = list(existing_records)
        processed_in_book = 0
        resumed_in_book = 0
        for question_index, qa in enumerate(piece["qa"]):
            question_id = _stable_question_id(qa, question_index)
            stable_key = _resume_key(dataset_name, book_id, question_id)
            if stable_key in completed_keys:
                resumed_questions += 1
                resumed_in_book += 1
                continue
            if generator is None:
                generator = AnswerGenerator(config)
            question = str(qa["question"])
            stem, _ = split_choice_question(question) if dataset_name in {"InfiniteChoice", "NovelQA"} else (question, [])
            retrieval_query = stem if adapter_cfg.get("retrieval_input_mode", "original") == "stem" else question
            retrieval = adapter.query(retrieval_query, question_stem=stem, original_question=question)
            retrieval = _apply_aecf(
                retrieval,
                question=stem,
                adapter=adapter,
                anchor_payload=bundle["anchor_index_result"]["payload"],
                config=config,
            )
            prediction, option_probs, generation_mode = generator.generate(
                dataset_name=dataset_name,
                question=question,
                evidence=str(retrieval.get("chunks", "")),
            )
            compatible_debug = {
                key: value
                for key, value in retrieval.items()
                if key != "_route1_original_result"
            }
            record = {
                **compatible_debug,
                "dataset_name": dataset_name,
                "book_index": position,
                "book_id": book_id,
                "question_index": question_index,
                "question_id": question_id,
                "question": question,
                "answer": qa.get("answer"),
                "output_text": prediction,
                "option_probs": option_probs,
                "answer_generation_mode": generation_mode,
                "evidence": retrieval.get("chunks", ""),
                "candidate_chunk_ids": retrieval.get("final_chunk_ids") or flatten_chunk_ids(retrieval.get("chunk_ids")),
                "retrieval_type": retrieval.get("final_retrieval_type") or retrieval.get("retrieval_type"),
                "retrieval_route": retrieval.get("final_retrieval_route")
                or normalize_retrieval_route(retrieval.get("retrieval_type")),
                "retrieval_subroute": retrieval.get("final_retrieval_subroute"),
                "entities": retrieval.get("entities", []),
                "selected_anchor_ids": retrieval.get("selected_anchor_ids", []),
                "anchor_reliability_policy": retrieval.get("anchor_reliability_policy"),
                "anchor_reliability_score": retrieval.get("anchor_reliability_score"),
                "phase4_risk_bucket": retrieval.get("phase4_risk_bucket"),
                "phase4_proposed_route_decision": retrieval.get("phase4_proposed_route_decision"),
                "aecf_triggered": retrieval.get("aecf_triggered", False),
                "aecf_trigger_reason": retrieval.get("aecf_trigger_reason"),
                "pre_aecf_chunk_ids": retrieval.get("pre_aecf_chunk_ids", []),
                "post_aecf_chunk_ids": retrieval.get("post_aecf_chunk_ids", []),
                "aecf_replacement_used": retrieval.get("aecf_replacement_used", False),
                "aecf_preserve_reason": retrieval.get("aecf_preserve_reason"),
                "graph_signal_has_valid_path": retrieval.get("graph_signal_has_valid_path"),
                "graph_evidence_used": retrieval.get("graph_evidence_used"),
                "original_graph_fallback_used": retrieval.get("original_graph_fallback_used", False),
            }
            records.append(record)
            write_json(output_path, records)
            question_count += 1
            processed_in_book += 1
            completed_keys.add(stable_key)
        if processed_in_book:
            processed += 1
        elif resumed_in_book == len(piece["qa"]):
            skipped += 1
    summary = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "processed_books": processed,
        "resumed_books": skipped,
        "processed_questions": question_count,
        "resumed_questions": resumed_questions,
        "answer_dir": str(output_dir),
        "index_action": index_action,
        "index_state_before": index_state_before,
        "index_state_after": index_state_after,
        "index_build_summary": index_build_summary,
        "phase3_gate_canonical": adapter_cfg["phase3_gate_canonical"],
        "phase3_gate_variant": adapter_cfg["phase3_gate_variant"],
        "phase3_gate_selection_key": adapter_cfg["phase3_gate_selection_key"],
        "phase3_gate_selection_reason": adapter_cfg["phase3_gate_selection_reason"],
    }
    write_json(answer_root / f"{dataset_name}_run_summary.json", summary)
    return summary
