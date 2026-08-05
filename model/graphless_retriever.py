from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np

from .anchor_schema import normalize_entity
from .graph_signal import build_graph_signal
from .text_merge import sequential_merge


logger = logging.getLogger(__name__)


def _natural_chunk_key(chunk_id: Any) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(chunk_id))
    return tuple(int(part) if part.isdigit() else part for part in parts)


class GraphlessRetriever:
    """Retriever-compatible implementation used by the standalone model runner."""

    def __init__(self, cache_tree, G, index, appearance_count: Dict[str, Any], nlp, **kwargs) -> None:
        self.cache_tree = cache_tree or {}
        self.G = G
        self.index = index or {}
        self.appearance_count = appearance_count or {}
        self.inverse_index = self.get_inverse_index()
        self.nlp = nlp
        self.device = kwargs.get("device", "cuda:0")
        self.merge_num = kwargs.get("merge_num", 5)
        self.min_count = kwargs.get("min_count", 2)
        self.overlap = kwargs.get("overlap", 100)
        self.tokenizer = self._load_tokenizer(kwargs.get("tokenizer"))
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(self.cache_tree)
        self.dense_embeddings = kwargs.get("dense_embeddings")
        dense_ids = kwargs.get("dense_ids")
        if dense_ids:
            self.collapse_tree_ids = [str(value) for value in dense_ids]
        self.embedder = kwargs.get("embedder_instance")
        self.faiss_index = None
        embedder_name = kwargs.get("embedder", "/path/to/embedder")
        if self.embedder is None and embedder_name is not None:
            self.embedder = self._load_embedder(embedder_name)
        if self.embedder is not None:
            self.faiss_index = self._build_faiss_index()

    def _load_tokenizer(self, tokenizer_name: Optional[str]) -> Any:
        if not tokenizer_name:
            return None
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as exc:
            logger.warning("Tokenizer unavailable for graphless retriever: %s", exc)
            return None

    def _load_embedder(self, embedder_name: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(embedder_name, device=self.device)
            embedder.eval()
            return embedder
        except Exception as exc:
            logger.warning("Dense embedder unavailable for graphless retriever: %s", exc)
            return None

    def __del__(self):
        try:
            if hasattr(self, "embedder"):
                del self.embedder
            if hasattr(self, "faiss_index"):
                del self.faiss_index
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as exc:
            logger.error("Error during GraphlessRetriever cleanup: %s", exc)

    def update(self, cache_tree, G, index, appearance_count):
        self.cache_tree = cache_tree or {}
        self.G = G
        self.index = index or {}
        self.appearance_count = appearance_count or {}
        self.inverse_index = self.get_inverse_index()
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(self.cache_tree)
        if self.embedder is not None:
            self.faiss_index = self._build_faiss_index()

    def get_inverse_index(self) -> Dict[str, List[str]]:
        inverse_index: Dict[str, List[str]] = {}
        for key, values in (self.index or {}).items():
            for chunk_id in values or []:
                inverse_index.setdefault(str(chunk_id), []).append(str(key))
        return inverse_index

    def _collapse_tree(self, cache_tree: Dict[str, Dict[str, Any]]) -> tuple[List[str], List[str]]:
        texts: List[str] = []
        ids: List[str] = []
        for key in sorted((cache_tree or {}).keys(), key=_natural_chunk_key):
            value = cache_tree.get(key, {})
            if not isinstance(value, dict):
                continue
            text = value.get("text")
            if text:
                texts.append(str(text))
                ids.append(str(key))
        return texts, ids

    def _leaf_ids(self) -> List[str]:
        return [
            chunk_id
            for chunk_id in sorted((self.cache_tree or {}).keys(), key=_natural_chunk_key)
            if str(chunk_id).startswith("leaf_")
        ]

    def _build_faiss_index(self):
        if self.embedder is None or not self.collapse_tree:
            return None
        try:
            import faiss

            doc_embeds = self.dense_embeddings
            if doc_embeds is None:
                doc_embeds = self.embedder.encode(self.collapse_tree, batch_size=16, device=self.device)
            vector_database = faiss.IndexFlatIP(doc_embeds.shape[1])
            vector_database.add(doc_embeds)
            return vector_database
        except Exception as exc:
            logger.warning("FAISS index unavailable for graphless retriever: %s", exc)
            return None

    def _detect_neighbor_nodes(self, keys: Set[str], chunk_id: str) -> List[str]:
        match = re.search(r"(\d+)$", str(chunk_id))
        if not match:
            return [str(chunk_id)]
        center = int(match.group(1))
        neighbors = [str(chunk_id)]
        for offset in (-1, 1):
            candidate = f"leaf_{center + offset}"
            if candidate in self.cache_tree and keys & set(self.inverse_index.get(candidate, [])):
                neighbors.append(candidate)
        return sorted(set(neighbors), key=_natural_chunk_key)

    def detect_contiguous_chunks(self, chunk_ids: List[str]) -> List[List[str]]:
        result: List[List[str]] = []
        current: List[str] = []
        for chunk_id in sorted(chunk_ids, key=_natural_chunk_key):
            match = re.search(r"(\d+)$", str(chunk_id))
            number = int(match.group(1)) if match else None
            prev_match = re.search(r"(\d+)$", str(current[-1])) if current else None
            prev_number = int(prev_match.group(1)) if prev_match else None
            if not current or number is None or prev_number is None or number == prev_number + 1:
                current.append(str(chunk_id))
            else:
                result.append(current)
                current = [str(chunk_id)]
        if current:
            result.append(current)
        return result

    def get_contiguous_chunks(self, leaf_nodes: List[str]) -> str:
        texts = [str((self.cache_tree.get(node) or {}).get("text", "")) for node in leaf_nodes]
        texts = [text for text in texts if text]
        if not texts:
            return ""
        if self.tokenizer is None:
            return "\n".join(texts)
        try:
            return sequential_merge(texts, self.tokenizer, self.overlap)
        except Exception:
            return "\n".join(texts)

    def format_res(self, res: Dict[str, List[str]]) -> str:
        parts: List[str] = []
        for key, chunks in (res or {}).items():
            for chunk_list in self.detect_contiguous_chunks([str(chunk_id) for chunk_id in chunks or []]):
                text = self.get_contiguous_chunks(chunk_list)
                if text:
                    parts.append(f"{key}: {text}")
        return "\n".join(parts) + ("\n" if parts else "")

    def _count_chunks(self, res: Dict[str, List[str]]) -> int:
        return sum(len(values or []) for values in (res or {}).values())

    def _extract_entities(self, query: str) -> List[str]:
        if self.nlp is None or not hasattr(self.nlp, "naive_extract_graph"):
            return []
        try:
            result = self.nlp.naive_extract_graph(str(query).split("\n")[0])
            entities = [
                normalize_entity(entity)
                for entity in result.get("nouns", [])
                if entity
            ]
            return [entity for entity in entities if entity]
        except Exception:
            return []

    def graph_filter(self, entities: Iterable[str], shortest_path_k: int) -> List[tuple[str, str]]:
        graph = self.G
        if graph is None:
            return []
        pairs = []
        try:
            nodes = set(graph.nodes())
            for head, tail in combinations(list(entities or []), 2):
                if head not in nodes or tail not in nodes:
                    continue
                try:
                    path = graph.shortest_path(head, tail) if hasattr(graph, "shortest_path") else None
                except Exception:
                    path = None
                if path is None:
                    try:
                        import networkx as nx

                        path = nx.shortest_path(graph, head, tail)
                    except Exception:
                        continue
                if len(path) <= shortest_path_k:
                    pairs.append((str(head), str(tail)))
        except Exception:
            return []
        return pairs

    def index_mapping(self, entity_groups: Iterable[Any]) -> Dict[str, List[str]]:
        chunk_ids: Dict[str, List[str]] = {}
        for entity_group in entity_groups or []:
            if isinstance(entity_group, str):
                if entity_group in self.index:
                    chunk_ids[entity_group] = [str(chunk_id) for chunk_id in self.index.get(entity_group, []) or []]
                continue
            if isinstance(entity_group, (tuple, list)):
                entity_key = "_".join(str(entity) for entity in entity_group)
                chunk_ids_set: Optional[Set[str]] = None
                for entity in entity_group:
                    postings = {str(chunk_id) for chunk_id in self.index.get(str(entity), []) or []}
                    if chunk_ids_set is None:
                        chunk_ids_set = postings
                    else:
                        chunk_ids_set &= postings
                chunk_ids[entity_key] = sorted(chunk_ids_set or [], key=_natural_chunk_key)
        return chunk_ids

    def local_retrieval(self, entities: List[str], shortest_path_k: int = 4) -> Dict[str, List[str]]:
        shortest_path_pairs = self.graph_filter(entities, shortest_path_k)
        init_chunk_ids = self.index_mapping(shortest_path_pairs)
        return self.merge_keys(init_chunk_ids)

    def _has_valid_graph_path(self, entities: Iterable[str], shortest_path_k: int) -> bool:
        return bool(
            self._count_chunks(
                self.local_retrieval(list(entities or []), shortest_path_k)
            )
        )

    def dense_retrieval(self, query: str, k: int) -> Dict[str, List[str]]:
        k = max(1, int(k or 1))
        if self.embedder is None:
            return {"": self._leaf_ids()[:k]}
        query_embed = self.embedder.encode(query).reshape(1, -1)
        if self.faiss_index is not None:
            _, candidate_indexes = self.faiss_index.search(query_embed, k=min(k, len(self.collapse_tree_ids)))
            indexes = candidate_indexes[0]
        elif self.dense_embeddings is not None:
            scores = np.asarray(self.dense_embeddings) @ np.asarray(query_embed[0])
            indexes = np.argsort(-scores, kind="stable")[: min(k, len(self.collapse_tree_ids))]
        else:
            return {"": self._leaf_ids()[:k]}
        candidate_ids = [self.collapse_tree_ids[int(i)] for i in indexes]
        return {"": candidate_ids}

    def _check_children(self, chunk_id: str, entities: List[str], visited=None) -> int:
        if visited is None:
            visited = set()
        if chunk_id in visited:
            return 0
        visited.add(chunk_id)
        children = (self.cache_tree.get(chunk_id) or {}).get("children", [])
        if not children:
            stats = self.appearance_count.get(chunk_id, {}) or {}
            return sum(int(stats.get(entity, 0) or 0) for entity in entities)
        return sum(self._check_children(str(child), entities, visited) for child in children)

    def merge_keys(self, neighbor_nodes: Dict[str, List[str]]) -> Dict[str, List[str]]:
        chunks_to_keys: Dict[str, Set[str]] = {}
        for key, chunk_ids in (neighbor_nodes or {}).items():
            for chunk_id in chunk_ids or []:
                chunks_to_keys.setdefault(str(chunk_id), set()).add(str(key))
        merged: Dict[str, List[str]] = {}
        for chunk_id in sorted(chunks_to_keys.keys(), key=_natural_chunk_key):
            keys = chunks_to_keys[chunk_id]
            merged_key = "_".join(sorted(entity for key in keys for entity in key.split("_") if entity))
            merged.setdefault(merged_key, []).append(chunk_id)
        return merged

    def occurrence_ranking(self, candidate_chunk_ids: List[str], entities: List[str], top_k: int) -> Dict[str, List[str]]:
        scored = []
        for chunk_id in candidate_chunk_ids:
            count = self._check_children(str(chunk_id), entities)
            scored.append((count, str(chunk_id)))
        positives = [(count, chunk_id) for count, chunk_id in scored if count > 0]
        if not positives:
            return {"": [str(chunk_id) for chunk_id in candidate_chunk_ids[:top_k]]}
        positives.sort(key=lambda item: (-item[0], _natural_chunk_key(item[1])))
        selected = [chunk_id for _, chunk_id in positives[:top_k]]
        grouped: Dict[str, List[str]] = {}
        for chunk_id in selected:
            for entity in entities:
                if entity in self.inverse_index.get(chunk_id, []):
                    grouped.setdefault(entity, []).append(chunk_id)
        return self.merge_keys(grouped) if grouped else {"": selected}

    def entityaware_filter(
        self,
        candidate_chunks: Dict[str, List[str]],
        entities: List[str],
        top_k: int,
    ) -> Dict[str, List[str]]:
        chunks_info = []
        for key, chunk_ids in (candidate_chunks or {}).items():
            key_entities = [entity for entity in str(key).split("_") if entity]
            key_set = set(key_entities)
            for chunk_id in chunk_ids or []:
                stats = self.appearance_count.get(str(chunk_id), {}) or {}
                entity_count = sum(int(stats.get(entity, 0) or 0) for entity in key_entities)
                chunks_info.append(
                    {
                        "chunk_id": str(chunk_id),
                        "key_count": len(key_entities),
                        "neighbor_nodes_count": len(self._detect_neighbor_nodes(key_set, str(chunk_id))),
                        "entity_count": entity_count,
                    }
                )
        if not chunks_info:
            return {}
        sorted_chunks_info = sorted(
            chunks_info,
            key=lambda item: (item["key_count"], item["neighbor_nodes_count"], item["entity_count"]),
            reverse=True,
        )
        top_k_chunk_ids = [item["chunk_id"] for item in sorted_chunks_info[: max(0, int(top_k or 0))]]
        filtered_res: Dict[str, List[str]] = {}
        for chunk_id in top_k_chunk_ids:
            for entity in entities:
                if chunk_id in self.index.get(entity, []):
                    filtered_res.setdefault(entity, []).append(chunk_id)
        return self.merge_keys(filtered_res) if filtered_res else {"": top_k_chunk_ids}

    def _build_supplement_info(
        self,
        chunk_ids: Dict[str, List[str]],
        entities: List[str],
        len_chunks: int,
        chunk_counts_history: List[Any],
        graph_signal_has_valid_path: bool,
    ) -> Dict[str, Any]:
        signal = build_graph_signal(
            chunk_count=1 if graph_signal_has_valid_path else 0
        )
        return {
            "chunk_ids": chunk_ids,
            "entities": entities,
            "neighbor_nodes": chunk_ids,
            "keys": list(chunk_ids.keys()),
            "len_chunks": len_chunks,
            "chunk_counts_history": chunk_counts_history,
            **signal,
            "graph_evidence_used": False,
            "non_graph_retrieval_route": "dense_retrieval",
            "original_graph_fallback_used": False,
        }

    def query(self, query, **kwargs):
        max_chunks = int(kwargs.get("max_chunk_setting", 25) or 25)
        shortest_path_k = int(kwargs.get("shortest_path_k", 4) or 4)
        entities = self._extract_entities(str(query))
        try:
            graph_signal_has_valid_path = False
            if entities:
                graph_signal_has_valid_path = self._has_valid_graph_path(
                    entities,
                    shortest_path_k,
                )
        except Exception:
            graph_signal_has_valid_path = False
        chunk_counts_history = []
        chunk_ids = self.dense_retrieval(str(query), max_chunks)
        result = {"chunks": self.format_res(chunk_ids)}
        if kwargs.get("debug", True):
            result.update(
                self._build_supplement_info(
                    chunk_ids,
                    entities,
                    self._count_chunks(chunk_ids),
                    chunk_counts_history,
                    graph_signal_has_valid_path,
                )
            )
            result["retrieval_type"] = "Dense Retrieval"
        return result
