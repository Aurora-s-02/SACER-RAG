from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

from .anchor_localizer import tokenize
from .anchor_schema import normalize_entity


def _unique_text(values: Iterable[Any]) -> Tuple[str, ...]:
    unique: list[str] = []
    seen = set()
    for value in values or []:
        if value is None:
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return tuple(unique)


def _posting_chunk_ids(posting: Any) -> Tuple[str, ...]:
    if isinstance(posting, Mapping):
        return _unique_text(posting.get("chunk_ids", []) or [])
    if isinstance(posting, list):
        return _unique_text(posting)
    return tuple()


def _anchor_ids_from_chunk_entry(entry: Any) -> Tuple[str, ...]:
    if isinstance(entry, Mapping):
        anchor_id = entry.get("anchor_id")
        return (str(anchor_id),) if anchor_id is not None else tuple()
    if isinstance(entry, str):
        return (entry,)
    return tuple()


@dataclass(frozen=True)
class Phase2RuntimeView:
    anchor_ids: Tuple[str, ...]
    anchor_text_tokens: Dict[str, frozenset[str]]
    token_to_anchor_ids: Dict[str, Tuple[str, ...]]
    token_document_frequency: Dict[str, int]
    anchor_to_entities: Dict[str, frozenset[str]]
    entity_to_anchor_to_chunks: Dict[str, Dict[str, Tuple[str, ...]]]
    anchor_descendant_chunk_ids: Dict[str, Tuple[str, ...]]
    anchor_levels: Dict[str, Any]
    anchor_descendant_counts: Dict[str, int]
    chunk_to_anchor_ids: Dict[str, Tuple[str, ...]]
    chunk_to_entities: Dict[str, Tuple[str, ...]]
    doc_count: int


def build_phase2_runtime_view(anchor_index: Dict[str, Any]) -> Phase2RuntimeView:
    anchor_to_chunks = anchor_index.get("anchor_to_chunks", {}) if isinstance(anchor_index, dict) else {}
    anchor_to_entities = anchor_index.get("anchor_to_entities", {}) if isinstance(anchor_index, dict) else {}
    entity_to_anchor = anchor_index.get("entity_to_anchor_to_chunks", {}) if isinstance(anchor_index, dict) else {}
    chunk_to_anchors = anchor_index.get("chunk_to_anchors", {}) if isinstance(anchor_index, dict) else {}
    chunk_to_entities = anchor_index.get("chunk_to_entities", {}) if isinstance(anchor_index, dict) else {}

    anchor_ids = tuple(str(anchor_id) for anchor_id in anchor_to_chunks.keys()) if isinstance(anchor_to_chunks, Mapping) else tuple()
    anchor_text_tokens: Dict[str, frozenset[str]] = {}
    token_to_anchor_ids: Dict[str, list[str]] = defaultdict(list)
    anchor_descendant_chunk_ids: Dict[str, Tuple[str, ...]] = {}
    anchor_levels: Dict[str, Any] = {}
    anchor_descendant_counts: Dict[str, int] = {}

    if isinstance(anchor_to_chunks, Mapping):
        for anchor_id in anchor_ids:
            info = anchor_to_chunks.get(anchor_id, {})
            if not isinstance(info, Mapping):
                tokens = frozenset()
                descendants = tuple()
                level = None
                descendant_count = 0
            else:
                tokens = frozenset(tokenize(info.get("anchor_text", "")))
                descendants = _unique_text(info.get("descendant_chunk_ids", []) or [])
                level = info.get("level")
                descendant_count = int(info.get("num_descendant_chunks", 0) or len(descendants))
            anchor_text_tokens[anchor_id] = tokens
            anchor_descendant_chunk_ids[anchor_id] = descendants
            anchor_levels[anchor_id] = level
            anchor_descendant_counts[anchor_id] = descendant_count
            for token in tokens:
                token_to_anchor_ids[token].append(anchor_id)

    normalized_anchor_to_entities: Dict[str, frozenset[str]] = {}
    if isinstance(anchor_to_entities, Mapping):
        for anchor_id, entry in anchor_to_entities.items():
            entities = entry.keys() if isinstance(entry, Mapping) else entry if isinstance(entry, list) else []
            normalized_anchor_to_entities[str(anchor_id)] = frozenset(
                entity for entity in (normalize_entity(value) for value in entities) if entity
            )

    normalized_entity_to_anchor: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    if isinstance(entity_to_anchor, Mapping):
        for raw_entity, anchor_map in entity_to_anchor.items():
            entity = normalize_entity(raw_entity)
            if not entity or not isinstance(anchor_map, Mapping):
                continue
            target = normalized_entity_to_anchor.setdefault(entity, {})
            for anchor_id, posting in anchor_map.items():
                chunks = _posting_chunk_ids(posting)
                if chunks:
                    target[str(anchor_id)] = chunks

    normalized_chunk_to_anchors: Dict[str, Tuple[str, ...]] = {}
    if isinstance(chunk_to_anchors, Mapping):
        for chunk_id, entry in chunk_to_anchors.items():
            anchors: list[str] = []
            if isinstance(entry, list):
                for item in entry:
                    anchors.extend(_anchor_ids_from_chunk_entry(item))
            else:
                anchors.extend(_anchor_ids_from_chunk_entry(entry))
            normalized_chunk_to_anchors[str(chunk_id)] = _unique_text(anchors)

    normalized_chunk_to_entities: Dict[str, Tuple[str, ...]] = {}
    if isinstance(chunk_to_entities, Mapping):
        for chunk_id, entry in chunk_to_entities.items():
            entities = entry.keys() if isinstance(entry, Mapping) else entry if isinstance(entry, list) else []
            normalized_chunk_to_entities[str(chunk_id)] = tuple(
                entity for entity in (normalize_entity(value) for value in entities) if entity
            )

    return Phase2RuntimeView(
        anchor_ids=anchor_ids,
        anchor_text_tokens=anchor_text_tokens,
        token_to_anchor_ids={token: tuple(ids) for token, ids in token_to_anchor_ids.items()},
        token_document_frequency={token: len(ids) for token, ids in token_to_anchor_ids.items()},
        anchor_to_entities=normalized_anchor_to_entities,
        entity_to_anchor_to_chunks=normalized_entity_to_anchor,
        anchor_descendant_chunk_ids=anchor_descendant_chunk_ids,
        anchor_levels=anchor_levels,
        anchor_descendant_counts=anchor_descendant_counts,
        chunk_to_anchor_ids=normalized_chunk_to_anchors,
        chunk_to_entities=normalized_chunk_to_entities,
        doc_count=len(anchor_ids),
    )
