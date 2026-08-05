from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from model.anchor_index_loader import load_required_summary_anchor_index
from utils.io import read_json
from utils.io import sha256_file
from indexing.builder import BUILDER_VERSION, SCHEMA_VERSION, index_config_fingerprint


REQUIRED_BUNDLE_FILES = (
    "chunks.json",
    "summary_anchors.json",
    "entities.json",
    "tree.json",
    "summary_anchor_index.json",
    "entity_anchor_map.json",
    "anchor_chunk_map.json",
    "dense_index.npy",
    "dense_ids.json",
    "manifest.json",
)


def load_index_bundle(book_dir: str | Path) -> Dict[str, Any]:
    root = Path(book_dir)
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete SACER-RAG index at {root}: missing {', '.join(missing)}. "
            "Run `python main.py build-index`."
        )
    anchor_result = load_required_summary_anchor_index(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible index version at {root}: {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    import numpy as np

    entities = read_json(root / "entities.json")
    return {
        "root": root,
        "chunks": read_json(root / "chunks.json"),
        "summary_anchors": read_json(root / "summary_anchors.json"),
        "tree": read_json(root / "tree.json"),
        "entities": entities,
        "entity_index": entities.get("entity_to_chunks", {}),
        "appearance_count": entities.get("appearance_count", {}),
        "graph_edges": entities.get("graph_edges", []),
        "dense_embeddings": np.load(root / "dense_index.npy", allow_pickle=False),
        "dense_ids": read_json(root / "dense_ids.json"),
        "anchor_index_result": anchor_result,
        "manifest": manifest,
    }


def _anchor_link_errors(anchor: Dict[str, Any], tree: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    anchor_to_chunks = anchor.get("anchor_to_chunks", {})
    chunk_to_anchors = anchor.get("chunk_to_anchors", {})
    valid_leaf_ids = {
        str(chunk_id)
        for chunk_id in tree
        if str(chunk_id).startswith("leaf_")
    }
    for anchor_id, info in anchor_to_chunks.items():
        descendants = {
            str(chunk_id)
            for chunk_id in (info or {}).get("descendant_chunk_ids", [])
        }
        if not descendants:
            errors.append(f"empty_anchor_to_chunks:{anchor_id}")
        for chunk_id in descendants - valid_leaf_ids:
            errors.append(f"dangling_anchor_chunk:{anchor_id}:{chunk_id}")
    for chunk_id, chain in chunk_to_anchors.items():
        if str(chunk_id) not in valid_leaf_ids:
            errors.append(f"dangling_chunk_to_anchor_chunk:{chunk_id}")
        for entry in chain or []:
            anchor_id = str((entry or {}).get("anchor_id"))
            if anchor_id not in anchor_to_chunks:
                errors.append(f"dangling_chunk_anchor:{chunk_id}:{anchor_id}")
    for entity, anchors in anchor.get("entity_to_anchor_to_chunks", {}).items():
        for anchor_id, posting in (anchors or {}).items():
            if anchor_id not in anchor_to_chunks:
                errors.append(f"dangling_entity_anchor:{entity}:{anchor_id}")
                continue
            allowed = set(
                anchor_to_chunks[anchor_id].get("descendant_chunk_ids", [])
            )
            for chunk_id in (posting or {}).get("chunk_ids", []):
                if chunk_id not in allowed:
                    errors.append(
                        f"dangling_entity_anchor_chunk:{entity}:{anchor_id}:{chunk_id}"
                    )
    return errors


def inspect_index_state(
    index_root: str | Path,
    dataset_name: str,
    book_ids,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    root = Path(index_root) / dataset_name
    books = []
    complete = True
    for book_id in book_ids:
        book_dir = root / str(book_id)
        reasons = []
        base_files = ("chunks.json", "summary_anchors.json", "entities.json", "tree.json")
        saci_files = (
            "summary_anchor_index.json",
            "entity_anchor_map.json",
            "anchor_chunk_map.json",
        )
        dense_files = ("dense_index.npy", "dense_ids.json")
        base_complete = all((book_dir / name).is_file() for name in base_files)
        saci_complete = all((book_dir / name).is_file() for name in saci_files)
        dense_complete = all((book_dir / name).is_file() for name in dense_files)
        missing = [name for name in REQUIRED_BUNDLE_FILES if not (book_dir / name).is_file()]
        if missing:
            reasons.append(f"missing_files:{','.join(missing)}")
        if not reasons:
            try:
                manifest = read_json(book_dir / "manifest.json")
                if manifest.get("schema_version") != SCHEMA_VERSION:
                    reasons.append(
                        f"schema_version:{manifest.get('schema_version')!r}!={SCHEMA_VERSION!r}"
                    )
                if manifest.get("builder_version") != BUILDER_VERSION:
                    reasons.append(
                        f"builder_version:{manifest.get('builder_version')!r}!={BUILDER_VERSION!r}"
                    )
                if config is not None and manifest.get("index_config_fingerprint") != index_config_fingerprint(config):
                    reasons.append("index_config_fingerprint_mismatch")
                for name, expected in (manifest.get("sha256") or {}).items():
                    path = book_dir / name
                    if not path.is_file():
                        reasons.append(f"missing_hashed_file:{name}")
                    elif sha256_file(path) != expected:
                        reasons.append(f"sha256_mismatch:{name}")
                anchor = load_required_summary_anchor_index(book_dir)["payload"]
                tree = read_json(book_dir / "tree.json")
                link_errors = _anchor_link_errors(anchor, tree)
                if link_errors:
                    reasons.append(
                        "anchor_link_validation:"
                        + ",".join(link_errors[:10])
                    )
                if read_json(book_dir / "entity_anchor_map.json") != anchor["entity_to_anchor_to_chunks"]:
                    reasons.append("entity_anchor_map_mismatch")
                sidecar = read_json(book_dir / "anchor_chunk_map.json")
                if sidecar.get("anchor_to_chunks") != anchor["anchor_to_chunks"]:
                    reasons.append("anchor_to_chunks_mismatch")
                if sidecar.get("chunk_to_anchors") != anchor["chunk_to_anchors"]:
                    reasons.append("chunk_to_anchors_mismatch")
                dense_ids = [
                    str(value)
                    for value in read_json(book_dir / "dense_ids.json")
                ]
                if any(not chunk_id.startswith("leaf_") for chunk_id in dense_ids):
                    reasons.append("dense_ids_include_non_leaf")
                if any(chunk_id not in tree for chunk_id in dense_ids):
                    reasons.append("dense_ids_dangling")
            except Exception as exc:
                reasons.append(f"load_or_validation_error:{exc}")
        book_complete = not reasons
        complete = complete and book_complete
        books.append(
            {
                "book_id": str(book_id),
                "complete": book_complete,
                "base_complete": base_complete,
                "saci_complete": saci_complete,
                "dense_complete": dense_complete,
                "reasons": reasons,
            }
        )
    return {
        "dataset_name": dataset_name,
        "index_root": str(root),
        "complete": complete and bool(books),
        "base_complete": bool(books) and all(book["base_complete"] for book in books),
        "saci_complete": bool(books) and all(book["saci_complete"] for book in books),
        "dense_complete": bool(books) and all(book["dense_complete"] for book in books),
        "book_count": len(books),
        "books": books,
    }
