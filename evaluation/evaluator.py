from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from utils.config import resolve_path
from utils.io import write_json


def _records(output_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(output_dir.glob("book_*.json")):
        with path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
        if not isinstance(payload, list):
            raise ValueError(f"Answer file must contain a list: {path}")
        yield from payload


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _match(prediction: Any, gold: Any) -> bool:
    if isinstance(gold, list):
        return any(_normalize(prediction) == _normalize(item) for item in gold)
    return _normalize(prediction) == _normalize(gold)


def _rouge_l_f1(prediction: str, reference: str) -> float:
    try:
        from rouge import Rouge

        return float(
            Rouge().get_scores(
                _normalize(prediction),
                _normalize(reference),
            )[0]["rouge-l"]["f"]
        )
    except (ImportError, ValueError, IndexError, KeyError):
        # Keep evaluation usable for empty strings and minimal test
        # environments, while formal installations use the source-compatible
        # `rouge` package declared in requirements.txt.
        pass
    left = list(_normalize(prediction))
    right = list(_normalize(reference))
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if token == other else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(left)
    recall = lcs / len(right)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def evaluate_outputs(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    paths = config.get("paths", {})
    answer_root = resolve_path(config, paths.get("answer_path", "outputs/answers"))
    output_dir = answer_root / dataset_name
    records: List[Dict[str, Any]] = list(_records(output_dir))
    if not records:
        raise FileNotFoundError(f"No answer files found in {output_dir}")

    if dataset_name in {"InfiniteChoice", "NovelQA"}:
        correct = sum(_match(record.get("output_text"), record.get("answer")) for record in records)
        metrics = {
            "metric": "accuracy",
            "score": correct / len(records),
            "correct": correct,
            "total": len(records),
        }
    else:
        scores = []
        for record in records:
            gold = record.get("answer")
            references = gold if isinstance(gold, list) else [gold]
            scores.append(max((_rouge_l_f1(record.get("output_text", ""), item) for item in references), default=0.0))
        metrics = {
            "metric": "rouge_l_f1",
            "score": sum(scores) / len(scores),
            "total": len(scores),
        }
    evaluation_root = resolve_path(config, paths.get("evaluation_path", "outputs/evaluation"))
    result = {"dataset_name": dataset_name, **metrics}
    write_json(evaluation_root / f"{dataset_name}.json", result)
    return result
