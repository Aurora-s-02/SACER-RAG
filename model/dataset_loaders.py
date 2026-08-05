from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


logger = logging.getLogger(__name__)


class NovelQALoader:
    def __init__(self, path: str):
        self.parent_folder = path
        self.qapath = os.path.join(path, "Data")
        self.docpath = os.path.join(path, "Books")
        self.dataset = self._initialize_dataset()
        self.available_ids = list(self.dataset.keys())

    def _initialize_dataset(self) -> Dict[int, Dict[str, Any]]:
        dataset: Dict[int, Dict[str, Any]] = {}
        for _, dirs, _ in os.walk(self.docpath):
            for directory in dirs:
                directory_path = os.path.join(self.docpath, directory)
                for filename in os.listdir(directory_path):
                    with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as infile:
                        book_id = int(filename.split(".")[0][1:])
                        dataset[book_id] = {"book": infile.read()}
        for _, dirs, _ in os.walk(self.qapath):
            for directory in dirs:
                directory_path = os.path.join(self.qapath, directory)
                for filename in os.listdir(directory_path):
                    with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as infile:
                        qa_id = int(filename.split(".")[0][1:])
                        dataset.setdefault(qa_id, {})["qa"] = json.loads(infile.read())
        return dataset

    def _format_qa(self, qa_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        formatted_qa = []
        for qa_id, qa in qa_dict.items():
            question_text = str(qa["Question"]) + "\n"
            for option, text in qa["Options"].items():
                question_text += option + ". " + text
                if option != "D":
                    question_text += "\n"
            formatted_qa.append(
                {
                    "id": qa_id,
                    "question": question_text,
                    "answer": qa["Gold"],
                    "evidence": qa["Evidences"],
                }
            )
        return formatted_qa

    def __getitem__(self, index: int) -> Dict[str, Any]:
        book_id = self.available_ids[index]
        return {
            "book": self.dataset[book_id]["book"],
            "qa": self._format_qa(self.dataset[book_id]["qa"]),
        }

    def __len__(self) -> int:
        return len(self.available_ids)


class InfiniteQALoader:
    def __init__(self, path: str):
        self.dataset, self.available_ids = self._initialize_dataset(Path(path))

    def _initialize_dataset(self, path: Path) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
        dataset: Dict[int, Dict[str, Any]] = {}
        context_id = -1
        prev_context = ""
        with path.open("r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON line: %s", exc)
                    continue
                context = data["context"]
                if context != prev_context:
                    context_id += 1
                    prev_context = context
                    dataset[context_id] = {"book": context, "qa": []}
                dataset[context_id]["qa"].append(
                    {
                        "question_id": data.get("question_id", data.get("id")),
                        "question": data["input"],
                        "answer": data["answer"],
                    }
                )
        return dataset, list(dataset.keys())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.dataset[self.available_ids[index]]

    def __len__(self) -> int:
        return len(self.available_ids)


class InfiniteChoiceLoader:
    def __init__(self, path: str):
        self.dataset, self.available_ids = self._initialize_dataset(Path(path))

    def _initialize_dataset(self, path: Path) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
        dataset: Dict[int, Dict[str, Any]] = {}
        prev_context = ""
        context_id = -1
        with path.open("r", encoding="utf-8") as infile:
            for line in infile:
                data = json.loads(line)
                context = data["context"]
                if context != prev_context:
                    context_id += 1
                    prev_context = context
                    dataset[context_id] = {"book": context, "qa": []}
                dataset[context_id]["qa"].append(
                    self._format_qa(
                        {
                            "question_id": data.get("question_id", data.get("id")),
                            "question": data["input"],
                            "answer": data["answer"],
                            "options": data["options"],
                        }
                    )
                )
        return dataset, list(dataset.keys())

    def _format_qa(self, qa_dict: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        result["question_id"] = qa_dict.get("question_id")
        option_names = ["A", "B", "C", "D"]
        formatted_question = str(qa_dict["question"]) + "\n"
        for index, option in enumerate(qa_dict["options"]):
            formatted_question += option_names[index] + ". " + option + "\n"
            if qa_dict["answer"][0] == option:
                result["answer"] = option_names[index]
        result["question"] = formatted_question
        return result

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.dataset[self.available_ids[index]]

    def __len__(self) -> int:
        return len(self.available_ids)


def resolve_book_ids(
    dataset: Any,
    dataset_name: str,
    dataset_config: Dict[str, Any],
    *,
    original_dataset_path: Path | None = None,
) -> List[Any]:
    explicit = dataset_config.get("book_ids")
    if isinstance(explicit, dict):
        explicit = explicit.get(dataset_name)
    if explicit is not None:
        resolved = list(explicit)
        if len(resolved) != len(dataset):
            raise ValueError(
                f"Configured book_ids length {len(resolved)} does not match "
                f"{dataset_name} length {len(dataset)}"
            )
        return resolved

    mode = str(dataset_config.get("book_id_mode", "all")).strip().casefold()
    if mode in {"all", "subset", "dataset"}:
        return list(dataset.available_ids)
    if mode != "original":
        raise ValueError(
            "dataset.book_id_mode must be one of all, subset, dataset, original"
        )
    if dataset_name != "InfiniteChoice":
        return list(dataset.available_ids)
    if original_dataset_path is None:
        raise ValueError(
            "dataset.original_dataset_path is required for "
            "InfiniteChoice book_id_mode=original"
        )
    full_dataset = InfiniteChoiceLoader(str(original_dataset_path))
    full_ids_by_context = {
        full_dataset[position]["book"]: full_dataset.available_ids[position]
        for position in range(len(full_dataset))
    }
    resolved = []
    for position in range(len(dataset)):
        context = dataset[position]["book"]
        if context not in full_ids_by_context:
            raise ValueError(
                f"Subset context at position {position} is absent from "
                f"{original_dataset_path}"
            )
        resolved.append(full_ids_by_context[context])
    return resolved
