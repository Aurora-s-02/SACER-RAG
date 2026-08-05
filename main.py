from __future__ import annotations

import argparse
import json
from typing import Sequence

from evaluation import evaluate_outputs
from indexing import build_indexes
from model.runner import run_dataset
from utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="SACER-RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("build-index", "Build and persist chunks, anchors, entities, and link indexes."),
        ("run", "Run SACI -> AFER -> ARG-R -> CACR -> AECF retrieval and answer generation."),
        ("evaluate", "Evaluate saved answers for one dataset."),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, help="YAML configuration path.")
        command.add_argument("--dataset-name", help="Override dataset.dataset_name.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def dispatch(args: argparse.Namespace):
    config = load_config(args.config)
    dataset_name = args.dataset_name or config.get("dataset", {}).get("dataset_name")
    if not dataset_name:
        raise ValueError("Dataset name is required through --dataset-name or dataset.dataset_name")
    if args.command == "build-index":
        return build_indexes(config, dataset_name)
    if args.command == "run":
        return run_dataset(config, dataset_name)
    if args.command == "evaluate":
        return evaluate_outputs(config, dataset_name)
    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    result = dispatch(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
