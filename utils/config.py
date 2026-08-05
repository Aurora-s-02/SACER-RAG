from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_REMOVED_SACER_CONTROLS = {
    "ablation" + "_profile",
    "force_" + "aecf_all",
    "force_" + "all_queries",
    "use_saci_index",
    "enable_anchor_first_routing",
    "enable_anchor_reliability_gate",
    "enable_phase4_selective_routing",
}
_INTERNAL_GATE_CONTROLS = {
    "phase3_gate_variant",
    "gate_variant",
    "phase3v2_gate_profile",
}


def validate_public_config(config: Dict[str, Any]) -> None:
    """Validate the full-model-only configuration contract."""
    sacer = config.get("sacer", {})
    if not isinstance(sacer, dict):
        raise ValueError("Config field `sacer` must be a mapping")
    removed = sorted(_REMOVED_SACER_CONTROLS.intersection(sacer))
    if removed:
        raise ValueError(
            "The public release accepts only the complete SACER-RAG pipeline; "
            f"removed controls: {', '.join(removed)}"
        )
    internal = sorted(_INTERNAL_GATE_CONTROLS.intersection(sacer))
    if internal:
        raise ValueError(
            "Gate variants are selected internally from dataset and model: "
            + ", ".join(internal)
        )
    policy = str(sacer.get("anchor_evidence_policy", "gate")).casefold()
    if policy != "gate":
        raise ValueError("sacer.anchor_evidence_policy must be `gate`")
    low_route = str(sacer.get("arg_r_low_confidence_route", "aecf")).casefold()
    if low_route != "aecf":
        raise ValueError("sacer.arg_r_low_confidence_route must be `aecf`")
    aecf = sacer.get("aecf", {})
    if aecf is not None and not isinstance(aecf, dict):
        raise ValueError("Config field `sacer.aecf` must be a mapping")
    if isinstance(aecf, dict) and "enabled" in aecf:
        raise ValueError(
            "The public release does not expose module-disable controls; "
            "remove `sacer.aecf.enabled`."
        )


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as infile:
        config = yaml.safe_load(infile) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    config = deepcopy(config)
    validate_public_config(config)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_path(config: Dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_dir = Path(config.get("_config_dir") or PROJECT_ROOT)
    return (config_dir / path).resolve()


def require_mapping(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field `{key}` must be a mapping")
    return value
