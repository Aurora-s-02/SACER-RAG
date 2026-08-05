from __future__ import annotations

import re
from typing import Any, Tuple


SCHEMA_VERSION = "route1.summary_anchor_index.v1"
BUILDER_VERSION = "route1-anchor-builder-0.1"


def normalize_entity(entity: Any) -> str:
    return re.sub(r"\s+", " ", str(entity or "").strip().casefold())


def natural_sort_key(value: Any) -> Tuple[Any, ...]:
    text = str(value)
    parts = re.split(r"(\d+)", text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def anchor_level(anchor_id: str) -> Any:
    """Return human-facing level where summary_0_* is level 1."""
    match = re.match(r"^summary_(\d+)_\d+$", str(anchor_id))
    if match:
        return int(match.group(1)) + 1
    return None

