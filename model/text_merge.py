from __future__ import annotations

from typing import Any, List


def sequential_merge(chunks: List[str], tokenizer: Any, overlap: int) -> str:
    if not chunks:
        return ""
    if tokenizer is None:
        return "\n".join(chunks)
    result = chunks[0]
    for chunk in chunks[1:]:
        tokenized = tokenizer(chunk, return_tensors="pt")
        ids = tokenized["input_ids"][0][max(0, int(overlap or 0)) :]
        result += tokenizer.decode(ids)
    return result

