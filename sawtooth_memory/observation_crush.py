"""Deterministic, reversible compaction for large tool observations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable

_IMPORTANT_LOG = re.compile(
    r"\b(error|exception|fatal|failed|failure|warning|warn|traceback|panic)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ObservationCrushResult:
    content: str
    original_tokens: int
    compressed_tokens: int
    cache_id: str | None = None
    strategy: str = "none"

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def crushed(self) -> bool:
        return self.cache_id is not None


def crush_observation(
    content: str,
    *,
    count_text: Callable[[str], int],
    min_tokens: int,
) -> ObservationCrushResult:
    """Compact a large tool observation without invoking an LLM.

    The caller owns storage of the original under ``cache_id``. Small inputs and
    transformations that do not save tokens pass through byte-for-byte.
    """
    original_tokens = count_text(content)
    if original_tokens < min_tokens:
        return ObservationCrushResult(content, original_tokens, original_tokens)

    compact, strategy = _compact_json(content)
    if compact is None:
        compact, strategy = _compact_log(content), "log"

    cache_id = f"obs_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
    wrapped = (
        f"[OBSERVATION_CRUSHED id={cache_id} strategy={strategy} "
        "retrieve=retrieve_observation]\n"
        f"{compact}"
    )
    compressed_tokens = count_text(wrapped)
    if compressed_tokens >= original_tokens:
        return ObservationCrushResult(content, original_tokens, original_tokens)

    return ObservationCrushResult(
        wrapped,
        original_tokens,
        compressed_tokens,
        cache_id=cache_id,
        strategy=strategy,
    )


def _compact_json(content: str) -> tuple[str | None, str]:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None, "none"

    compact = _sample_json(value)
    return json.dumps(compact, separators=(",", ":"), ensure_ascii=False), "json"


def _sample_json(value: object) -> object:
    if isinstance(value, list):
        if len(value) <= 8:
            return value
        return {
            "_type": "array_sample",
            "_count": len(value),
            "head": value[:4],
            "tail": value[-2:],
        }
    if isinstance(value, dict):
        sampled: dict[str, object] = {}
        for key, item in value.items():
            if isinstance(item, list) and len(item) > 8:
                sampled[key] = _sample_json(item)
            else:
                sampled[key] = item
        return sampled
    return value


def _compact_log(content: str) -> str:
    lines = content.splitlines()
    if len(lines) <= 12:
        return content

    keep: set[int] = set(range(min(3, len(lines))))
    keep.update(range(max(0, len(lines) - 3), len(lines)))
    for index, line in enumerate(lines):
        if _IMPORTANT_LOG.search(line):
            keep.update(range(max(0, index - 1), min(len(lines), index + 2)))

    selected: list[str] = []
    previous = -2
    for index in sorted(keep):
        if index > previous + 1:
            selected.append(f"... [{index - previous - 1} lines omitted] ...")
        selected.append(lines[index])
        previous = index
    return "\n".join(selected)
