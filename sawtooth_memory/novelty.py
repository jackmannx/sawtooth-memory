"""Residualize fold content against deterministic memory already retained."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .state import EntityLedger

_FOLD_HEADER = re.compile(r"^\[FOLD [^\]]+\]\s*")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class NoveltyResult:
    residual: str
    source_tokens: int
    residual_tokens: int

    @property
    def residual_ratio(self) -> float:
        return self.residual_tokens / max(self.source_tokens, 1)


def residualize(
    source: str,
    ledger: EntityLedger,
    existing_narrative: str,
    *,
    count_text: Callable[[str], int],
) -> NoveltyResult:
    """Remove exact ledger-covered spans and duplicate narrative lines."""
    source_tokens = count_text(source)
    known_lines = {
        _normalize(line)
        for line in existing_narrative.splitlines()
        if line.strip() and not line.startswith("[FOLD ")
    }
    protected: list[str] = list(ledger.entities)
    for history in ledger.entities.values():
        protected.extend(history)
    protected.sort(key=len, reverse=True)

    residual_lines: list[str] = []
    for raw_line in source.splitlines():
        line = _FOLD_HEADER.sub("", raw_line).strip()
        if not line or _normalize(line) in known_lines:
            continue
        for value in protected:
            if len(value) >= 2:
                line = re.sub(re.escape(value), "", line, flags=re.IGNORECASE)
        line = _SPACE.sub(" ", line).strip(" ,;:-")
        if line and _normalize(line) not in known_lines:
            residual_lines.append(line)

    residual = "\n".join(dict.fromkeys(residual_lines))
    return NoveltyResult(
        residual=residual,
        source_tokens=source_tokens,
        residual_tokens=count_text(residual),
    )


def _normalize(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()
