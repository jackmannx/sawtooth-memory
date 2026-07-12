"""
sawtooth_memory/ner.py
Deterministic, local-first Named Entity Recognition pipeline.
Runs entirely in-process with pre-compiled patterns for zero-latency extraction.
"""

from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass, field
from typing import Dict, Literal, Protocol

from .salience import SalienceConfig, SalienceEntityExtractor

# Ambient context channel for telemetry tracking across async boundaries
active_strategy_context: contextvars.ContextVar[Dict[str, str]] = (
    contextvars.ContextVar("active_strategy_context", default={})
)

EntityStrategy = Literal[
    "deterministic", "salience_heuristic", "pinned", "llm_synthesis"
]


@dataclass
class ExtractionResult:
    """Entities plus per-key extraction strategy provenance."""

    entities: dict[str, str] = field(default_factory=dict)
    strategies: dict[str, EntityStrategy] = field(default_factory=dict)


class DeterministicExtractor(Protocol):
    """Protocol for any deterministic value extractor."""

    def extract(self, text: str) -> Dict[str, str]: ...


class RegexEntityExtractor:
    """High-performance regex matcher with configurable patterns."""

    _DEFAULT_PATTERNS: Dict[str, str] = {
        "uuid": r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        "file_path": r"(?:/[a-zA-Z0-9_\-.]+)+",
        "uri": r"[a-zA-Z][a-zA-Z0-9+\-.]*://[^\s]+",
    }

    def __init__(self, extra_patterns: Dict[str, str] | None = None) -> None:
        self._patterns: Dict[str, re.Pattern[str]] = {}
        all_patterns = {**self._DEFAULT_PATTERNS, **(extra_patterns or {})}
        for entity_name, regex_str in all_patterns.items():
            self._patterns[entity_name] = re.compile(regex_str)

    def extract(self, text: str) -> Dict[str, str]:
        """Return all matches for each pattern in the text."""
        result: Dict[str, str] = {}
        for name, pattern in self._patterns.items():
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            result[name] = matches[0].group(0)
            for idx, match in enumerate(matches[1:], start=2):
                result[f"{name}_{idx}"] = match.group(0)
        return result


class NERPipeline:
    """Orchestrates deterministic extraction layers."""

    def __init__(self, *extractors: DeterministicExtractor) -> None:
        self._extractors = list(extractors)
        self._regex_extractor: RegexEntityExtractor | None = None
        self._salience_extractor: SalienceEntityExtractor | None = None
        for ext in self._extractors:
            if isinstance(ext, RegexEntityExtractor):
                self._regex_extractor = ext
            elif isinstance(ext, SalienceEntityExtractor):
                self._salience_extractor = ext

    def extract(self, text: str) -> Dict[str, str]:
        """Backward-compatible entity-only extraction."""
        return self.extract_with_metadata(text).entities

    def extract_with_metadata(self, text: str) -> ExtractionResult:
        """Extract entities and record per-key strategy provenance."""
        result = ExtractionResult()
        regex_entities: Dict[str, str] = {}

        if self._regex_extractor is not None:
            regex_entities = self._regex_extractor.extract(text)
            result.entities.update(regex_entities)
            for key in regex_entities:
                result.strategies[key] = "deterministic"

        if self._salience_extractor is not None:
            exclude = set(regex_entities.values())
            salience_entities = self._salience_extractor.extract(
                text, exclude_values=exclude
            )
            for key, value in salience_entities.items():
                if value in result.entities.values():
                    continue
                result.entities[key] = value
                result.strategies[key] = "salience_heuristic"

        # Generic extractors without strategy metadata
        for ext in self._extractors:
            if isinstance(ext, (RegexEntityExtractor, SalienceEntityExtractor)):
                continue
            generic = ext.extract(text)
            for key, value in generic.items():
                if key not in result.entities:
                    result.entities[key] = value
                    result.strategies[key] = "deterministic"

        return result

    @classmethod
    def from_config(
        cls,
        enable: bool = True,
        custom_patterns: Dict[str, str] | None = None,
        *,
        enable_salience: bool = True,
        salience_threshold: float = 0.5,
        salience_max_entities: int = 20,
    ) -> "NERPipeline":
        if not enable:
            return cls()
        extractors: list[DeterministicExtractor] = [RegexEntityExtractor(custom_patterns)]
        if enable_salience:
            extractors.append(
                SalienceEntityExtractor(
                    SalienceConfig(
                        threshold=salience_threshold,
                        max_entities=salience_max_entities,
                    )
                )
            )
        return cls(*extractors)
