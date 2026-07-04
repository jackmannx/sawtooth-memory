"""Event types for Sawtooth telemetry - using dataclasses for speed (no Pydantic overhead)."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal
from uuid import uuid4


def _utcnow() -> datetime:
    """Return UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


@dataclass
class SawtoothEvent:
    """Base event."""

    event_type: str = "base"
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=_utcnow)
    session_id: Optional[str] = None
    cycle_id: Optional[str] = None


@dataclass
class L1EvictionEvent(SawtoothEvent):
    event_type: Literal["l1.eviction"] = "l1.eviction"
    tokens_evicted: int = 0
    messages_evicted: int = 0
    tokens_remaining_l1: int = 0
    evicted_message_ids: List[str] = field(default_factory=list)
    trigger: Literal["soft_limit_exceeded", "manual", "hard_limit_forced"] = (
        "soft_limit_exceeded"
    )


@dataclass
class EntityAnchoredEvent(SawtoothEvent):
    event_type: Literal["l1_5.entity_anchored"] = "l1_5.entity_anchored"
    entity_key: str = ""
    entity_value: Any = None
    operation: Literal["insert", "update", "delete"] = "insert"
    source_message_id: Optional[str] = None


@dataclass
class L2SummaryGeneratedEvent(SawtoothEvent):
    event_type: Literal["l2.summary_generated"] = "l2.summary_generated"
    summary_text: str = ""
    compressed_message_count: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    provider: str = ""
    model: str = ""
    compression_duration_ms: int = 0
    fallback_used: bool = False


@dataclass
class CompressionCycleCompleteEvent(SawtoothEvent):
    event_type: Literal["compression.cycle_complete"] = "compression.cycle_complete"
    l1_tokens_evicted: int = 0
    l1_5_entities_retained: Dict[str, Any] = field(default_factory=dict)
    l2_summary_generated: str = ""
    messages_compressed: int = 0
    final_l1_tokens: int = 0
    total_duration_ms: int = 0


@dataclass
class CompressionCycleFailedEvent(SawtoothEvent):
    event_type: Literal["compression.cycle_failed"] = "compression.cycle_failed"
    error_type: str = ""
    error_message: str = ""
    fallback_triggered: bool = False


@dataclass
class SoftLimitReachedEvent(SawtoothEvent):
    event_type: Literal["monitor.soft_limit_reached"] = "monitor.soft_limit_reached"
    current_tokens: int = 0
    soft_limit: int = 0
    hard_limit: int = 0


@dataclass
class HardLimitReachedEvent(SawtoothEvent):
    event_type: Literal["monitor.hard_limit_reached"] = "monitor.hard_limit_reached"
    current_tokens: int = 0
    soft_limit: int = 0
    hard_limit: int = 0


@dataclass
class CompressionCycleStartEvent(SawtoothEvent):
    event_type: Literal["compression.cycle_started"] = "compression.cycle_started"
    current_l1_tokens: int = 0
    chunk_size: int = 0


@dataclass
class L3VectorIndexedEvent(SawtoothEvent):
    event_type: Literal["l3.vector_indexed"] = "l3.vector_indexed"
    chunks_indexed: int = 0
    total_chunks: int = 0
    source_chars: int = 0
    embedding_backend: str = ""
    embedding_model: str = ""
