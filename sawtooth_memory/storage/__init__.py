"""
storage — Persistence adapters for Sawtooth Memory.

Public surface:

    BaseStorageAdapter      — abstract session/pool persistence contract
    RedisStorageAdapter     — high-speed ephemeral Redis backend (no L3 vectors)
    PostgresStorageAdapter  — durable Postgres + pgvector L3 backend
    SemanticChunkResult     — result row from L3 similarity search
"""

from .base import BaseStorageAdapter
from .postgres_adapter import PostgresStorageAdapter
from .redis_adapter import RedisStorageAdapter
from .semantic import SemanticChunkResult

__all__ = [
    "BaseStorageAdapter",
    "RedisStorageAdapter",
    "PostgresStorageAdapter",
    "SemanticChunkResult",
]
