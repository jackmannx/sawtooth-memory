"""Benchmark report generation and environment metadata."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sawtooth_memory


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    """Capture reproducibility metadata for a benchmark run."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "sawtooth_memory_version": getattr(sawtooth_memory, "__version__", "unknown"),
        "git_sha": _git_sha(),
    }


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    """Persist a benchmark report as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def merge_report(base: dict[str, Any], **sections: Any) -> dict[str, Any]:
    """Merge named sections into a report payload."""
    merged = dict(base)
    for key, value in sections.items():
        merged[key] = value
    return merged
