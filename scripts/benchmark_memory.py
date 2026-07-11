#!/usr/bin/env python3
"""
Comparative performance benchmark: Sawtooth vs blocking summary memory.

For the full benchmark suite, use:
    python scripts/run_benchmarks.py all
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.harness import main

if __name__ == "__main__":
    main()
