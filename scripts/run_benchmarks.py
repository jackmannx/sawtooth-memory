#!/usr/bin/env python3
"""CLI entrypoint for the Sawtooth benchmark suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "harness":
        from benchmarks.harness import main as harness_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        harness_main()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "micro":
        extra = sys.argv[2:]
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "benchmarks/micro",
            "--benchmark-only",
            *extra,
        ]
        raise SystemExit(subprocess.call(cmd, cwd=ROOT))

    if len(sys.argv) > 1 and sys.argv[1] == "scenarios":
        extra = sys.argv[2:]
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "benchmarks/scenarios",
            "-m",
            "integration_benchmark",
            *extra,
        ]
        raise SystemExit(subprocess.call(cmd, cwd=ROOT))

    if len(sys.argv) > 1 and sys.argv[1] == "all":
        extra = sys.argv[2:]
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "benchmarks",
            *extra,
        ]
        raise SystemExit(subprocess.call(cmd, cwd=ROOT))

    print("Sawtooth Memory Benchmark Suite")
    print()
    print("Usage:")
    print("  python scripts/run_benchmarks.py harness [--turns N] [--mode mock|live]")
    print("  python scripts/run_benchmarks.py micro   [--benchmark-only]")
    print("  python scripts/run_benchmarks.py scenarios")
    print("  python scripts/run_benchmarks.py all")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
