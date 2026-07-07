# Contributing to Sawtooth Memory

Thank you for your interest in contributing to `sawtooth-memory`.

To maintain a high standard of code quality, performance, and structural integrity, please review and follow these guidelines before setting up your environment or submitting any changes.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md) at all times. Please report any violations or unacceptable behavior privately to **i.am.the.godddddddd@gmail.com**.

---

## Development Setup

We use `pyproject.toml` to manage package configurations and dependencies. To get started with development:

1. **Fork and Clone** the repository to your local machine.
2. Navigate to the root directory and install the package along with its development dependencies in editable mode:
   ```bash
   pip install -e ".[dev,langgraph,redis]"
   ```
3. **Install pre-commit hooks** (recommended):
   ```bash
   pip install pre-commit ruff
   pre-commit install
   ```
   This enables automatic formatting and linting via ruff, mypy, and basic file checks before each commit.
