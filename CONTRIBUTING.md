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

---

## API stability

Public API guarantees are documented in [STABILITY.md](STABILITY.md). Changes to
`sawtooth_memory.__all__` require a CHANGELOG entry and, if breaking, a major
version bump per semver.

---

## Releasing

Version numbers must stay in sync across `pyproject.toml` and
`sawtooth_memory.__init__.py` (enforced by `tests/test_public_api.py`).

1. Ensure `main` is green (lint, pytest on 3.11–3.13, build smoke).
2. Move `[Unreleased]` entries in `CHANGELOG.md` to a dated version section.
3. Bump `version` in `pyproject.toml` and `__version__` in `sawtooth_memory/__init__.py`.
4. Open a PR titled `Release X.Y.Z` and merge to `main`.
5. Tag the release and push:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
6. The [Release workflow](.github/workflows/release.yaml) builds and publishes to PyPI.

### PyPI trusted publishing (one-time setup)

Configure a [trusted publisher](https://docs.pypi.org/trusted-publishers/) on
PyPI for this repository:

- **Owner:** `jackmannx`
- **Repository:** `sawtooth-memory`
- **Workflow:** `release.yaml`
- **Environment name:** `pypi` (matches the GitHub Actions environment)

For release candidates, tag `vX.Y.Zrc1` and verify on TestPyPI before the final tag.

### Post-release checklist

- Confirm `pip install sawtooth-memory==X.Y.Z` works in a clean virtualenv.
- Create a GitHub Release with notes from `CHANGELOG.md`.
- Update `SECURITY.md` if the supported version line changed.
