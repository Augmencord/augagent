# Contributing to AugAgent

First off, thank you for considering contributing to `augagent`! 

## Development Setup

1. Fork and clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it and install dev dependencies: `pip install -e .[dev,test]`
4. Ensure tests pass before submitting a PR: `pytest tests/`

## Pull Request Process

1. Ensure your branch is named appropriately (`feature/xxx` or `fix/xxx`).
2. Add tests for your changes.
3. Update the `CHANGELOG.md` with a note of your changes.
4. **Never** push directly to `main`. Always open a Pull Request.
5. Wait for maintainers to review and merge your PR.
