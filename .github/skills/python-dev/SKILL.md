---
name: python-dev
description: "Python development workflow. Use when implementing Python features, refactoring code, or working with the standard package layout. Covers environment setup, project templates, and quality check procedures."
---

# Python Development

> For coding rules and standards, see `instructions/python.instructions.md`. This skill covers **procedures and templates only**.

## When to Use

- Implementing new Python features or modules
- Refactoring existing Python code
- Setting up project configuration or dependencies

## Procedure

### 1. Environment Setup

1. Install dependencies: `uv sync`
2. Run commands via: `uv run <command>`
3. Install pre-commit hooks (first time): `uv run pre-commit install`
4. Verify setup: `uv run pytest`

### 2. Project Structure Template

```
{project-name}/
├── src/
│   └── {package_name}/
│       ├── __init__.py
│       ├── main.py
│       └── utils/
│           ├── __init__.py
│           └── module.py
├── tests/
│   ├── conftest.py
│   └── {package_name}/
│       └── utils/
│           └── test_module.py
├── scripts/
├── docs/
├── pyproject.toml
├── Makefile
└── .github/
```

### 3. pyproject.toml Template

```toml
[project]
name = "{project-name}"
requires-python = ">=3.11"

[dependency-groups]
dev = [
    "pytest>=8.4.2",
    "pytest-cov>=5.0.0",
    "pre-commit>=4.0.0",
    "tox>=4.0.0",
    "tox-uv>=1.0.0",
    "commitizen>=4.9.1",
    "ruff>=0.6.9",
    "mypy>=1.13.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{package_name}"]
```

### 5. API Documentation (if applicable)

If the project generates API docs (e.g., via `pdoc` or `mkdocs`):

```bash
make docs       # regenerate from source — never edit output manually
```

Typical setup in `pyproject.toml`:
```toml
[dependency-groups]
docs = ["pdoc>=14.0"]
```

In `Makefile`:
```makefile
docs:
	uv run pdoc src/{package_name} --output-dir docs/api
```

> Only add API doc generation if the project exposes a public library interface. Skip for internal scripts or services.

### 6. Quality Check Commands (run after every change)

These are **mandatory** — always run in order and fix any failures before finishing:

```bash
uv run ruff check src/ tests/   # must pass with no errors
uv run pytest -q                # must pass with no failures
```

Additional checks:

```bash
make lint          # all pre-commit hooks (ruff + mypy)
make format        # auto-fix formatting
make type-check    # mypy strict
make test          # uv run pytest
make coverage      # pytest + HTML coverage report
make tox           # multi-version (py311, py312, py313)
```
