---
description: "Use when writing or modifying Python source files. Covers Python 3.11+ standards, standard package layout, pyproject.toml patterns, ruff linting, mypy type checking, and modern idioms."
applyTo: "**/*.py"
---

# Python Standards

## Version Target

- Python 3.11+ (project minimum)
- Use features available in 3.11+: `match` statements, `X | Y` union syntax, `ParamSpec`, `ExceptionGroup`

## Package Layout

- Standard package layout: `src/{package_name}/`
- Build system: `hatchling` with all configuration in `pyproject.toml`
- No `setup.cfg`, `setup.py`, or `requirements.txt` — everything in `pyproject.toml`

> For project structure patterns and development workflow, see the **python-dev** skill.

## Type System

- Type hints required on function parameters and return types
- Use `X | Y` union syntax (not `Union[X, Y]`)
- Use `list[str]` lowercase generics (not `List[str]`)
- Run `mypy` (strict mode) to validate — `make type-check` or `uv run mypy src/`

## Modern Idioms

- `pathlib.Path` over `os.path` for all file operations
- `dataclasses.dataclass` for data containers where appropriate
- Structural pattern matching (`match`/`case`) over chained `if`/`elif` for 3+ conditions
- F-strings for all string formatting — no `.format()` or `%`
- `itertools`, `functools` for functional patterns

## Structure

- One class per file unless tightly coupled
- No `__all__` — it adds maintenance overhead without benefit for application code; use `per-file-ignores` in ruff for `__init__.py` re-exports
- Private functions prefixed with `_`
- Constants in `UPPER_SNAKE_CASE` at module level

## Imports

- Never use `from __future__ import annotations` — Python 3.11+ native syntax (`X | Y`, lowercase generics) is sufficient
- All imports must be at the top of the file; never define imports inside functions or methods

## Code Quality

- Lint & format with `ruff` (line-length 88) — run `make lint` or `uv run ruff check src/ tests/`
- Auto-format: `make format` or `uv run ruff format src/ tests/`
- Pre-commit hooks run `ruff`, `mypy`, and `commitizen` automatically on commit

## Error Handling

- Custom exception classes inheriting from domain-specific bases
- Never catch bare `Exception` — always catch specific types

## Mandatory Validation

After every code change, run quality checks per the **python-dev** skill. Do not consider the task complete while lint or tests are failing.
