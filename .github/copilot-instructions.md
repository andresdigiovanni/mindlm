# Project Guidelines

## Stack & Runtime
- Python >=3.11, standard package layout: `src/{package_name}/`
- Build: `hatchling` with `pyproject.toml` (single config file for all tooling)
- Package manager: `uv` — dependencies managed via `pyproject.toml` `[dependency-groups]`
- Environment: `uv sync` to install, `uv run` to execute commands

## Quality
- Lint & format: `ruff` (line-length 88) — run with `make lint` or `uv run ruff check src/ tests/`
- Type checking: `mypy` (strict mode) — run with `make type-check`
- Tests: `uv run pytest` (quick) or `tox` for multi-version (py311, py312, py313)
- Pre-commit hooks: `ruff`, `mypy`, `commitizen` — auto-run on commit
- Standards: see `instructions/` for Python and test rules

## Git
- Commit format: conventional commits via `commitizen` — run `uv run cz commit`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Releases: `uv run cz bump` — auto-updates version, CHANGELOG, and tags
