---
description: "Use when writing or modifying test files. Enforces edge case coverage, Arrange-Act-Assert pattern, and tox compatibility."
applyTo: "**/test_*.py, **/tests/**/*.py"
---

# Test Standards

## Runner

- Quick runs: `uv run pytest` or `make test`
- With coverage: `uv run pytest --cov` or `make coverage`
- Multi-version (py311, py312, py313): `uv run tox` or `make tox`
- Specific test: `uv run pytest tests/path/test_file.py::TestClass::test_name`
- Dev dependencies defined in `pyproject.toml` `[dependency-groups]` — install with `uv sync`

## Edge Case Coverage (Mandatory)

Every test module must include tests for:
- **Empty inputs**: empty strings, empty lists, empty dicts, `None`
- **Boundary values**: 0, -1, `sys.maxsize`, minimum/maximum valid values
- **Invalid types**: wrong argument types, unexpected data shapes
- **Error paths**: exceptions, timeouts, network failures (mocked)
- **Concurrency**: thread-safety if applicable
- **Large inputs**: performance-sensitive paths with large datasets
- **Unicode/encoding**: special characters, emoji, multi-byte strings where relevant
- **State transitions**: initial → in-progress → final states, repeated calls, idempotent operations

## Conventions

- Follow **Arrange-Act-Assert** pattern with blank line separators
- Test names: `test_should_<expected>_when_<condition>`
- One assertion per test (logical assertion — multiple `assert` for one concept is fine)
- Group tests in classes by feature: `class TestMyServiceProcess:`
- Use `@pytest.mark.parametrize` for testing multiple inputs with the same logic
- Use `pytest` fixtures for shared setup — avoid `setUp`/`tearDown` methods
- Never mock the system under test — only its dependencies
- Use `freezegun` or `time-machine` for time-dependent tests

> For code examples, patterns, and step-by-step procedures, see the **testing** skill.
