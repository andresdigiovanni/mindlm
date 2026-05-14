# mindlm

A Python project with cookiecutter structure, featuring automated testing, code quality checks, and version management.

## Table of Contents

- [mindlm](#mindlm)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Getting Started](#getting-started)
  - [Development](#development)
    - [Testing](#testing)
    - [Code Quality](#code-quality)
    - [Making Commits](#making-commits)
    - [Creating Releases](#creating-releases)
  - [Dependencies](#dependencies)
  - [Building \& Publishing](#building--publishing)
  - [Quick Reference](#quick-reference)

---

## Requirements

- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package manager
- **(Optional)** Python 3.12 and 3.13 for multi-version testing with tox

---

## Getting Started

```bash
# Install dependencies
uv sync

# Install git hooks
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# Verify installation
uv run pytest
```

---

## Development

### Testing

**Quick testing:**
```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov

# Specific file
uv run pytest tests/mindlm/core/test_models.py

# Specific test
uv run pytest tests/mindlm/core/test_models.py::TestSyncResult::test_sync_result_defaults

# Verbose output
uv run pytest -v
```

**Coverage reports:**
```bash
# Generate HTML coverage report
uv run pytest --cov

# Open in browser
open htmlcov/index.html
```

**Multi-version testing (Python 3.11, 3.12, 3.13):**
```bash
# Test all versions
uv run tox

# Test specific version
uv run tox -e py311

# With coverage
uv run tox -e coverage

# Parallel execution
uv run tox -p auto
```

---

### Code Quality

**Pre-commit hooks** (run automatically on commit):
- **ruff** - Linting and formatting
- **mypy** - Type checking
- **commitizen** - Commit message validation

**Manual checks:**
```bash
# Run all checks
uv run pre-commit run --all-files

# Run on staged files
uv run pre-commit run

# Update hooks
uv run pre-commit autoupdate
```

**Bypass hooks (emergency only):**
```bash
git commit --no-verify -m "emergency fix"
```

---

### Making Commits

**Interactive (recommended):**
```bash
git add .
uv run cz commit
```

**Manual:**
```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug"
git commit -m "docs: update documentation"
```

**Commit types:**

| Type | Description | Version Bump |
|------|-------------|--------------|
| `feat:` | New feature | Minor (0.1.0 → 0.2.0) |
| `fix:` | Bug fix | Patch (0.1.0 → 0.1.1) |
| `feat!:` / `BREAKING CHANGE:` | Breaking change | Major (0.1.0 → 1.0.0) |
| `docs:` | Documentation | None |
| `style:` | Formatting | None |
| `refactor:` | Code refactoring | None |
| `test:` | Tests | None |
| `chore:` | Maintenance | None |

---

### Creating Releases

```bash
# Preview release
uv run cz bump --dry-run

# Create release
uv run cz bump

# Push release
git push && git push --tags
```

This automatically:
- ✅ Analyzes commits since last tag
- ✅ Calculates new version
- ✅ Updates `pyproject.toml`
- ✅ Updates `CHANGELOG.md`
- ✅ Creates git commit and tag

**Manual version control:**
```bash
# Force specific bump
uv run cz bump --increment MAJOR  # → 1.0.0
uv run cz bump --increment MINOR  # → 0.2.0
uv run cz bump --increment PATCH  # → 0.1.1

# Pre-release
uv run cz bump --prerelease alpha  # → 0.1.1a0

# View current version
uv run cz version --project
```

---

## Dependencies

**Add:**
```bash
# Production
uv add requests

# Development
uv add --dev pytest-mock

# Multiple
uv add httpx pydantic
```

**Remove:**
```bash
uv remove <package-name>
```

**Update:**
```bash
# All dependencies
uv lock --upgrade && uv sync

# Specific package
uv lock --upgrade-package requests && uv sync

# List outdated
uv pip list --outdated
```

---

## Building & Publishing

**Build:**
```bash
# Wheel + source
uv build

# Wheel only
uv build --wheel

# Source only
uv build --sdist
```

**Test locally:**
```bash
uv pip install dist/mindlm-*.whl
mindlm
```

**Publish to PyPI:**
```bash
# Build
uv build

# Test on TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/

# Publish to PyPI
uv publish
```

---

## Quick Reference

```bash
# Development
uv run pytest                            # Run tests
uv run pytest --cov                      # Tests with coverage
uv run pre-commit run --all-files        # Run all quality checks

# Commits & Releases
uv run cz commit                         # Interactive commit
uv run cz bump                           # Create release

# Dependencies
uv add <package>                         # Add dependency
uv lock --upgrade && uv sync             # Update all

# Build
uv build                                 # Build package
uv run mindlm                  # Run application
```
