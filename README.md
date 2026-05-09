# mindlm

> Short description of your project.

---

## Getting Started

> **First time setup** — run these commands after creating this project from the template.

```bash
# 1. Install dependencies (creates .venv automatically)
uv sync

# 2. Initialize git repository
git init && git add . && git commit -m 'chore: initial commit'

# 3. Set up git hooks (runs ruff, mypy and commit validation automatically)
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# 4. Verify everything works
uv run pytest
uv run pre-commit run --all-files

# 5. Configure your remote
git remote add origin https://github.com/username/mindlm.git
git push -u origin main
```

---

## Development

```bash
make test       # Run tests
make coverage   # Tests with coverage report (opens htmlcov/)
make lint       # Run all quality checks (ruff + mypy)
make format     # Format and auto-fix code
make commit     # Interactive commit with conventional commits
make bump       # Create a new release (bumps version + updates CHANGELOG)
make docs       # Generate API documentation
make build      # Build distributable package
make clean      # Remove build artifacts and caches
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

---

## Project Structure

```
src/mindlm/   # Main package
tests/                  # Test suite (mirrors src/ structure)
scripts/                # Utility scripts
.github/workflows/      # CI/CD (lint, test matrix, release to PyPI)
```

---

## Next Steps

- [ ] Replace the description at the top of this file
- [ ] Update `description` in `pyproject.toml`
- [ ] Add `[project.urls]` to `pyproject.toml` (Homepage, Repository, Issues)
- [ ] Delete the sample code in `src/` and `tests/` and add your own
- [ ] Add production dependencies: `uv add <package>`
- [ ] Set up PyPI trusted publishing if you plan to publish (see `.github/workflows/release.yml`)

---

## License

[MIT](LICENSE)
