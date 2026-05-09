.PHONY: help install test lint format type-check coverage build clean docs

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

test: ## Run tests
	uv run pytest

coverage: ## Run tests with coverage report
	uv run pytest --cov --cov-report=term-missing --cov-report=html

lint: ## Run all linters and checks
	uv run pre-commit run --all-files

format: ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .

type-check: ## Run type checking with mypy
	uv run mypy src/

tox: ## Run tests across Python versions
	uv run tox -p auto

commit: ## Interactive commit with commitizen
	uv run cz commit

bump: ## Bump version with commitizen
	uv run cz bump

build: ## Build package
	uv build

docs: ## Generate API documentation
	uv run python scripts/generate_api_docs.py

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .tox/
