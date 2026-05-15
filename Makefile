.PHONY: help install test lint format type-check coverage tox \
        commit bump build docs \
        docker-build docker-start docker-stop docker-logs docker-clean \
        clean

help: ## Show this help message
	@grep -E '^(##|[a-zA-Z_-]+:.*?## ).*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; \
	       /^## / { printf "\n\033[1m%s\033[0m\n", substr($$0, 4) } \
	       /^[a-zA-Z]/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }'

## Development

install: ## Install dependencies
	uv sync
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

test: ## Run tests
	uv run pytest

lint: ## Run all linters and checks
	uv run pre-commit run --all-files

format: ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .

type-check: ## Run type checking with mypy
	uv run mypy src/

coverage: ## Run tests with coverage report
	uv run pytest --cov --cov-report=term-missing --cov-report=html

tox: ## Run tests across Python versions
	uv run tox -p auto

## Versioning & Release

commit: ## Interactive commit with commitizen
	uv run cz commit

bump: ## Bump version with commitizen
	uv run cz bump

build: ## Build package
	uv build

## Docker

docker-build: ## Build Docker images
	docker compose build

docker-start: ## Start all Docker services
	docker compose up -d

docker-stop: ## Stop all Docker services
	docker compose down

docker-logs: ## Follow Docker service logs
	docker compose logs -f

docker-clean: ## Remove containers and volumes (destructive — deletes all data)
	@echo "WARNING: This will delete all volumes (qdrant_data, ollama_models, hf_cache)."
	@echo "Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	docker compose down -v

## Cleanup

clean: ## Remove Python build artifacts and caches
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .tox/
