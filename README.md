![mindlm](assets/images/logo.png)

A local, configurable RAG (Retrieval-Augmented Generation) platform built for private, Docker-first deployments. Ingest documents, search them semantically, and ask questions — all using local models (Ollama + HuggingFace) and Qdrant as the vector store. Exposed as both a REST API and an MCP server.

---

## Features

- **YAML-driven configuration** — pipelines, models, chunking, retrieval, and reranking all declared in a single config file
- **Local models** — Ollama for generation, HuggingFace sentence-transformers for embeddings (downloaded at runtime)
- **Qdrant vector store** — multi-tenant via named collections; supports vector and hybrid BM25 (RRF) retrieval
- **Document ingestion** — PDF, HTML, Markdown, DOCX, PPTX, PNG, JPEG; raw, structured, and OCR (Surya) parsing strategies
- **Incremental sync** — detects changes via Qdrant payload hashes; full reingestion also supported
- **Reranking** — cross-encoder or MMR, configurable and optional
- **REST API** — FastAPI with 6 endpoints (health, collections, ingest, search, ask)
- **MCP server** — 5 tools over stdio transport for LLM agent integration
- **Docker Compose** — all services (api, mcp, ollama, qdrant) run with a single command
- **Path traversal protection** — `ingestion.allowed_base_dir` restricts which directories can be ingested

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- A copy of the config file (see [Configuration](#configuration))

### Deploy

```bash
# 1. Clone the repository
git clone https://github.com/username/mindlm.git
cd mindlm

# 2. Configure the platform
cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml to set your models, paths, and preferences

# 3. Start all services
docker compose up
```

Services started:
- **api** — REST API at `http://localhost:8000`
- **mcp** — MCP server (stdio)
- **qdrant** — vector store at `http://localhost:6333`
- **ollama** — LLM runtime at `http://localhost:11434`

---

## Configuration

Copy `configs/config.example.yaml` to `configs/config.yaml` and adjust. The file is mounted as a volume into the containers — no rebuild required for config changes.

Key settings:

| Section | What it controls |
|---|---|
| `llm` | Ollama model name and base URL |
| `embeddings` | HuggingFace model name |
| `qdrant` | Connection URL and collection defaults |
| `ingestion` | Source types, parsing strategy, deduplication, `allowed_base_dir` |
| `chunking` | Strategy (fixed / sliding / semantic) and parameters |
| `retrieval` | Mode (vector or hybrid BM25 via RRF), top-k |
| `reranking` | Reranker type (cross-encoder / MMR) and whether it's enabled |

> **Security**: `ingestion.allowed_base_dir` restricts which host paths can be submitted for ingestion. Set this to the narrowest directory that covers your document sources.

---

## REST API

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Aggregate healthcheck (qdrant, ollama, embeddings) |
| `GET` | `/collections` | List all Qdrant collections |
| `POST` | `/ingest/sync` | Incremental document sync (only changed files) |
| `POST` | `/ingest/full` | Full reingestion — drops and rebuilds the index |
| `POST` | `/search` | Semantic search with optional reranking |
| `POST` | `/ask` | RAG QA — returns answer from Ollama; `503` if Ollama unavailable |

---

## MCP Server

The MCP server exposes 5 tools over stdio transport for use with LLM agents (e.g., Claude Desktop):

| Tool | Description |
|---|---|
| `search_documents` | Semantic search across a collection |
| `ask_rag` | RAG question answering via Ollama |
| `ingest_sync` | Trigger incremental document sync |
| `ingest_full` | Trigger full reingestion |
| `list_collections` | List available Qdrant collections |

To run the MCP server directly (outside Docker):

```bash
mindlm-mcp
```

---

## Scripts

| Command | Description |
|---|---|
| `mindlm-api` | Start the FastAPI REST server |
| `mindlm-mcp` | Start the MCP server (stdio transport) |

These are registered as entry points and available after `uv sync`.

---

## Project Structure

```
src/mindlm/
├── core/           # Domain models, config, embeddings, parsing, chunking,
│                   # vectorstore, retrieval, reranking, generation,
│                   # ingestion, synchronization
├── api/            # FastAPI REST server (routers: health, collections, ingest, search)
└── mcp/            # MCP server (5 tools, stdio transport)
configs/            # config.example.yaml — copy to config.yaml
docker/             # Dockerfiles for api and mcp services
tests/              # Test suite (mirrors src/ structure)
scripts/            # Utility scripts (API doc generation)
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

To install dependencies for local development:

```bash
uv sync
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

---

## License

[MIT](LICENSE)
