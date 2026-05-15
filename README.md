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
- **Query processing** — 6 configurable pre-retrieval techniques (rewriting, expansion, HyDE, multi-query, decomposition, step-back) that improve recall; combinable
- **REST API** — FastAPI with 6 endpoints (health, collections, ingest/sync, ingest/full, search, ask)
- **MCP server** — 5 tools over stdio transport for LLM agent integration
- **Docker Compose** — all services (api, mcp, ollama, qdrant) run with a single command
- **Path traversal protection** — `ingestion.allowed_base_dir` restricts which directories can be ingested
- **Observability (opt-in)** — Langfuse tracing for the full RAG pipeline; all stages (`search`/`ask` → retrieve → query processing, embedding, reranking, generation) are instrumented; disabled by default

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

# 3. Add your documents
mkdir data
# Copy documents into data/ — this folder is mounted as /data inside the containers.
# To use a different path, update the volume in docker-compose.yml and
# set ingestion.allowed_base_dir in config.yaml accordingly.

# 4. Start all services
docker compose up
```

Services started:
- **api** — REST API at `http://localhost:8000`
- **mcp** — MCP server (stdio)
- **qdrant** — vector store at `http://localhost:6333` · dashboard at `http://localhost:6333/dashboard`
- **ollama** — LLM runtime at `http://localhost:11434`

To also start self-hosted Langfuse observability:

```bash
docker compose --profile langfuse up
```

Langfuse UI: `http://localhost:3000` (default credentials: `admin@example.com` / `changeme` — change before production).

---

## Configuration

Copy `configs/config.example.yaml` to `configs/config.yaml` and adjust. The file is mounted as a volume into all containers — no rebuild is required after changes.

### Quick Start

**Option A — Interactive wizard** (recommended):
```bash
./config-wizard.sh
# or via mindlm.sh:
bash mindlm.sh config-wizard
```

**Option B — Use a profile preset**:
```bash
cp configs/profiles/balanced.yaml configs/config.yaml   # recommended default
# cp configs/profiles/minimal.yaml configs/config.yaml  # lightweight alternative
# cp configs/profiles/full.yaml configs/config.yaml     # maximum quality
```

**Option C — Manual copy**:
```bash
cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml to set your models, paths, and preferences
```

### Profiles

| Profile | Use Case | Chunking | Embedding Model | Retrieval | Reranking | Query Processors |
|---------|----------|----------|-----------------|-----------|-----------|------------------|
| `minimal` | Quick trials, low resource | fixed/512 | all-MiniLM-L6-v2 (384d) | vector, top 5 | disabled | none |
| `balanced` | Most production use cases | recursive/500 | bge-small-en-v1.5 (384d) | vector, top 10 | cross-encoder | rewriting |
| `full` | Maximum quality | semantic/500 | bge-large-en-v1.5 (1024d) | hybrid, top 10 | cross-encoder | all |

See [`configs/profiles/`](configs/profiles/) for ready-to-use YAML files.

---

### Infrastructure

### `app`

General platform settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | `"local-rag"` | Display name used in logs |

---

### `llm`

Controls the language model used for answer generation.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | `"ollama"` | `"ollama"` | LLM provider. Only `ollama` is supported |
| `model` | string | `"llama3"` | Ollama model name (must be pulled with `ollama pull <model>`) |
| `base_url` | string | `"http://ollama:11434"` | Ollama service URL. Use `http://localhost:11434` for local dev |
| `temperature` | float | `0.7` | Sampling temperature. `0.0` = deterministic, `1.0` = most random |
| `max_tokens` | int (> 0) | `1024` | Maximum tokens to generate per response |

---

### `embeddings`

Controls the embedding model used to vectorize documents and queries.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | `"huggingface"` | `"huggingface"` | Embeddings provider. Only `huggingface` is supported |
| `model` | string | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace model name. Downloaded at first run into the `hf_cache` volume |
| `dimensions` | int (> 0) | `384` | Output vector dimensions. **Must match the model's actual output size** |

Common model / dimension pairs:

| Model | Dimensions |
|-------|-----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `sentence-transformers/all-mpnet-base-v2` | 768 |
| `BAAI/bge-large-en-v1.5` | 1024 |

---

### `vector_store`

Controls the Qdrant vector database connection.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | `"qdrant"` | `"qdrant"` | Vector store provider. Only `qdrant` is supported |
| `mode` | `"local"` \| `"cloud"` | `"local"` | `local` connects to a self-hosted instance; `cloud` uses Qdrant Cloud |
| `host` | string | `"qdrant"` | Qdrant hostname. Use `localhost` for local dev outside Docker |
| `port` | int | `6333` | Qdrant HTTP port |
| `collection` | string | `"documents"` | Default collection name (created automatically on first ingest) |
| `api_key` | string \| null | `null` | API key for Qdrant Cloud. Leave unset for local mode |

---

### Ingestion Pipeline

### `ingestion`

Controls which files are accepted and how they are parsed.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `source_type` | list | all types | File types to accept. Subset of: `pdf`, `html`, `markdown`, `png`, `jpeg`, `pptx`, `docx` |
| `parsing_strategy` | `"raw"` \| `"structured"` \| `"ocr"` | `"structured"` | See below |
| `deduplication` | bool | `true` | Skip unchanged files on incremental sync (hash-based) |
| `allowed_base_dir` | string | `"/data"` | **Security boundary**: only paths under this directory are accepted. Set to the narrowest directory covering your document sources |

**Parsing strategies:**

| Strategy | Description |
|----------|-------------|
| `raw` | Extract plain text with no layout analysis |
| `structured` | Preserve headings, lists, and tables (recommended for most documents) |
| `ocr` | Run OCR via Surya for scanned images and PDFs with embedded images. Requires the `surya-ocr` optional dependency |

---

### `chunking`

Controls how documents are split into chunks before indexing.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strategy` | `"fixed"` \| `"sliding"` \| `"semantic"` \| `"recursive"` | `"fixed"` | See below |
| `chunk_size` | int (> 0) | `500` | Target chunk size in tokens (fixed/sliding/recursive) or characters (semantic) |
| `overlap` | int (≥ 0) | `50` | Token/character overlap between consecutive chunks |
| `semantic_model` | string \| null | `null` | HuggingFace model used for semantic splitting. **Required when `strategy: semantic`** |
| `parent_chunk_size` | int \| null | `null` | When set, enables parent-document retrieval: small child chunks are indexed for precise retrieval; results are replaced with their parent content before returning. Must be greater than `chunk_size` |
| `separators` | list[string] | `["\n\n", "\n", ". ", " ", ""]` | Separator hierarchy for recursive chunking. Tried in order; falls back to hard character splitting |

**Chunking strategies:**

| Strategy | Description |
|----------|-------------|
| `fixed` | Split into chunks of exactly `chunk_size` tokens with `overlap` |
| `sliding` | Sliding window: similar to fixed but every chunk shifts by `chunk_size − overlap` |
| `semantic` | Group sentences by semantic similarity using `semantic_model`; `chunk_size` is the upper bound |
| `recursive` | Try separators in order (`\n\n` → `\n` → `. ` → ` ` → character), recursing into oversized pieces. Best for structured text with paragraphs and sentences |

---

### Retrieval Pipeline

### `retrieval`

Controls how documents are retrieved for a given query.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strategy` | `"vector"` \| `"hybrid"` | `"vector"` | See below |
| `top_k` | int (> 0) | `5` | Number of documents to return before optional reranking |

**Retrieval strategies:**

| Strategy | Description |
|----------|-------------|
| `vector` | Dense vector similarity search only |
| `hybrid` | Combines dense vector search with sparse BM25 keyword search via Reciprocal Rank Fusion (RRF). Recommended for most use cases |

---

### `reranking`

Optional post-retrieval reranking step to improve result relevance.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable or disable reranking |
| `method` | `"cross_encoder"` \| `"mmr"` \| null | `null` | Reranking algorithm. Required when `enabled: true` |
| `model` | string \| null | `null` | HuggingFace model used for cross-encoder scoring. Required when `method: cross_encoder` |

**Reranking methods:**

| Method | Description |
|--------|-------------|
| `cross_encoder` | Uses a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score query–document pairs. More accurate but slower |
| `mmr` | Maximal Marginal Relevance: re-ranks by balancing relevance and diversity. No extra model needed |

> **Security**: `ingestion.allowed_base_dir` restricts which host paths can be submitted for ingestion. Set this to the narrowest directory that covers your document sources.

---

### Query Processing Pipeline

### `query_processing`

Optional pre-retrieval pipeline that transforms the query into one or more alternative representations to improve recall. All processors are disabled by default. Multiple processors can be enabled simultaneously — the dispatcher fans out, merges all result sets, and deduplicates before reranking.

| Processor | Config key | Description |
|-----------|------------|-------------|
| Query Rewriting | `rewriting` | Reformulates the query for better semantic search alignment |
| Query Expansion | `expansion` | Adds synonyms and related terms to broaden the search surface |
| HyDE | `hyde` | Generates a hypothetical answer passage and embeds that instead of the raw query |
| Multi-Query | `multi_query` | Generates N rephrased variants, retrieves for each, then merges and deduplicates |
| Query Decomposition | `decomposition` | Breaks a complex query into focused sub-questions, retrieves for each |
| Step-Back Prompting | `step_back` | Generates a more abstract version of the query for wider recall |

Per-processor config keys:

| Key | Type | Default | Applies to |
|-----|------|---------|------------|
| `enabled` | bool | `false` | all processors |
| `num_variants` | int (2–10) | `3` | `multi_query` only |
| `max_subqueries` | int (2–10) | `4` | `decomposition` only |

Example — enable query rewriting:

```yaml
query_processing:
  rewriting:
    enabled: true
```

> Processors are composed additively: if multiple processors are enabled, each generates its own query variants; results are merged and deduplicated before retrieval.

---

### `observability`

Optional Langfuse tracing for the RAG pipeline. When enabled, `search` and `ask` requests are traced end-to-end: retrieval, query processing, embedding, reranking, and generation each appear as a named span.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable or disable Langfuse tracing. All `@observe` decorators are transparent no-ops when disabled |
| `public_key` | string | `"pk-lf-local-dev"` | Langfuse project public key |
| `secret_key` | string | `"sk-lf-local-dev"` | Langfuse project secret key |
| `host` | string | `"http://langfuse:3000"` | Langfuse host URL. Use `http://localhost:3000` for local dev outside Docker; `https://cloud.langfuse.com` for Langfuse Cloud |
| `flush_at` | int | `15` | Events batched before sending |
| `flush_interval` | float | `0.5` | Maximum seconds before flushing a batch |

Self-hosted Langfuse runs as an optional Docker Compose profile — see [Quick Start](#quick-start) for the command.

---

## RAG Techniques

### Chunking strategies

| Strategy | When to use |
|----------|-------------|
| `fixed` | Uniform documents; simplest baseline |
| `sliding` | Overlapping windows; reduces context loss at chunk boundaries |
| `semantic` | Variable-length chunks that respect semantic boundaries; best for dense narrative text |
| `recursive` | Structured text with paragraphs and sentences; tries progressively finer separators and recurses into oversized pieces |

**Parent-document retrieval** (`parent_chunk_size`): Index small child chunks for precise vector matching, but surface the parent chunk (broader context) in results. Set `parent_chunk_size` to an integer greater than `chunk_size`. Child chunks are stored with `parent_id` and `parent_content` in the Qdrant payload; retrieval automatically substitutes parent content before returning results. Use this when fine retrieval granularity and full-context answers are both required.

### Query processing techniques

Query processing runs before retrieval to transform the incoming query into alternative representations, improving recall without changing the retrieval or reranking configuration.

| Technique | Config key | What it does |
|-----------|------------|---------------|
| Query Rewriting | `rewriting` | Reformulates the query for better semantic alignment |
| Query Expansion | `expansion` | Adds synonyms and related terms to broaden the search surface |
| HyDE | `hyde` | Embeds a generated hypothetical answer passage instead of the raw query |
| Multi-Query | `multi_query` | Generates N rephrased variants, retrieves for each, merges results |
| Query Decomposition | `decomposition` | Splits a complex query into focused sub-questions, retrieves for each |
| Step-Back Prompting | `step_back` | Abstracts the query to a higher level for wider recall |

Multiple techniques can be active at once. The `QueryProcessorDispatcher` fans out to all enabled processors and deduplicates the combined result set before passing it to reranking.

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

### Multi-tenant collections

Each request to `/search`, `/ask`, `/ingest/sync`, and `/ingest/full` accepts an optional `"collection"` field. When supplied, it targets that Qdrant collection instead of the default in `config.yaml`. Use this to maintain independent knowledge bases per project or team.

```json
{ "query": "what is RAG?", "collection": "tech-notes" }
```

Collections are created automatically on first ingest. List existing collections with `GET /collections`.

### Examples

**GET /health**
```bash
curl http://localhost:8000/health
# {"status": "ok", "services": {"qdrant": "ok", "ollama": "ok", "embeddings": "ok"}}
```

**GET /collections**
```bash
curl http://localhost:8000/collections
# ["documents", "tech-notes"]
```

**POST /ingest/sync**
```bash
curl -X POST http://localhost:8000/ingest/sync \
  -H "Content-Type: application/json" \
  -d '{"paths": ["/data/docs/report.pdf"]}'
# {"added": 12, "updated": 0, "skipped": 3, "errors": []}
```

**POST /search**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is RAG", "top_k": 3}'
```

**POST /ask**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the benefits of hybrid retrieval?"}'
```

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

### Claude Desktop integration

Add to `claude_desktop_config.json` (`~/Library/Application Support/Claude/` on macOS, `%APPDATA%\Claude\` on Windows):

**Via Docker (recommended — services must be running):**
```json
{
  "mcpServers": {
    "mindlm": {
      "command": "docker",
      "args": ["exec", "-i", "mindlm-mcp-1", "mindlm-mcp"]
    }
  }
}
```

**Via local install:**
```json
{
  "mcpServers": {
    "mindlm": {
      "command": "uv",
      "args": ["run", "mindlm-mcp"],
      "env": { "CONFIG_PATH": "/absolute/path/to/configs/config.yaml" }
    }
  }
}
```

Restart Claude Desktop after editing the config.

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
```

---

## CLI (`mindlm.sh`)

A bash script to manage the platform from any directory.

> **Windows users:** `mindlm.sh` is a Unix Bash script. On native Windows CMD/PowerShell, use Docker Compose directly: `docker compose up`, `docker compose down`. All `docker compose` commands in this README work on Windows as-is.

### Install

```bash
./mindlm.sh install
source ~/.bashrc   # or open a new shell
```

This creates a symlink at `~/.local/bin/mindlm` and adds it to your `PATH`.

### Commands

| Command | Description |
|---------|-------------|
| `mindlm start` | Start all services and wait for the API to become healthy |
| `mindlm stop` | Stop all services |
| `mindlm status` | Show container status |
| `mindlm health` | Print API health JSON |
| `mindlm collections` | List all Qdrant collections |
| `mindlm search "<query>"` | Search the knowledge base |
| `mindlm ask "<question>"` | Ask a question (RAG) |
| `mindlm ingest <path>...` | Incremental document sync |
| `mindlm ingest-full <path>...` | Full document re-index |
| `mindlm uninstall` | Remove the `~/.local/bin/mindlm` symlink |

### Options

`search` accepts:
- `--top-k N` — number of results (default: `5`)
- `--collection NAME` — restrict to a specific collection

`ask` accepts:
- `--collection NAME` — restrict to a specific collection

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINDLM_API_BASE` | `http://localhost:8000` | Override the API base URL |
| `NO_COLOR` | unset | Set to any value to disable colored output |

> Output is pretty-printed with `jq` when available; falls back to raw JSON.

---

## Development

```bash
# Quality
make test       # Run tests
make coverage   # Tests with coverage report (opens htmlcov/)
make lint       # Run all quality checks (ruff + mypy)
make format     # Format and auto-fix code
make tox        # Run tests across Python 3.11, 3.12, 3.13

# Releases
make commit     # Interactive commit with conventional commits
make bump       # Create a new release (bumps version + updates CHANGELOG)
make docs       # Generate API documentation
make build      # Build distributable package
make clean      # Remove Python build artifacts and caches

# Docker
make docker-build   # Build Docker images
make docker-start   # Start all services (docker compose up -d)
make docker-stop    # Stop all services (docker compose down)
make docker-logs    # Follow service logs
make docker-clean   # Remove containers + volumes (destructive)
```

To install dependencies for local development:

```bash
uv sync
```

### Pre-commit hooks

Install hooks (one-time, after `uv sync`):

```bash
uv run pre-commit install
```

Hooks that run automatically on `git commit`:

| Hook | What it does |
|------|-------------|
| `ruff` + `ruff-format` | Lints and formats Python code |
| `mypy` | Type-checks the package |
| `commitizen` | Enforces conventional commit message format |
| `trailing-whitespace`, `end-of-file-fixer` | Basic file hygiene |
| `check-yaml` / `check-toml` / `check-json` | Validates config syntax |

Run all hooks manually:

```bash
uv run pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

---

## License

[MIT](LICENSE)
