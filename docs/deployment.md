# Deployment Guide

MindLM is designed for local-first deployment via Docker Compose.

---

## 🐳 Docker (recommended)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose v2
- `curl` (used by `mindlm.sh`)

### Start all services

```bash
bash mindlm.sh start
```

This starts all services and waits for the API to report healthy. Alternatively:

```bash
docker compose up -d
```

### Services

| Service | Description | Default URL |
|---------|-------------|-------------|
| `api` | REST API | `http://localhost:8000` |
| `mcp` | MCP server (stdio) | — |
| `qdrant` | Vector store | `http://localhost:6333` · dashboard at `/dashboard` |
| `ollama` | LLM runtime | `http://localhost:11434` |
| `langfuse` | Observability UI | `http://localhost:3000` |
| `langfuse-db` | Langfuse Postgres backend | — |

---

## 🤖 Ollama setup

Pull the model configured in `configs/config.yaml` (`llm.model`, default: `gemma4`):

```bash
docker exec -it mindlm-ollama-1 ollama pull gemma4
```

Or from the host if Ollama is also installed locally:

```bash
ollama pull gemma4
```

To use a different model, update `llm.model` in `configs/config.yaml` — no rebuild required.

---

## 🔐 Security

Before deploying in any shared or production environment:

1. **Change Langfuse secrets** in `docker-compose.yml`:
   - `NEXTAUTH_SECRET`
   - `SALT`
   - `LANGFUSE_INIT_USER_PASSWORD`
   - `POSTGRES_PASSWORD`

2. **Set `ingestion.allowed_base_dir`** in `configs/config.yaml` to the narrowest directory covering your document sources (default: `/data`).

---

## 📦 MCP Server

The `mcp` service exposes 5 tools over stdio transport for LLM agent integration:

| Tool | Description |
|------|-------------|
| `search_documents` | Semantic search over indexed documents |
| `ask_rag` | RAG question-answering |
| `ingest_sync` | Incremental document synchronization |
| `ingest_full` | Full re-ingest (drops and rebuilds index) |
| `list_collections` | List all Qdrant collections |

To use with an MCP-compatible client, configure it to run the `mcp` container or the `mindlm-mcp` entry point.

---

## ⚡ GPU support

Embedding model inference and Ollama both support GPU acceleration:

- CUDA-capable GPU: add a `deploy.resources.reservations.devices` section to the relevant services in `docker-compose.yml`
- CPU fallback is available and works out of the box

---

## 🚀 Production recommendations

- Use `hybrid` retrieval for best recall on mixed-vocabulary corpora
- Disable HyDE and query decomposition for latency-sensitive workloads
- Enable cross-encoder reranking selectively (adds one model forward pass per candidate)
- Tune `chunking.chunk_size` per dataset — smaller chunks improve precision, larger improve context
- Set `observability.enabled: true` and review Langfuse traces to identify bottlenecks
