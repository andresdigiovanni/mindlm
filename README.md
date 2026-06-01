![mindlm](assets/images/logo.png)

MindLM is a local-first RAG framework designed to build high-quality document intelligence systems using modular retrieval, reranking, and query enhancement pipelines.

It is built around a fully configurable architecture using Ollama, Qdrant, and HuggingFace embeddings.

---

## ✨ Core Features

### 🧠 Local-first AI stack
- Ollama-based LLM inference
- Fully local deployment (no external APIs required)
- Qdrant vector database

### 📄 Document processing
- Recursive chunking
- Semantic chunking
- Sentence window retrieval
- Parent-child chunking

### 🔎 Retrieval system
- Dense vector search
- Hybrid retrieval (dense + sparse)
- Metadata filtering
- Configurable top-k strategies

### ⚙️ Pipeline control
- Fully YAML-configurable pipeline
- Modular architecture per stage
- Extensible components

### 🚀 Deployment
- Docker support
- MCP Server integration
- Production-ready local deployment

---

## 🧪 Advanced Features

### 🔍 Query enhancement
- Query expansion
- Query rewriting
- Multi-query retrieval
- HyDE (Hypothetical Document Embeddings)
- Step-back prompting
- Query decomposition

### 📊 Advanced retrieval
- Contextual retrieval
- Parent-child retrieval
- Reciprocal Rank Fusion (RRF)

### 🎯 Reranking
- Cross-encoder reranking
- LLM-based reranking

### 🧩 Context optimization
- Context compression
- Contextual summarization

---

## ⚡ Quick Start (5 minutes)

### 1. Start all services

```bash
bash mindlm.sh start
```

Or, to install the CLI to your PATH first:

```bash
bash mindlm.sh install   # installs to ~/.local/bin
mindlm start
```

### 2. Ingest documents

Documents are mounted at `/data` inside the containers (see `docker-compose.yml`):

```bash
mindlm ingest /data
```

### 3. Ask questions

```bash
mindlm ask "What are the key ideas in these documents?"
```

---

## 📦 Configuration

All behavior is controlled via `configs/config.yaml`.

Key sections:

* llm
* embeddings
* vector_store
* ingestion
* chunking
* contextual_retrieval
* retrieval
* reranking
* compression
* query_processing
* iterative_retrieval
* observability

---

## 🧭 Documentation

* [Architecture](docs/architecture.md)
* [Retrieval Techniques](docs/retrieval-techniques.md)
* [Configuration Reference](docs/configuration.md)
* [Deployment Guide](docs/deployment.md)

---

## 🧪 Philosophy

MindLM is designed to:

* Keep everything local-first
* Make retrieval pipelines modular
* Allow full control over RAG behavior
* Expose advanced techniques without hidden complexity

---

## 📌 License

See LICENSE file for details.
