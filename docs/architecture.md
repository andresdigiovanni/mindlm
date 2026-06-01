# Architecture

MindLM is composed of two main pipelines plus an observability layer:

1. Ingestion pipeline (document processing)
2. Retrieval pipeline (query answering)
3. Observability (Langfuse tracing)

---

## 🧾 Ingestion pipeline

Documents are transformed into searchable embeddings.

Steps:

1. Document loading and parsing (`raw` / `structured` / `ocr`)
2. Chunking (fixed, sliding, semantic, recursive, or sentence_window)
3. Optional contextual enrichment (LLM-generated chunk context and document summary)
4. Embedding generation (HuggingFace sentence-transformers)
5. Storage in Qdrant (dense vectors + optional sparse BM25 vectors)

### Chunking strategies

- **Recursive** — general-purpose splitting using a separator hierarchy
- **Semantic** — meaning-based segmentation using a sentence-transformer model
- **Sentence window** — indexes individual sentences; stores surrounding sentences in the payload for wider context at retrieval time
- **Parent-document** — small child chunks are indexed for precision; results are expanded to their parent chunk before returning (enabled by setting `chunking.parent_chunk_size`)

### Optional: Contextual retrieval

Enhances chunks using LLM-generated metadata before embedding. Controlled by `contextual_retrieval` in config:

- `chunk_context_enabled`: one LLM call per chunk → `payload["chunk_context"]`
- `document_summary_enabled`: one LLM call per document → `payload["document_summary"]`

Raw chunk text is always embedded unchanged; context fields are metadata only.

---

## 🗄 Storage layer

- Qdrant vector database
- Dense and sparse (BM25) vectors stored per chunk
- Metadata: source path, page number, character offsets, chunk index, contextual fields
- Collection-based document isolation

---

## 🔁 Retrieval pipeline

1. Query processing (optional fan-out: rewriting, expansion, HyDE, multi-query, decomposition, step-back)
2. Retrieval (vector / hybrid BM25+dense)
3. Fusion (Reciprocal Rank Fusion across query variants and retrieval strategies)
4. Context resolution (expand sentence-window or parent-document results)
5. Reranking (cross-encoder, MMR, or LLM-as-reranker)
6. Context compression (optional: LLM extracts relevant sentences)
7. LLM generation (Ollama)

### Iterative retrieval (opt-in)

When `iterative_retrieval.enabled: true`, the pipeline adds a grounding-check loop after each generate step. If the answer is not supported by the retrieved context and a refined query is suggested, retrieval is retried up to `max_iterations` times. Results are accumulated and deduplicated across iterations.

---

## 🔌 Interfaces

- **REST API** (FastAPI): `health`, `collections`, `ingest/sync`, `ingest/full`, `search`, `ask`
- **MCP server** (stdio): `search_documents`, `ask_rag`, `ingest_sync`, `ingest_full`, `list_collections`

---

## 📡 Observability

When `observability.enabled: true`, Langfuse traces cover the full pipeline:

- Query processing stage
- Embedding calls
- Retrieval and fusion
- Reranking
- Generation

Traces are viewable in the Langfuse UI at `http://localhost:3000` (default Docker deployment).

---

## ⚠️ Design principles

- Every stage is modular and replaceable
- Configuration-driven behavior — no code changes needed to swap strategies
- Security boundary enforced at ingestion: `allowed_base_dir` restricts accessible paths
