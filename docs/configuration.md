# Configuration Reference

All configuration lives in `configs/config.yaml`. The file is mounted as a read-only volume into all containers — no rebuild is required after changes.

---

## `llm`

Controls the language model used for answer generation.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | `"ollama"` | `"ollama"` | LLM provider. Only `ollama` is currently supported |
| `model` | string | `"gemma4"` | Ollama model name. Must be pulled with `ollama pull <model>` |
| `base_url` | string | `"http://ollama:11434"` | Ollama service URL. Use `http://localhost:11434` for local dev outside Docker |
| `temperature` | float | `0.7` | Sampling temperature. `0.0` = deterministic, `1.0` = most random |
| `max_tokens` | int | `2048` | Maximum tokens to generate per response |

---

## `embeddings`

Controls the embedding model used to vectorize documents and queries.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | `"huggingface"` | `"huggingface"` | Embeddings provider. Only `huggingface` is currently supported |
| `model` | string | `"BAAI/bge-large-en-v1.5"` | HuggingFace model name. Downloaded at first run into the `hf_cache` volume |
| `dimensions` | int | `1024` | Output vector dimensions. **Must match the model's actual output size** |

Common model / dimension pairs:

| Model | Dimensions |
|-------|-----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `sentence-transformers/all-mpnet-base-v2` | 768 |
| `BAAI/bge-large-en-v1.5` | 1024 |

> **Important:** changing the model after a collection has been created requires a full re-ingest because the vector dimensions will differ.

---

## `vector_store`

Controls the Qdrant vector database connection.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | `"qdrant"` | `"qdrant"` | Vector store provider. Only `qdrant` is currently supported |
| `mode` | `"local"` \| `"cloud"` | `"local"` | `local` connects to a self-hosted instance; `cloud` uses Qdrant Cloud |
| `host` | string | `"qdrant"` | Qdrant hostname. Use `localhost` for local dev outside Docker |
| `port` | int | `6333` | Qdrant HTTP port |
| `collection` | string | `"documents"` | Default collection name (created automatically on first ingest) |
| `api_key` | string \| null | `null` | API key for Qdrant Cloud. Leave unset for local mode |

---

## `ingestion`

Controls which files are accepted and how they are parsed.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `source_type` | list | all types | File types to accept. Subset of: `pdf`, `html`, `markdown`, `png`, `jpeg`, `pptx`, `docx` |
| `parsing_strategy` | `"raw"` \| `"structured"` \| `"ocr"` | `"structured"` | See below |
| `deduplication` | bool | `true` | Skip unchanged files on incremental sync (hash-based) |
| `allowed_base_dir` | string | `"/data"` | **Security boundary**: only paths under this directory are accepted |

**Parsing strategies:**

| Strategy | Description |
|----------|-------------|
| `raw` | Extract plain text with no layout analysis |
| `structured` | Preserve headings, lists, and tables (recommended for most documents) |
| `ocr` | Run OCR via Surya for scanned images and PDFs with embedded images. Requires the `surya-ocr` optional dependency |

---

## `chunking`

Controls how documents are split into chunks before indexing.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strategy` | `"fixed"` \| `"sliding"` \| `"semantic"` \| `"recursive"` \| `"sentence_window"` | `"semantic"` | See below |
| `chunk_size` | int | `500` | Target chunk size in tokens (fixed/sliding/recursive) or characters (semantic) |
| `overlap` | int | `50` | Token/character overlap between consecutive chunks |
| `semantic_model` | string \| null | `null` | HuggingFace model for semantic splitting. **Required when `strategy: semantic`** |
| `window_size` | int \| null | `null` | Surrounding sentences stored per sentence chunk. **Required when `strategy: sentence_window`** |
| `parent_chunk_size` | int \| null | `null` | When set, enables parent-document retrieval: small child chunks are indexed; results are expanded to parent content before returning. Must be greater than `chunk_size` |
| `separators` | list[string] | `["\n\n", "\n", ". ", " ", ""]` | Separator hierarchy for recursive chunking |

**Chunking strategies:**

| Strategy | Description |
|----------|-------------|
| `fixed` | Split into chunks of exactly `chunk_size` tokens with `overlap` |
| `sliding` | Sliding window similar to fixed; every chunk shifts by `chunk_size − overlap` |
| `semantic` | Group sentences by semantic similarity; `chunk_size` is the upper bound |
| `recursive` | Try separators in order, recursing into oversized pieces. Best for structured text |
| `sentence_window` | Index individual sentences; each stores a window of surrounding sentences in its payload for wider context at retrieval time |

---

## `contextual_retrieval`

Optional LLM-generated metadata stored at ingest time. Raw chunk text is always embedded unchanged; these fields appear only in search result metadata.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `chunk_context_enabled` | bool | `false` | Generate a one-sentence context per chunk (1 LLM call per chunk → `payload["chunk_context"]`) |
| `document_summary_enabled` | bool | `false` | Generate a one-sentence document summary (1 LLM call per document → `payload["document_summary"]`) |

> Enabling these fields increases ingestion cost proportionally to the number of chunks/documents.

---

## `retrieval`

Controls how documents are retrieved for a given query.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strategy` | `"vector"` \| `"hybrid"` | `"hybrid"` | See below |
| `top_k` | int | `40` | Final number of candidates passed to reranking (or returned directly if reranking is disabled) |
| `per_query_top_k` | int \| null | `20` | Per-query candidate pool when query processing generates multiple variants. Each variant fetches this many results before merging |

**Retrieval strategies:**

| Strategy | Description |
|----------|-------------|
| `vector` | Dense vector similarity search only |
| `hybrid` | Combines dense vector search with sparse BM25 keyword search via Reciprocal Rank Fusion (RRF). Recommended for most use cases |

---

## `reranking`

Optional post-retrieval reranking step to improve result relevance.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable or disable reranking |
| `method` | `"cross_encoder"` \| `"mmr"` \| `"llm"` \| null | `null` | Reranking algorithm. Required when `enabled: true` |
| `model` | string \| null | `null` | HuggingFace model for cross-encoder scoring. Required when `method: cross_encoder` |
| `top_k` | int \| null | `null` | Number of results to keep after reranking |
| `score_threshold` | float \| null | `null` | Minimum reranking score; results below this threshold are discarded |

**Reranking methods:**

| Method | Description |
|--------|-------------|
| `cross_encoder` | Uses a cross-encoder model to re-score query–document pairs. More accurate but slower |
| `mmr` | Maximal Marginal Relevance: balances relevance and diversity. No extra model needed |
| `llm` | LLM-as-reranker: scores each (query, chunk) pair on a 1–10 scale. No extra model needed |

---

## `compression`

Optional post-retrieval context compression. Uses the LLM to extract only the query-relevant sentences from each retrieved result; results where no content is relevant are dropped entirely.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable or disable contextual compression |

> Adds one LLM call per retrieved result. Pairs well with `sentence_window` chunking to prune oversized context windows.

---

## `query_processing`

Optional pre-retrieval pipeline that transforms the query into alternative representations to improve recall. Multiple processors can be enabled simultaneously; results are merged and deduplicated before reranking.

| Processor | Config key | Description |
|-----------|------------|-------------|
| Query Rewriting | `rewriting` | Reformulates the query for better semantic alignment |
| Query Expansion | `expansion` | Adds synonyms and related terms |
| HyDE | `hyde` | Generates a hypothetical answer passage and embeds it instead of the raw query |
| Multi-Query | `multi_query` | Generates N rephrased variants, retrieves for each, then merges |
| Query Decomposition | `decomposition` | Breaks a complex query into focused sub-questions |
| Step-Back Prompting | `step_back` | Generates a more abstract version of the query |
| Adaptive Planner | `planner` | LLM selects which processors to run per query; skips inappropriate ones |

Per-processor config:

| Key | Type | Default | Applies to |
|-----|------|---------|------------|
| `enabled` | bool | `false` | all processors |
| `num_variants` | int (2–10) | `3` | `multi_query` only |
| `max_subqueries` | int (2–10) | `4` | `decomposition` only |

Example — enable rewriting and multi-query:

```yaml
query_processing:
  rewriting:
    enabled: true
  multi_query:
    enabled: true
    num_variants: 3
  planner:
    enabled: true   # LLM decides which enabled processors to use per query
```

---

## `iterative_retrieval`

Optional grounding-check loop for the `/ask` endpoint. After each generate step, the LLM evaluates whether the answer is supported by the retrieved context. If not grounded and a refined query is suggested, retrieval is retried.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable the iterative retrieval loop |
| `max_iterations` | int (1–5) | `3` | Maximum retrieve–generate iterations. Exits early once grounded |

> On LLM errors the grounding check falls back to `is_grounded=True`, so the loop exits rather than retrying indefinitely.

---

## `observability`

Langfuse tracing for the full RAG pipeline.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable Langfuse tracing |

When enabled, all pipeline stages (query processing, embedding, retrieval, reranking, generation) emit traces to the Langfuse service defined in `docker-compose.yml` (default UI: `http://localhost:3000`).

---

## Constraints

* `embeddings.dimensions` must match the model's actual output size and the Qdrant collection dimensions.
* `chunking.parent_chunk_size` must be greater than `chunking.chunk_size`.
* `chunking.semantic_model` is required when `strategy: semantic`.
* `chunking.window_size` is required when `strategy: sentence_window`.
* `reranking.model` is required when `reranking.method: cross_encoder`.
* HyDE and multi-query increase LLM call count and latency proportionally.
* `ingestion.allowed_base_dir` is a security boundary — set it to the narrowest directory covering your document sources.
