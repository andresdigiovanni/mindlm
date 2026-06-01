# Retrieval Techniques

This document describes all retrieval and query enhancement methods.

---

## 🔍 Retrieval methods

### Vector search
Pure semantic similarity using embeddings.

### Hybrid search
Combines:
- dense embeddings
- sparse keyword matching

### Parent-child retrieval
Retrieves small chunks using context from larger parent chunks.

---

## ⚙️ Query processing

### Query expansion
Generates variations of the query.

### Query rewriting
Rewrites query for better semantic alignment.

### Multi-query retrieval
Runs multiple query variants in parallel.

### HyDE
Generates hypothetical documents for improved retrieval.

### Step-back prompting
Abstracts query to higher-level concepts.

### Query decomposition
Splits complex queries into sub-queries.

---

## 📊 Fusion

### RRF (Reciprocal Rank Fusion)
Combines results from multiple retrieval methods.

---

## 🎯 Reranking

### Cross-encoder reranking
Re-evaluates retrieved documents using a joint encoder.

### LLM reranking
Uses LLM reasoning to reorder results.

---

## 🧩 Context optimization

### Context compression
Reduces irrelevant tokens before generation.

### Contextual summarization
Summarizes retrieved chunks before prompting.
