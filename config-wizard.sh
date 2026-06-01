#!/usr/bin/env bash
# config-wizard.sh — Interactive configuration generator for mindlm
# Usage: bash config-wizard.sh [--output PATH]
set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'
  YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
else
  BOLD=''; CYAN=''; GREEN=''; YELLOW=''; RED=''; NC=''
fi

# ── Helpers ─────────────────────────────────────────────────────────────────
info()    { echo -e "${CYAN}[info]${NC}  $*"; }
success() { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
error()   { echo -e "${RED}[error]${NC} $*" >&2; }

# Read a value from stdin; use default if empty.
# Usage: read_value "Prompt text" "default" varname
read_value() {
  local prompt="$1" default="$2" varname="$3"
  local value
  printf '%b%s%b [%s]: ' "${BOLD}" "${prompt}" "${NC}" "${default}"
  read -r value
  printf -v "${varname}" '%s' "${value:-${default}}"
}

# Read a yes/no value; returns "true" or "false"
read_bool() {
  local prompt="$1" default="$2" varname="$3"
  local value display_default
  display_default=$([ "${default}" = "true" ] && echo "y" || echo "n")
  printf '%b%s%b [%s]: ' "${BOLD}" "${prompt}" "${NC}" "${display_default}"
  read -r value
  value="${value:-${display_default}}"
  case "${value,,}" in
    y|yes|true)  printf -v "${varname}" 'true'  ;;
    n|no|false)  printf -v "${varname}" 'false' ;;
    *)
      warn "Invalid input '${value}'. Expected y/n. Using default: ${display_default}"
      printf -v "${varname}" '%s' "${default}"
      ;;
  esac
}

# Read a value that must not be empty. Re-prompts until non-empty.
read_required() {
  local prompt="$1" varname="$2"
  local value
  while true; do
    printf '%b%s (required)%b: ' "${BOLD}" "${prompt}" "${NC}"
    read -r value
    if [[ -n "${value}" ]]; then
      printf -v "${varname}" '%s' "${value}"
      return
    fi
    warn "This field is required. Please enter a value."
  done
}

# Read an integer that must be > another integer. Re-prompts on violation.
read_int_gt() {
  local prompt="$1" default="$2" min_ref_val="$3" min_label="$4" varname="$5"
  local value
  while true; do
    read_value "${prompt}" "${default}" value
    if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > min_ref_val )); then
      printf -v "${varname}" '%s' "${value}"
      return
    fi
    warn "Value must be an integer greater than ${min_label} (${min_ref_val}). Got: ${value}"
  done
}

# Read an integer constrained to a closed interval.
read_int_range() {
  local prompt="$1" default="$2" min_val="$3" max_val="$4" varname="$5"
  local value
  while true; do
    read_value "${prompt}" "${default}" value
    if [[ "${value}" =~ ^[0-9]+$ ]] && (( value >= min_val && value <= max_val )); then
      printf -v "${varname}" '%s' "${value}"
      return
    fi
    warn "Value must be an integer between ${min_val} and ${max_val}. Got: ${value}"
  done
}

# Validate output path: reject traversal and sensitive system paths.
validate_output_path() {
  local path="$1"
  # reject path traversal
  if [[ "${path}" == *..* ]]; then
    error "Output path must not contain '..'"
    return 1
  fi
  # reject absolute paths to system directories
  local forbidden=(/etc /dev /proc /sys /root /boot /bin /sbin /usr/bin /usr/sbin)
  for f in "${forbidden[@]}"; do
    if [[ "${path}" == "${f}"* ]]; then
      error "Output path '${path}' is not allowed."
      return 1
    fi
  done
  # must end in .yaml or .yml
  if [[ "${path}" != *.yaml && "${path}" != *.yml ]]; then
    error "Output path must end in .yaml or .yml"
    return 1
  fi
  return 0
}

# ── Parse CLI args ──────────────────────────────────────────────────────────
CLI_OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output|-o) CLI_OUTPUT="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: bash config-wizard.sh [--output PATH]"
      echo ""
      echo "Options:"
      echo "  --output PATH   Write config to PATH (default: configs/config.yaml)"
      echo "  --help          Show this help"
      exit 0
      ;;
    *) error "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Header ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║       mindlm — Config Wizard             ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "Press Enter to accept the default shown in [brackets]."
echo ""

# ── Output path ─────────────────────────────────────────────────────────────
if [[ -n "${CLI_OUTPUT}" ]]; then
  OUTPUT_PATH="${CLI_OUTPUT}"
  validate_output_path "${OUTPUT_PATH}" || exit 1
else
  while true; do
    read_value "Output file path" "configs/config.yaml" OUTPUT_PATH
    validate_output_path "${OUTPUT_PATH}" && break
  done
fi

# Backup existing file
if [[ -f "${OUTPUT_PATH}" ]]; then
  warn "File '${OUTPUT_PATH}' already exists."
  BACKUP_PATH="${OUTPUT_PATH}.bak"
  read_bool "Create backup at ${BACKUP_PATH}?" "true" DO_BACKUP
  if [[ "${DO_BACKUP}" == "true" ]]; then
    cp "${OUTPUT_PATH}" "${BACKUP_PATH}"
    success "Backup created at ${BACKUP_PATH}"
  fi
elif [[ -d "${OUTPUT_PATH}" ]]; then
  error "'${OUTPUT_PATH}' is a directory, not a file."
  exit 1
fi

# ── Custom configuration ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}─── LLM ────────────────────────────────────────${NC}"
read_value "Provider" "ollama" LLM_PROVIDER
read_value "Model" "gemma4" LLM_MODEL
read_value "Base URL" "http://ollama:11434" LLM_BASE_URL
read_value "Temperature (0.0–1.0)" "0.7" LLM_TEMPERATURE
read_value "Max tokens" "2048" LLM_MAX_TOKENS

echo ""
echo -e "${BOLD}${CYAN}─── Embeddings ─────────────────────────────────${NC}"
read_value "Provider" "huggingface" EMB_PROVIDER
read_value "Model (HuggingFace model id)" "BAAI/bge-large-en-v1.5" EMB_MODEL
read_value "Dimensions (must match model output)" "1024" EMB_DIMENSIONS

echo ""
echo -e "${BOLD}${CYAN}─── Vector Store ───────────────────────────────${NC}"
read_value "Provider" "qdrant" VS_PROVIDER
read_value "Mode (local / cloud)" "local" VS_MODE
read_value "Host" "qdrant" VS_HOST
read_value "Port" "6333" VS_PORT
read_value "Collection name" "documents" VS_COLLECTION
VS_API_KEY="null"
VS_API_KEY_LINE="  api_key: null"
if [[ "${VS_MODE}" == "cloud" ]]; then
  echo ""
  warn "Cloud mode selected. API key input is hidden."
  printf '%bQdrant API key%b: ' "${BOLD}" "${NC}"
  read -rs VS_API_KEY_INPUT
  echo ""
  VS_API_KEY="${VS_API_KEY_INPUT}"
  if [[ -n "${VS_API_KEY}" ]]; then
    VS_API_KEY_LINE="  api_key: \"${VS_API_KEY}\""
  fi
fi

echo ""
echo -e "${BOLD}${CYAN}─── Ingestion ──────────────────────────────────${NC}"
echo "  Valid values: pdf, html, markdown, png, jpeg, pptx, docx"
read_value "Source types (comma-separated)" "pdf,docx,pptx,html,markdown,png,jpeg" INGESTION_SOURCE_TYPES_RAW
# Convert comma-separated to YAML inline list
INGESTION_SOURCE_TYPES="[$(echo "${INGESTION_SOURCE_TYPES_RAW}" | sed 's/,/, /g')]"
echo "  Accepted values for parsing_strategy: raw | structured | ocr"
read_value "Parsing strategy" "structured" INGESTION_PARSING_STRATEGY
read_bool "Enable deduplication" "true" INGESTION_DEDUP
read_value "Allowed base directory (security boundary)" "/data" INGESTION_BASE_DIR

echo ""
echo -e "${BOLD}${CYAN}─── Chunking ───────────────────────────────────${NC}"
echo "  Strategies: fixed | recursive | sliding | semantic | sentence_window"
read_value "Strategy" "semantic" CHUNKING_STRATEGY
read_value "Chunk size (characters)" "500" CHUNKING_SIZE
read_value "Overlap (characters)" "50" CHUNKING_OVERLAP

CHUNKING_SEMANTIC_MODEL_LINE=""
if [[ "${CHUNKING_STRATEGY}" == "semantic" ]]; then
  read_value "Semantic model (HuggingFace model id)" "${EMB_MODEL}" CHUNKING_SEMANTIC_MODEL
  CHUNKING_SEMANTIC_MODEL_LINE="  semantic_model: ${CHUNKING_SEMANTIC_MODEL}"
fi

CHUNKING_PARENT_LINE=""
read_bool "Enable parent-document retrieval (parent_chunk_size)?" "true" USE_PARENT_CHUNK
if [[ "${USE_PARENT_CHUNK}" == "true" ]]; then
  if [[ "${CHUNKING_STRATEGY}" == "sentence_window" ]]; then
    warn "parent_chunk_size cannot be used with strategy=sentence_window. Skipping."
    USE_PARENT_CHUNK="false"
  fi
fi
if [[ "${USE_PARENT_CHUNK}" == "true" ]]; then
  read_int_gt "Parent chunk size (must be > chunk_size ${CHUNKING_SIZE})" "1500" \
    "${CHUNKING_SIZE}" "chunk_size" CHUNKING_PARENT_SIZE
  CHUNKING_PARENT_LINE="  parent_chunk_size: ${CHUNKING_PARENT_SIZE}"
fi

CHUNKING_SEPARATORS_LINE="  separators: [\"\n\n\", \"\n\", \" \", \"\"]"
if [[ "${CHUNKING_STRATEGY}" == "recursive" ]]; then
  read_value "Separators (YAML flow sequence)" '["\\n\\n", "\\n", ". ", " ", ""]' CHUNKING_SEPS_RAW
  CHUNKING_SEPARATORS_LINE="  separators: ${CHUNKING_SEPS_RAW}"
fi

CHUNKING_WINDOW_SIZE_LINE=""
if [[ "${CHUNKING_STRATEGY}" == "sentence_window" ]]; then
  read_int_range "Window size (number of surrounding sentences)" "2" "1" "100" CHUNKING_WINDOW_SIZE
  CHUNKING_WINDOW_SIZE_LINE="  window_size: ${CHUNKING_WINDOW_SIZE}"
fi

echo ""
echo -e "${BOLD}${CYAN}─── Retrieval ──────────────────────────────────${NC}"
echo "  Strategies: vector | hybrid"
read_value "Strategy" "hybrid" RETRIEVAL_STRATEGY
read_value "Top-k (chunks before reranking)" "40" RETRIEVAL_TOP_K
read_value "Per-query top-k (blank uses top_k)" "20" RETRIEVAL_PER_QUERY_TOP_K
RETRIEVAL_PER_QUERY_TOP_K_LINE="  per_query_top_k: null"
if [[ -n "${RETRIEVAL_PER_QUERY_TOP_K}" ]]; then
  RETRIEVAL_PER_QUERY_TOP_K_LINE="  per_query_top_k: ${RETRIEVAL_PER_QUERY_TOP_K}"
fi

echo ""
echo -e "${BOLD}${CYAN}─── Reranking ──────────────────────────────────${NC}"
read_bool "Enable reranking?" "true" RERANKING_ENABLED
RERANKING_METHOD_LINE="  method: null"
RERANKING_MODEL_LINE="  model: null"
RERANKING_TOP_K_LINE="  top_k: null"
RERANKING_THRESHOLD_LINE="  score_threshold: null"
if [[ "${RERANKING_ENABLED}" == "true" ]]; then
  echo "  Methods: cross_encoder | mmr | llm"
  read_value "Method" "cross_encoder" RERANKING_METHOD
  RERANKING_METHOD_LINE="  method: ${RERANKING_METHOD}"
  if [[ "${RERANKING_METHOD}" == "cross_encoder" ]]; then
    read_value "Reranking model (HuggingFace model id)" "BAAI/bge-reranker-v2-m3" RERANKING_MODEL
    RERANKING_MODEL_LINE="  model: ${RERANKING_MODEL}"
  fi
  read_value "Reranking top-k (blank keeps all)" "10" RERANKING_TOP_K
  if [[ -n "${RERANKING_TOP_K}" ]]; then
    RERANKING_TOP_K_LINE="  top_k: ${RERANKING_TOP_K}"
  fi
  read_value "Score threshold 0.0-1.0 (blank disables)" "0.1" RERANKING_THRESHOLD
  if [[ -n "${RERANKING_THRESHOLD}" ]]; then
    RERANKING_THRESHOLD_LINE="  score_threshold: ${RERANKING_THRESHOLD}"
  fi
fi

echo ""
echo -e "${BOLD}${CYAN}─── Contextual Retrieval ───────────────────────${NC}"
read_bool "Enable chunk context enrichment?" "false" CR_CHUNK_CONTEXT
read_bool "Enable document summary enrichment?" "true" CR_DOC_SUMMARY

CR_PROMPT_TEMPLATE_LINE=""
CR_DOC_PROMPT_TEMPLATE_LINE=""
read_bool "Customize chunk-context prompt template?" "false" CR_USE_CUSTOM_TEMPLATE
if [[ "${CR_USE_CUSTOM_TEMPLATE}" == "true" ]]; then
  read_required "Chunk-context prompt template (single line; use {document} and {chunk})" CR_PROMPT_TEMPLATE
  CR_PROMPT_TEMPLATE_LINE="  prompt_template: \"${CR_PROMPT_TEMPLATE}\""
fi
read_bool "Customize document-summary prompt template?" "false" CR_USE_CUSTOM_DOC_TEMPLATE
if [[ "${CR_USE_CUSTOM_DOC_TEMPLATE}" == "true" ]]; then
  read_required "Document-summary prompt template (single line; use {document})" CR_DOC_PROMPT_TEMPLATE
  CR_DOC_PROMPT_TEMPLATE_LINE="  document_summary_prompt_template: \"${CR_DOC_PROMPT_TEMPLATE}\""
fi

echo ""
echo -e "${BOLD}${CYAN}─── Compression ────────────────────────────────${NC}"
read_bool "Enable contextual compression?" "false" COMPRESSION_ENABLED

echo ""
echo -e "${BOLD}${CYAN}─── Query Processing ───────────────────────────${NC}"
echo "  Each processor is independent. Enable any combination."
read_bool "Enable query rewriting?" "true" QP_REWRITING
read_bool "Enable query expansion?" "true" QP_EXPANSION
read_bool "Enable HyDE (hypothetical document)?" "true" QP_HYDE

read_bool "Enable multi-query?" "true" QP_MULTI_QUERY
QP_MULTI_QUERY_VARIANTS_LINE="    num_variants: 3"
if [[ "${QP_MULTI_QUERY}" == "true" ]]; then
  read_value "Number of query variants (2–10)" "3" QP_MULTI_QUERY_VARIANTS
  QP_MULTI_QUERY_VARIANTS_LINE="    num_variants: ${QP_MULTI_QUERY_VARIANTS}"
fi

read_bool "Enable query decomposition?" "true" QP_DECOMP
QP_DECOMP_SUBQUERIES_LINE="    max_subqueries: 4"
if [[ "${QP_DECOMP}" == "true" ]]; then
  read_int_range "Max sub-queries (2-10)" "4" "2" "10" QP_DECOMP_SUBQUERIES
  QP_DECOMP_SUBQUERIES_LINE="    max_subqueries: ${QP_DECOMP_SUBQUERIES}"
fi

read_bool "Enable step-back prompting?" "true" QP_STEP_BACK
read_bool "Enable adaptive query planner?" "true" QP_PLANNER_ENABLED

echo ""
echo -e "${BOLD}${CYAN}─── Iterative Retrieval ────────────────────────${NC}"
read_bool "Enable iterative retrieval loop for /ask?" "true" ITERATIVE_ENABLED
ITERATIVE_MAX_ITERS_LINE="  max_iterations: 3"
if [[ "${ITERATIVE_ENABLED}" == "true" ]]; then
  read_int_range "Max iterations (1-5)" "3" "1" "5" ITERATIVE_MAX_ITERS
  ITERATIVE_MAX_ITERS_LINE="  max_iterations: ${ITERATIVE_MAX_ITERS}"
fi

echo ""
echo -e "${BOLD}${CYAN}─── Observability (Langfuse) ───────────────────${NC}"
read_bool "Enable observability?" "true" OBS_ENABLED
read_value "Public key" "pk-lf-local-dev" OBS_PUBLIC_KEY
read_value "Secret key" "sk-lf-local-dev" OBS_SECRET_KEY
read_value "Host" "http://langfuse:3000" OBS_HOST
read_value "Flush at" "15" OBS_FLUSH_AT
read_value "Flush interval (seconds)" "0.5" OBS_FLUSH_INTERVAL

# ── Write config ─────────────────────────────────────────────────────────────
mkdir -p "$(dirname "${OUTPUT_PATH}")"

cat > "${OUTPUT_PATH}" <<YAML
# Generated by config-wizard.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Edit this file to adjust your configuration.
# Re-run config-wizard.sh to regenerate from scratch.

# ── Infrastructure ──────────────────────────────────────────

llm:
  provider: ${LLM_PROVIDER}
  model: ${LLM_MODEL}
  base_url: ${LLM_BASE_URL}
  temperature: ${LLM_TEMPERATURE}
  max_tokens: ${LLM_MAX_TOKENS}

embeddings:
  provider: ${EMB_PROVIDER}
  model: ${EMB_MODEL}
  dimensions: ${EMB_DIMENSIONS}

vector_store:
  provider: ${VS_PROVIDER}
  mode: ${VS_MODE}
  host: ${VS_HOST}
  port: ${VS_PORT}
  collection: ${VS_COLLECTION}
${VS_API_KEY_LINE}

# ── Ingestion Pipeline ──────────────────────────────────────

ingestion:
  source_type: ${INGESTION_SOURCE_TYPES}
  parsing_strategy: ${INGESTION_PARSING_STRATEGY}
  deduplication: ${INGESTION_DEDUP}
  allowed_base_dir: ${INGESTION_BASE_DIR}

chunking:
  strategy: ${CHUNKING_STRATEGY}
  chunk_size: ${CHUNKING_SIZE}
  overlap: ${CHUNKING_OVERLAP}
${CHUNKING_SEMANTIC_MODEL_LINE:+${CHUNKING_SEMANTIC_MODEL_LINE}
}${CHUNKING_PARENT_LINE:+${CHUNKING_PARENT_LINE}
}${CHUNKING_WINDOW_SIZE_LINE:+${CHUNKING_WINDOW_SIZE_LINE}
}${CHUNKING_SEPARATORS_LINE}

# ── Retrieval Pipeline ──────────────────────────────────────

retrieval:
  strategy: ${RETRIEVAL_STRATEGY}
  top_k: ${RETRIEVAL_TOP_K}
${RETRIEVAL_PER_QUERY_TOP_K_LINE}

reranking:
  enabled: ${RERANKING_ENABLED}
${RERANKING_METHOD_LINE}
${RERANKING_MODEL_LINE}
${RERANKING_TOP_K_LINE}
${RERANKING_THRESHOLD_LINE}

contextual_retrieval:
  chunk_context_enabled: ${CR_CHUNK_CONTEXT}
  document_summary_enabled: ${CR_DOC_SUMMARY}
${CR_PROMPT_TEMPLATE_LINE:+${CR_PROMPT_TEMPLATE_LINE}
}${CR_DOC_PROMPT_TEMPLATE_LINE:+${CR_DOC_PROMPT_TEMPLATE_LINE}
}

compression:
  enabled: ${COMPRESSION_ENABLED}

# ── Query Processing Pipeline ───────────────────────────────

query_processing:
  rewriting:
    enabled: ${QP_REWRITING}
  expansion:
    enabled: ${QP_EXPANSION}
  hyde:
    enabled: ${QP_HYDE}
  multi_query:
    enabled: ${QP_MULTI_QUERY}
${QP_MULTI_QUERY_VARIANTS_LINE}
  decomposition:
    enabled: ${QP_DECOMP}
${QP_DECOMP_SUBQUERIES_LINE}
  step_back:
    enabled: ${QP_STEP_BACK}
  planner:
    enabled: ${QP_PLANNER_ENABLED}

iterative_retrieval:
  enabled: ${ITERATIVE_ENABLED}
${ITERATIVE_MAX_ITERS_LINE}

observability:
  enabled: ${OBS_ENABLED}
  public_key: ${OBS_PUBLIC_KEY}
  secret_key: ${OBS_SECRET_KEY}
  host: ${OBS_HOST}
  flush_at: ${OBS_FLUSH_AT}
  flush_interval: ${OBS_FLUSH_INTERVAL}
YAML

success "Config written to: ${OUTPUT_PATH}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "  1. Review ${OUTPUT_PATH} and adjust any remaining settings."
echo "  2. Run: bash mindlm.sh start"
echo ""
