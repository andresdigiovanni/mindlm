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

# ── Script directory ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# ── Profile selection ────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Select a configuration profile:${NC}"
echo ""
echo "  1) minimal   — quick start; fixed chunking, no query processing, low resource"
echo "  2) balanced  — recommended; recursive chunking, rewriting, cross-encoder reranking"
echo "  3) full      — maximum quality; semantic chunking, hybrid retrieval, all processors"
echo "  4) custom    — configure each option interactively"
echo ""
read_value "Profile" "2" PROFILE_CHOICE

case "${PROFILE_CHOICE}" in
  1) PROFILE_NAME="minimal"  ;;
  2) PROFILE_NAME="balanced" ;;
  3) PROFILE_NAME="full"     ;;
  4) PROFILE_NAME="custom"   ;;
  minimal|balanced|full|custom) PROFILE_NAME="${PROFILE_CHOICE}" ;;
  *) warn "Unknown choice '${PROFILE_CHOICE}'. Using 'balanced'."; PROFILE_NAME="balanced" ;;
esac

# ── Use a preset profile ─────────────────────────────────────────────────────
if [[ "${PROFILE_NAME}" != "custom" ]]; then
  PROFILE_FILE="${SCRIPT_DIR}/configs/profiles/${PROFILE_NAME}.yaml"
  if [[ ! -f "${PROFILE_FILE}" ]]; then
    error "Profile file not found: ${PROFILE_FILE}"
    exit 1
  fi

  echo ""
  info "Using profile: ${PROFILE_NAME}"
  echo ""

  # Allow patching common fields
  read_value "App name" "local-rag" APP_NAME
  read_value "Qdrant collection name" "documents" COLLECTION_NAME
  read_value "LLM model" "llama3" LLM_MODEL
  read_value "Ollama base URL" "http://ollama:11434" LLM_BASE_URL

  # Copy profile and patch fields safely using Python (avoids sed special-char injection)
  mkdir -p "$(dirname "${OUTPUT_PATH}")"
  cp "${PROFILE_FILE}" "${OUTPUT_PATH}"
  python3 - "${OUTPUT_PATH}" "${APP_NAME}" "${COLLECTION_NAME}" "${LLM_MODEL}" "${LLM_BASE_URL}" <<'PYEOF'
import sys, re
path, app_name, collection, llm_model, base_url = sys.argv[1:]
with open(path, encoding='utf-8') as f:
    text = f.read()
text = re.sub(r'^(  name:).*', lambda m: m.group(1) + ' ' + app_name, text, flags=re.MULTILINE)
text = re.sub(r'^(  collection:).*', lambda m: m.group(1) + ' ' + collection, text, flags=re.MULTILINE)
text = re.sub(r'^(  model:).*', lambda m: m.group(1) + ' ' + llm_model, text, count=1, flags=re.MULTILINE)
text = re.sub(r'^(  base_url:).*', lambda m: m.group(1) + ' ' + base_url, text, flags=re.MULTILINE)
with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
PYEOF

  success "Config written to: ${OUTPUT_PATH}"
  echo ""
  echo -e "${BOLD}Next steps:${NC}"
  echo "  1. Review ${OUTPUT_PATH} and adjust any remaining settings."
  echo "  2. Run: bash mindlm.sh start"
  echo ""
  exit 0
fi

# ── Custom configuration ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}─── App ────────────────────────────────────────${NC}"
read_value "App name" "local-rag" APP_NAME

echo ""
echo -e "${BOLD}${CYAN}─── LLM ────────────────────────────────────────${NC}"
read_value "Provider" "ollama" LLM_PROVIDER
read_value "Model" "llama3" LLM_MODEL
read_value "Base URL" "http://ollama:11434" LLM_BASE_URL
read_value "Temperature (0.0–1.0)" "0.7" LLM_TEMPERATURE
read_value "Max tokens" "1024" LLM_MAX_TOKENS

echo ""
echo -e "${BOLD}${CYAN}─── Embeddings ─────────────────────────────────${NC}"
read_value "Provider" "huggingface" EMB_PROVIDER
read_value "Model (HuggingFace model id)" "BAAI/bge-small-en-v1.5" EMB_MODEL
read_value "Dimensions (must match model output)" "384" EMB_DIMENSIONS

echo ""
echo -e "${BOLD}${CYAN}─── Vector Store ───────────────────────────────${NC}"
read_value "Provider" "qdrant" VS_PROVIDER
read_value "Mode (local / cloud)" "local" VS_MODE
read_value "Host" "qdrant" VS_HOST
read_value "Port" "6333" VS_PORT
read_value "Collection name" "documents" VS_COLLECTION
VS_API_KEY="null"
if [[ "${VS_MODE}" == "cloud" ]]; then
  echo ""
  warn "Cloud mode selected. API key input is hidden."
  printf '%bQdrant API key%b: ' "${BOLD}" "${NC}"
  read -rs VS_API_KEY_INPUT
  echo ""
  VS_API_KEY="${VS_API_KEY_INPUT:-null}"
fi

echo ""
echo -e "${BOLD}${CYAN}─── Ingestion ──────────────────────────────────${NC}"
echo "  Valid values: pdf, html, markdown, png, jpeg, pptx, docx"
read_value "Source types (comma-separated)" "pdf,docx,markdown" INGESTION_SOURCE_TYPES_RAW
# Convert comma-separated to YAML inline list
INGESTION_SOURCE_TYPES="[$(echo "${INGESTION_SOURCE_TYPES_RAW}" | sed 's/,/, /g')]"
echo "  Accepted values for parsing_strategy: raw | structured | ocr"
read_value "Parsing strategy" "raw" INGESTION_PARSING_STRATEGY
read_bool "Enable deduplication" "true" INGESTION_DEDUP
read_value "Allowed base directory (security boundary)" "/data" INGESTION_BASE_DIR

echo ""
echo -e "${BOLD}${CYAN}─── Chunking ───────────────────────────────────${NC}"
echo "  Strategies: fixed | recursive | sliding | semantic"
read_value "Strategy" "recursive" CHUNKING_STRATEGY
read_value "Chunk size (characters)" "512" CHUNKING_SIZE
read_value "Overlap (characters)" "64" CHUNKING_OVERLAP

CHUNKING_SEMANTIC_MODEL_LINE=""
if [[ "${CHUNKING_STRATEGY}" == "semantic" ]]; then
  read_required "Semantic model (HuggingFace model id)" CHUNKING_SEMANTIC_MODEL
  CHUNKING_SEMANTIC_MODEL_LINE="  semantic_model: ${CHUNKING_SEMANTIC_MODEL}"
fi

CHUNKING_PARENT_LINE=""
read_bool "Enable parent-document retrieval (parent_chunk_size)?" "false" USE_PARENT_CHUNK
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

echo ""
echo -e "${BOLD}${CYAN}─── Retrieval ──────────────────────────────────${NC}"
echo "  Strategies: vector | hybrid"
read_value "Strategy" "vector" RETRIEVAL_STRATEGY
read_value "Top-k (chunks before reranking)" "5" RETRIEVAL_TOP_K

echo ""
echo -e "${BOLD}${CYAN}─── Reranking ──────────────────────────────────${NC}"
read_bool "Enable reranking?" "false" RERANKING_ENABLED
RERANKING_METHOD_LINE="  # method: cross_encoder"
RERANKING_MODEL_LINE="  # model: cross-encoder/ms-marco-MiniLM-L-6-v2"
if [[ "${RERANKING_ENABLED}" == "true" ]]; then
  echo "  Methods: cross_encoder | mmr"
  read_value "Method" "cross_encoder" RERANKING_METHOD
  RERANKING_METHOD_LINE="  method: ${RERANKING_METHOD}"
  if [[ "${RERANKING_METHOD}" == "cross_encoder" ]]; then
    read_value "Reranking model (HuggingFace model id)" "cross-encoder/ms-marco-MiniLM-L-6-v2" RERANKING_MODEL
    RERANKING_MODEL_LINE="  model: ${RERANKING_MODEL}"
  fi
fi

echo ""
echo -e "${BOLD}${CYAN}─── Query Processing ───────────────────────────${NC}"
echo "  Each processor is independent. Enable any combination."
read_bool "Enable query rewriting?" "false" QP_REWRITING
read_bool "Enable query expansion?" "false" QP_EXPANSION
read_bool "Enable HyDE (hypothetical document)?" "false" QP_HYDE

read_bool "Enable multi-query?" "false" QP_MULTI_QUERY
QP_MULTI_QUERY_VARIANTS_LINE="    num_variants: 3"
if [[ "${QP_MULTI_QUERY}" == "true" ]]; then
  read_value "Number of query variants (2–10)" "3" QP_MULTI_QUERY_VARIANTS
  QP_MULTI_QUERY_VARIANTS_LINE="    num_variants: ${QP_MULTI_QUERY_VARIANTS}"
fi

read_bool "Enable query decomposition?" "false" QP_DECOMP
QP_DECOMP_SUBQUERIES_LINE="    max_subqueries: 4"
if [[ "${QP_DECOMP}" == "true" ]]; then
  read_value "Max sub-queries (2–10)" "4" QP_DECOMP_SUBQUERIES
  QP_DECOMP_SUBQUERIES_LINE="    max_subqueries: ${QP_DECOMP_SUBQUERIES}"
fi

read_bool "Enable step-back prompting?" "false" QP_STEP_BACK

# ── Write config ─────────────────────────────────────────────────────────────
mkdir -p "$(dirname "${OUTPUT_PATH}")"

cat > "${OUTPUT_PATH}" <<YAML
# Generated by config-wizard.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Edit this file to adjust your configuration.
# Re-run config-wizard.sh to regenerate from scratch.

app:
  name: ${APP_NAME}

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
  api_key: ${VS_API_KEY}

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
}${CHUNKING_SEPARATORS_LINE}

# ── Retrieval Pipeline ──────────────────────────────────────

retrieval:
  strategy: ${RETRIEVAL_STRATEGY}
  top_k: ${RETRIEVAL_TOP_K}

reranking:
  enabled: ${RERANKING_ENABLED}
${RERANKING_METHOD_LINE}
${RERANKING_MODEL_LINE}

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
YAML

success "Config written to: ${OUTPUT_PATH}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "  1. Review ${OUTPUT_PATH} and adjust any remaining settings."
echo "  2. Run: bash mindlm.sh start"
echo ""
