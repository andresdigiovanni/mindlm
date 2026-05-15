#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# mindlm — CLI for managing the mindlm RAG platform
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
API_BASE="${MINDLM_API_BASE:-http://localhost:8000}"

# ── Colors (suppressed when NO_COLOR is set or output is not a terminal) ───
if [[ -z "${NO_COLOR:-}" && -t 1 ]]; then
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  RED='\033[0;31m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  GREEN='' YELLOW='' RED='' BOLD='' NC=''
fi

info()    { printf "${BOLD}[mindlm]${NC} %s\n"         "$*"; }
success() { printf "${GREEN}[mindlm]${NC} %s\n"        "$*"; }
warn()    { printf "${YELLOW}[mindlm] WARNING:${NC} %s\n" "$*" >&2; }
error()   { printf "${RED}[mindlm] ERROR:${NC} %s\n"   "$*" >&2; }

# ── Startup guards ─────────────────────────────────────────────────────────
check_deps() {
  if ! docker compose version &>/dev/null; then
    error "docker compose v2 is required but not found."
    error "Install Docker Desktop or the docker-compose-plugin package."
    exit 1
  fi
  if ! command -v curl &>/dev/null; then
    error "'curl' is required but not found. Install it and retry."
    exit 1
  fi
}

# ── JSON pretty-printer ────────────────────────────────────────────────────
pretty_json() {
  if command -v jq &>/dev/null; then
    jq .
  else
    cat
  fi
}

# ── Safely JSON-encode a string ────────────────────────────────────────────
# Uses jq when available for correct escaping; falls back to basic quoting.
json_string() {
  local s="$1"
  if command -v jq &>/dev/null; then
    printf '%s' "$s" | jq -Rs .
  else
    # Basic: escape backslash and double-quote
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    printf '"%s"' "$s"
  fi
}

# ── Build a JSON array from positional args ────────────────────────────────
json_array() {
  if command -v jq &>/dev/null; then
    jq -nc '$ARGS.positional' --args -- "$@"
  else
    local arr="["
    local first=1
    for item in "$@"; do
      item="${item//\\/\\\\}"
      item="${item//\"/\\\"}"
      if [[ $first -eq 1 ]]; then
        arr+="\"$item\""
        first=0
      else
        arr+=",\"$item\""
      fi
    done
    arr+="]"
    printf '%s' "$arr"
  fi
}

# ── Usage ──────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
${BOLD}Usage:${NC} mindlm <command> [options]

${BOLD}Commands:${NC}
  start                  Start all services (waits for API health)
  stop                   Stop all services
  build                  Build Docker images
  status                 Show service status
  health                 Show API health status
  collections            List available collections
  search <query>         Search the knowledge base
    --top-k N              Number of results (default: 5)
    --collection NAME      Restrict to a specific collection
  ask <question>         Ask a question
    --collection NAME      Restrict to a specific collection
  ingest [<path>...]       Ingest documents (incremental sync; defaults to /data)
  ingest-full [<path>...]  Ingest documents (full re-index; defaults to /data)
  install                Install mindlm to ~/.local/bin
  uninstall              Remove mindlm from ~/.local/bin
  config-wizard            Launch interactive configuration generator
  help                   Show this help

${BOLD}Environment:${NC}
  MINDLM_API_BASE        Override API base URL (default: http://localhost:8000)
  NO_COLOR               Disable colored output

${BOLD}Notes:${NC}
  - 'search' and 'ask' use jq for pretty output when available.
  - For queries with special characters, install jq for correct JSON encoding.
EOF
}

# ── Commands ───────────────────────────────────────────────────────────────

cmd_start() {
  info "Starting services..."
  docker compose -f "$COMPOSE_FILE" up -d

  info "Waiting for API to become healthy (timeout: 90s)..."
  local elapsed=0
  while [[ $elapsed -lt 90 ]]; do
    local status
    status=$(curl -sf "$API_BASE/health" 2>/dev/null | grep -o '"status":"[^"]*"' || true)
    if [[ "$status" == *'"ok"'* ]]; then
      success "API is healthy at $API_BASE"
      return 0
    fi
    sleep 3
    elapsed=$((elapsed + 3))
  done

  error "Timed out waiting for the API to become healthy."
  error "Check logs with: docker compose logs api"
  exit 1
}

cmd_stop() {
  info "Stopping services..."
  docker compose -f "$COMPOSE_FILE" down
  success "Services stopped."
}

cmd_build() {
  info "Building Docker images..."
  docker compose -f "$COMPOSE_FILE" build
  success "Build complete."
}

cmd_status() {
  docker compose -f "$COMPOSE_FILE" ps
}

cmd_health() {
  curl -sf "$API_BASE/health" | pretty_json
}

cmd_collections() {
  curl -sf "$API_BASE/collections" | pretty_json
}

cmd_search() {
  local query=""
  local top_k=5
  local collection="null"

  # Parse arguments
  if [[ $# -eq 0 ]]; then
    error "Missing required argument: <query>"
    printf 'Usage: mindlm search "<query>" [--top-k N] [--collection NAME]\n' >&2
    exit 1
  fi

  query="$1"
  shift

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --top-k)
        if [[ $# -lt 2 ]]; then
          error "--top-k requires a value"; exit 1
        fi
        if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
          error "--top-k must be a positive integer, got: $2"; exit 1
        fi
        top_k="$2"; shift 2 ;;
      --collection)
        if [[ $# -lt 2 ]]; then
          error "--collection requires a value"; exit 1
        fi
        collection="$(json_string "$2")"; shift 2 ;;
      *)
        error "Unknown option: $1"
        printf 'Usage: mindlm search "<query>" [--top-k N] [--collection NAME]\n' >&2
        exit 1 ;;
    esac
  done

  local query_json
  query_json="$(json_string "$query")"
  local body
  body="$(printf '{"query":%s,"top_k":%d,"filters":null,"collection":%s}' \
    "$query_json" "$top_k" "$collection")"

  curl -s -X POST "$API_BASE/search" \
    -H "Content-Type: application/json" \
    -d "$body" | pretty_json
}

cmd_ask() {
  local question=""
  local collection="null"

  if [[ $# -eq 0 ]]; then
    error "Missing required argument: <question>"
    printf 'Usage: mindlm ask "<question>" [--collection NAME]\n' >&2
    exit 1
  fi

  question="$1"
  shift

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --collection)
        if [[ $# -lt 2 ]]; then
          error "--collection requires a value"; exit 1
        fi
        collection="$(json_string "$2")"; shift 2 ;;
      *)
        error "Unknown option: $1"
        printf 'Usage: mindlm ask "<question>" [--collection NAME]\n' >&2
        exit 1 ;;
    esac
  done

  local question_json
  question_json="$(json_string "$question")"
  local body
  body="$(printf '{"question":%s,"filters":null,"collection":%s}' \
    "$question_json" "$collection")"

  curl -s -X POST "$API_BASE/ask" \
    -H "Content-Type: application/json" \
    -d "$body" | pretty_json
}

cmd_ingest() {
  if [[ $# -eq 0 ]]; then
    printf '[mindlm] No paths specified. Ingest /data? [Y/n] '
    read -r reply
    reply="${reply:-Y}"
    if [[ "$reply" =~ ^[Yy]$ ]]; then
      set -- /data
    else
      error "Aborted."
      exit 1
    fi
  fi

  local paths_json
  paths_json="$(json_array "$@")"
  local body
  body="$(printf '{"paths":%s}' "$paths_json")"

  curl -s -X POST "$API_BASE/ingest/sync" \
    -H "Content-Type: application/json" \
    -d "$body" | pretty_json
}

cmd_ingest_full() {
  if [[ $# -eq 0 ]]; then
    printf '[mindlm] No paths specified. Full re-index /data? [Y/n] '
    read -r reply
    reply="${reply:-Y}"
    if [[ "$reply" =~ ^[Yy]$ ]]; then
      set -- /data
    else
      error "Aborted."
      exit 1
    fi
  fi

  local paths_json
  paths_json="$(json_array "$@")"
  local body
  body="$(printf '{"paths":%s}' "$paths_json")"

  curl -s -X POST "$API_BASE/ingest/full" \
    -H "Content-Type: application/json" \
    -d "$body" | pretty_json
}

cmd_config_wizard() {
  local wizard="${SCRIPT_DIR}/config-wizard.sh"
  if [[ ! -f "${wizard}" ]]; then
    error "config-wizard.sh not found at ${wizard}"
    exit 1
  fi
  bash "${wizard}" "$@"
}

cmd_install() {
  local script_path="$SCRIPT_DIR/mindlm.sh"
  local bin_dir="$HOME/.local/bin"
  local link_path="$bin_dir/mindlm"

  chmod +x "$script_path"
  mkdir -p "$bin_dir"
  ln -sf "$script_path" "$link_path"
  success "Symlink created: $link_path -> $script_path"

  # Add ~/.local/bin to PATH in ~/.bashrc if not already present
  local bashrc="$HOME/.bashrc"
  touch "$bashrc"
  if ! grep -qE '(~|[$]HOME)/.local/bin' "$bashrc"; then
    printf '\n# Added by mindlm install\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$bashrc"
    success "Added ~/.local/bin to PATH in ~/.bashrc"
    warn "Run 'source ~/.bashrc' or open a new shell to apply the PATH change."
  else
    info "~/.local/bin is already in PATH (found in ~/.bashrc)."
  fi

  success "Installation complete. Run 'mindlm help' to get started."
}

cmd_uninstall() {
  local link_path="$HOME/.local/bin/mindlm"
  if [[ -L "$link_path" ]]; then
    rm -f "$link_path"
    success "Removed $link_path"
  elif [[ -e "$link_path" ]]; then
    error "$link_path exists but is not a symlink. Remove it manually."
    exit 1
  else
    info "mindlm is not installed at $link_path — nothing to do."
  fi
}

# ── Main dispatch ──────────────────────────────────────────────────────────
main() {
  local cmd="${1:-help}"
  shift || true

  # Dependency checks (skip for help/install/uninstall)
  case "$cmd" in
    help|--help|-h|install|uninstall|config-wizard) ;;
    *) check_deps ;;
  esac

  case "$cmd" in
    start)           cmd_start "$@" ;;
    stop)            cmd_stop "$@" ;;
    build)           cmd_build "$@" ;;
    status)          cmd_status "$@" ;;
    health)          cmd_health "$@" ;;
    collections)     cmd_collections "$@" ;;
    search)          cmd_search "$@" ;;
    ask)             cmd_ask "$@" ;;
    ingest)          cmd_ingest "$@" ;;
    ingest-full)     cmd_ingest_full "$@" ;;
    install)         cmd_install "$@" ;;
    uninstall)       cmd_uninstall "$@" ;;
    config-wizard)   cmd_config_wizard "$@" ;;
    help|--help|-h)  usage; exit 0 ;;
    *)
      error "Unknown command: $cmd"
      usage >&2
      exit 1 ;;
  esac
}

main "$@"
