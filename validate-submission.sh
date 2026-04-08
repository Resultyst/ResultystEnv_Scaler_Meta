#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator (ResultystEnv)
# Adapted from the official hackathon pre-validation script.
#
# Usage:
#   ./validate-submission.sh <ping_url> [repo_dir]
#
# Examples:
#   ./validate-submission.sh https://your-space.hf.space
#   ./validate-submission.sh https://your-space.hf.space ./
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BOLD=''; NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then gtimeout "$secs" "$@"
  else
    "$@" & local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) & local watcher=$!
    wait "$pid" 2>/dev/null; local rc=$?
    kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null; return $rc
  fi
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

portable_mktemp() { mktemp "${TMPDIR:-/tmp}/${1:-validate}-XXXXXX" 2>/dev/null || mktemp; }

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"; exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"; exit 1
fi
PING_URL="${PING_URL%/}"

PASS=0
log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() { printf "\n${RED}${BOLD}Stopped at %s.${NC} Fix above before continuing.\n" "$1"; exit 1; }

printf "\n${BOLD}========================================${NC}\n"
printf "${BOLD}  ResultystEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

# ── Step 1: HF Space live ──
log "${BOLD}Step 1/4: Pinging HF Space${NC} ($PING_URL/reset) ..."
CURL_OUT=$(portable_mktemp "validate-curl"); CLEANUP_FILES+=("$CURL_OUT")
HTTP_CODE=$(curl -s -o "$CURL_OUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space live — /reset returned 200"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure the Space is running and try: curl -X POST $PING_URL/reset -d '{}'"
  stop_at "Step 1"
fi

# ── Step 2: openenv.yaml present ──
log "${BOLD}Step 2/4: Checking openenv.yaml${NC} ..."
if [ -f "$REPO_DIR/openenv.yaml" ]; then
  # Basic field checks
  if grep -q "^name:" "$REPO_DIR/openenv.yaml" && grep -q "^tasks:" "$REPO_DIR/openenv.yaml"; then
    pass "openenv.yaml present with name + tasks fields"
  else
    fail "openenv.yaml is missing required fields (name, tasks)"
    stop_at "Step 2"
  fi
else
  fail "openenv.yaml not found in repo root"
  stop_at "Step 2"
fi

# ── Step 3: Docker build ──
log "${BOLD}Step 3/4: Running docker build${NC} ..."
if ! command -v docker &>/dev/null; then
  fail "docker not found — install from https://docs.docker.com/get-docker/"
  stop_at "Step 3"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then DOCKER_CONTEXT="$REPO_DIR/server"
else fail "No Dockerfile found"; stop_at "Step 3"; fi

BUILD_OK=false
BUILD_OUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true
if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"; printf "%s\n" "$BUILD_OUT" | tail -20; stop_at "Step 3"
fi

# ── Step 4: inference.py present ──
log "${BOLD}Step 4/4: Checking inference.py${NC} ..."
if [ -f "$REPO_DIR/inference.py" ]; then
  if grep -q "log_start" "$REPO_DIR/inference.py" && grep -q "log_step" "$REPO_DIR/inference.py" && grep -q "log_end" "$REPO_DIR/inference.py"; then
    pass "inference.py present with required log_start/log_step/log_end functions"
  else
    fail "inference.py missing log_start, log_step, or log_end"
    stop_at "Step 4"
  fi
else
  fail "inference.py not found in repo root"
  stop_at "Step 4"
fi

printf "\n${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 4/4 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  ResultystEnv is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n\n"
exit 0
