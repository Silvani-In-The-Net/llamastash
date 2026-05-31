#!/usr/bin/env bash
# =============================================================================
# LlamaStash — one-shot UAT + benchmark runner for a handoff machine.
#
# Hand this to someone with the cloned repo. They run ONE command; it produces
# a single tarball of JSON reports to send back. Everything that can be derived
# (OS, arch, GPU backend, model path, ports, Python venv) is derived — the only
# things you might pre-install are Ollama and LM Studio (see PREREQUISITES).
#
# What it does, in order:
#   1. Preflight  — check toolchain, detect OS/arch + GPU backend, build venv.
#   2. Install    — `cargo install` the `llamastash` binary (with the `uat`
#                   feature) onto PATH.
#   3. Stage      — ensure `llama-server` is on PATH (installs it via
#                   `llamastash init` if missing) and pull the pinned model once.
#   4. UAT        — hardware lifecycle, warm mode  ->  uat-<backend>-warm.json
#   5. Benchmarks — Suite B (cross-tool), Suite A (overhead), Suite C (proxy).
#   6. Collect    — bundle every JSON + a host summary into one .tar.gz.
#
# Suites whose optional tools aren't present are SKIPPED, not failed, and one
# failing step never aborts the rest — you always get whatever was produced.
#
# -----------------------------------------------------------------------------
# PREREQUISITES (recipient)
#   Required : Rust toolchain (https://rustup.rs), git, make, a C/C++ compiler.
#              Python 3 OR uv (the bench harness builds its own venv).
#   Optional : Ollama       — install + ensure `ollama` daemon is running.
#              LM Studio     — install the app, run `~/.lmstudio/bin/lms
#                              bootstrap` so the `lms` CLI is on PATH, then
#                              DOWNLOAD this exact model inside LM Studio so it
#                              lands in the library:
#                                  Qwen2.5-0.5B-Instruct-GGUF  (Q4_K_M)
#   GPU drivers as usual for your card (NVIDIA / ROCm / Metal / Vulkan).
#
# USAGE
#   ./run-uat-and-bench.sh
#
# OVERRIDES (env vars, all optional)
#   HOST_BACKEND=nvidia|amd|apple_metal|vulkan|cpu_only   # skip GPU auto-detect
#   BENCH_MODELS=small[,mid,large_dense]                  # default: small
#   SKIP_CROSS_TOOL=1     # benchmark only llamastash vs raw llama.cpp
#   SKIP_UAT=1            # benchmarks only
#   SKIP_BENCH=1          # UAT only
#   OUT_DIR=/path/to/dir  # where reports + the tarball are written
# =============================================================================
set -euo pipefail

# --- pinned reference model (matches the UAT reference; small, ~469 MiB) -----
MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_REVISION="9217f5db79a29953eb74d5343926648285ec7e67"
MODEL_REF="${MODEL_REPO}:${MODEL_FILE}"
BENCH_MODELS="${BENCH_MODELS:-small}"

# --- pretty logging ----------------------------------------------------------
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fatal]\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- locate repo + output dir ------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$REPO_ROOT" ] || die "run this from inside the cloned llamastash git repo"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/handoff-results-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"
RUN_LOG="$OUT_DIR/run.log"
: > "$RUN_LOG"
START_MARKER="$(mktemp)"; trap 'rm -f "$START_MARKER"' EXIT

# Run a step, tee its result to the run log, and keep going on failure.
run_step() {
  local name="$1"; shift
  log "$name"
  if "$@" 2>&1 | tee -a "$RUN_LOG"; then
    echo "PASS: $name" >> "$RUN_LOG"
  else
    warn "$name FAILED — continuing"
    echo "FAIL: $name" >> "$RUN_LOG"
  fi
}

# =============================================================================
# 1. Preflight
# =============================================================================
log "Preflight"
have git   || die "git not found"
have cargo || die "Rust toolchain not found — install from https://rustup.rs"
have make  || die "make not found"
have python3 || have uv || die "need python3 or uv for the benchmark harness venv"
OS="$(uname -s)"; ARCH="$(uname -m)"
log "Host: OS=$OS ARCH=$ARCH"

# Bootstrap the bench harness venv up front; we reuse its python for JSON.
log "Bootstrapping benchmark venv (.venv) — uv if present, else python3 -m venv"
make .venv/bin/python >>"$OUT_DIR/venv.log" 2>&1 || warn "venv bootstrap had issues (see venv.log)"
PY="$REPO_ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3 || true)"

json_get() { # usage: echo "$json" | json_get a.b.c
  [ -x "$PY" ] || { echo ""; return; }
  "$PY" - "$1" <<'PY' 2>/dev/null || echo ""
import sys, json
d = None
try:
    d = json.load(sys.stdin)
    for k in sys.argv[1].split("."):
        d = d.get(k) if isinstance(d, dict) else None
except Exception:
    d = None
print(d if d is not None else "")
PY
}

# --- GPU backend (matches llamastash's probe order: nvidia>amd>metal>vulkan) --
detect_backend() {
  [ -n "${HOST_BACKEND:-}" ] && { echo "$HOST_BACKEND"; return; }
  if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then echo nvidia; return; fi
  if { have rocm-smi && rocm-smi >/dev/null 2>&1; } || { have rocminfo && rocminfo >/dev/null 2>&1; }; then echo amd; return; fi
  if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then echo apple_metal; return; fi
  if have vulkaninfo && vulkaninfo --summary 2>/dev/null | grep -qiE 'deviceName|GPU'; then echo vulkan; return; fi
  echo cpu_only
}
BACKEND="$(detect_backend)"
log "GPU backend: $BACKEND  (override with HOST_BACKEND=… if the UAT preflight disagrees)"

# =============================================================================
# 2. Build + install the llamastash binary (with the uat feature) onto PATH
# =============================================================================
log "Building + installing llamastash (this can take a few minutes)"
cargo install --path . --bin llamastash --features uat --force 2>&1 | tee -a "$OUT_DIR/build.log"
export PATH="$HOME/.cargo/bin:$PATH"
have llamastash || die "llamastash not on PATH after cargo install"
llamastash --version | tee -a "$RUN_LOG"

# =============================================================================
# 3. Stage llama-server + the reference model
# =============================================================================
LLAMA_SERVER_BIN=""
if have llama-server; then
  LLAMA_SERVER_BIN="$(command -v llama-server)"
  log "Found llama-server on PATH: $LLAMA_SERVER_BIN"
else
  log "Installing llama-server via 'llamastash init --only server'"
  init_json="$(llamastash init --recommended --no-tui --only server --json 2>>"$OUT_DIR/init.log" || true)"
  printf '%s\n' "$init_json" > "$OUT_DIR/init-summary.json"
  LLAMA_SERVER_BIN="$(printf '%s' "$init_json" | json_get install.path)"
  [ -n "$LLAMA_SERVER_BIN" ] && [ -x "$LLAMA_SERVER_BIN" ] \
    || die "llama-server install failed — install llama.cpp manually (so 'llama-server' is on PATH) and re-run"
  server_dir="$(dirname "$LLAMA_SERVER_BIN")"
  export PATH="$server_dir:$PATH"   # bench drivers resolve llama-server from PATH
fi
export LLAMASTASH_LLAMA_SERVER="$LLAMA_SERVER_BIN"

log "Pulling reference model: $MODEL_REF"
llamastash pull "$MODEL_REF" --revision "$MODEL_REVISION" >>"$OUT_DIR/pull.log" 2>&1 \
  || llamastash pull "$MODEL_REF" >>"$OUT_DIR/pull.log" 2>&1 \
  || warn "model pull failed (see pull.log)"

HF_ROOT="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
GGUF="$(find "$HF_ROOT" -name "$MODEL_FILE" 2>/dev/null | head -n1)"
[ -n "$GGUF" ] || GGUF="$(find "$HOME" -name "$MODEL_FILE" 2>/dev/null | head -n1)"
if [ -n "$GGUF" ]; then log "Model on disk: $GGUF"; else warn "could not locate $MODEL_FILE — UAT uses HF download, bench suites that need a path are skipped"; fi

# =============================================================================
# 4. UAT (warm mode)
# =============================================================================
if [ -z "${SKIP_UAT:-}" ]; then
  UAT_REPORT="$OUT_DIR/uat-${BACKEND}-warm.json"
  uat_cmd=( llamastash uat --host-backend "$BACKEND" --mode warm --report-out "$UAT_REPORT" )
  [ -n "$GGUF" ] && uat_cmd+=( --local-gguf "$GGUF" )
  run_step "UAT ($BACKEND, warm)" "${uat_cmd[@]}"
  [ -f "$UAT_REPORT" ] && log "UAT report: $UAT_REPORT"
else
  log "SKIP_UAT set — skipping UAT"
fi

# =============================================================================
# 5. Benchmarks
# =============================================================================
if [ -z "${SKIP_BENCH:-}" ] && [ -n "$GGUF" ]; then
  touch "$START_MARKER"

  # Which tools to compare. llamastash + raw llama.cpp always run; Ollama and
  # LM Studio are added only when their CLIs respond (so a missing tool can't
  # take the whole suite down).
  TOOLS="llamastash,llamacpp"
  if [ -z "${SKIP_CROSS_TOOL:-}" ]; then
    if have ollama && ollama list >/dev/null 2>&1; then
      TOOLS="$TOOLS,ollama"
    else
      warn "Ollama not detected/running — excluded from the cross-tool suite"
    fi
    if have lms && lms ls >/dev/null 2>&1; then
      TOOLS="$TOOLS,lmstudio"
    else
      warn "LM Studio 'lms' not ready — excluded. (Install LM Studio, run 'lms bootstrap', and download $MODEL_FILE in the app.)"
    fi
  fi
  log "Cross-tool set: $TOOLS"

  run_step "Suite B — cross-tool ($TOOLS)" \
    env LLAMASTASH_BENCH_MODELS_SMALL="$GGUF" \
    scripts/bench/end_to_end/run.sh --tools "$TOOLS" --models "$BENCH_MODELS"

  run_step "Suite A — overhead (llamastash vs raw llama-server)" \
    scripts/bench/overhead/run.sh --model "$GGUF"

  run_step "Suite C — proxy (proxy vs direct)" \
    scripts/bench/proxy/run.sh --model "$GGUF"
elif [ -n "${SKIP_BENCH:-}" ]; then
  log "SKIP_BENCH set — skipping benchmarks"
else
  warn "no local GGUF — skipping benchmark suites"
fi

# =============================================================================
# 6. Collect everything into one tarball
# =============================================================================
log "Collecting reports"
mkdir -p "$OUT_DIR/bench"
if [ -d docs/benchmarks ]; then
  while IFS= read -r f; do
    rel="${f#./}"; dest="$OUT_DIR/bench/$rel"
    mkdir -p "$(dirname "$dest")"; cp "$f" "$dest"
  done < <(find docs/benchmarks -type f -name '*.json' -newer "$START_MARKER" 2>/dev/null || true)
fi

{
  echo "date:        $(date -u +%FT%TZ)"
  echo "os/arch:     $OS / $ARCH"
  echo "backend:     $BACKEND"
  echo "git_sha:     $(git rev-parse HEAD 2>/dev/null)"
  echo "llamastash:  $(llamastash --version 2>/dev/null)"
  echo "llama_server:$("$LLAMA_SERVER_BIN" --version 2>&1 | head -n1)"
  echo "tools:       ${TOOLS:-n/a}"
  echo "model:       $MODEL_REF"
  echo "gguf_path:   ${GGUF:-not found}"
} > "$OUT_DIR/host-summary.txt"

TARBALL="$REPO_ROOT/$(basename "$OUT_DIR").tar.gz"
tar -czf "$TARBALL" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"

log "DONE"
printf '\nResults dir : %s\n' "$OUT_DIR"
printf 'Send back   : %s\n\n' "$TARBALL"
