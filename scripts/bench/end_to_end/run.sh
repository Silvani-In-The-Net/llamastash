#!/usr/bin/env bash
# Suite B (cross-tool end-to-end) bench wrapper. Auto-detects the
# GPU backend, ensures the venv is bootstrapped (`make .venv/bin/python`),
# and forwards extra args straight to the Python orchestrator.
#
# See docs/benchmarks/methodology.md for the per-tool fairness notes
# and the full settings policy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

VENV_PY="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    echo "==> bootstrapping .venv/bin/python (uv preferred, stdlib fallback)" >&2
    (cd "$REPO_ROOT" && make .venv/bin/python)
fi

if [[ ! -x "$VENV_PY" ]]; then
    echo "error: .venv/bin/python missing after bootstrap. Run 'make .venv/bin/python' manually." >&2
    exit 1
fi

cd "$REPO_ROOT"

# The orchestrator reads its own arg list; we just pass through.
# Environment variables honored by the orchestrator and drivers:
#   LLAMASTASH_BENCH_HOST_ID       — override the host-id slug
#   LLAMASTASH_BENCH_GPU_BACKEND   — override autodetected backend
#   LLAMASTASH_BENCH_PORT_BASE     — first free port to try (default 18000)
#   LLAMASTASH_BENCH_READY_TIMEOUT_S — driver readiness timeout (default 180)
#   LLAMASTASH_BENCH_KEEP_IMPORTS  — keep Ollama-imported models on stop()
#   LLAMASTASH_BENCH_MODELS_SMALL  — repo/file override for the small slot
#   LLAMASTASH_BENCH_MODELS_MID    — repo/file override for the mid slot
#   LLAMASTASH_BENCH_MODELS_LARGE_DENSE — large-dense slot override
#   LLAMASTASH_BENCH_MODELS_LARGE_MOE   — large-MoE slot override
exec "$VENV_PY" -m scripts.bench.end_to_end.orchestrator "$@"
