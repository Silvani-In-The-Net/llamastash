"""Pytest entry point for the bench harness package.

Adds the repo root to `sys.path` so `from scripts.bench.end_to_end.X
import ...` works under `pytest scripts/bench/` regardless of where
the maintainer invokes pytest from. We don't add a top-level
`pyproject.toml` for this — the repo is a Rust crate first, and the
Python harness is intentionally a loose-scripts collection (matching
`scripts/measure-overhead-band.py` and `scripts/regenerate-benchmark-snapshot.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))
