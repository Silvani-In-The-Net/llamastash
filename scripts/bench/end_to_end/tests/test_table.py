"""Unit tests for the bench summary table pivot.

Heavy lifting (markdown formatting) is exercised end-to-end via the
synthetic RunReport fixtures from test_render. Focus here is on:

- engine_for_host suffix mapping + override precedence
- parse_engine_map parsing
- emit_json record shape
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from scripts.bench.end_to_end.table import (
  DEFAULT_ENGINE_MAP,
  emit_json,
  engine_for_host,
  parse_engine_map,
)


def test_engine_for_host_picks_suffix_match() -> None:
  assert engine_for_host("box-rocm", {}) == DEFAULT_ENGINE_MAP["rocm"]
  assert engine_for_host("box-vulkan", {}) == DEFAULT_ENGINE_MAP["vulkan"]
  assert engine_for_host("box-hip-rocwmma-on", {}) == DEFAULT_ENGINE_MAP["hip-rocwmma-on"]


def test_engine_for_host_falls_back_to_default_label() -> None:
  # No recognised suffix — defaults to HIP/ROCm
  assert engine_for_host("plain-host", {}) == "HIP/ROCm"


def test_engine_for_host_override_wins() -> None:
  # Exact override beats the suffix mapping
  assert engine_for_host("box-rocm", {"box-rocm": "Metal"}) == "Metal"


def test_parse_engine_map_handles_csv_pairs() -> None:
  assert parse_engine_map(None) == {}
  assert parse_engine_map("") == {}
  assert parse_engine_map("a=1") == {"a": "1"}
  assert parse_engine_map(" h1 = Metal , h2 = CUDA ") == {"h1": "Metal", "h2": "CUDA"}
  # Malformed pair (no =) is silently dropped
  assert parse_engine_map("nope") == {}


def test_emit_json_round_trip_one_record() -> None:
  pivot = {
    ("small", "llamastash", "defaults", "HIP/ROCm", "chat_turn"): [(82.5, 51.0), (80.1, 52.0)],
  }
  buf = io.StringIO()
  with redirect_stdout(buf):
    emit_json(pivot)
  rows = json.loads(buf.getvalue())
  assert len(rows) == 1
  r = rows[0]
  assert r["size"] == "small"
  assert r["tool"] == "llamastash"
  assert r["mode"] == "defaults"
  assert r["engine"] == "HIP/ROCm"
  assert r["workload"] == "chat_turn"
  assert abs(r["decode_tps_mean"] - 81.3) < 0.001
  assert abs(r["ttft_ms_mean"] - 51.5) < 0.001
  assert r["n_runs"] == 2
