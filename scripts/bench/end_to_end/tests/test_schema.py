"""Schema v1 round-trip + validation tests.

Goal: catch drift early. Any cell field the renderer reads must
survive `model_dump_json` → `model_validate_json` byte-equal. Any
schema_version bump must fail the validator until consumers update.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from scripts.bench.end_to_end.schema import (
  Cell,
  Determinism,
  Host,
  ModelSpec,
  Provenance,
  Rep,
  RunReport,
  Summary,
)


def _fixture_host() -> Host:
  return Host(
    host_id="dev-box",
    os="Linux 6.6.5-arch1-1",
    cpu="AMD Ryzen 9 7950X3D",
    cpu_threads=32,
    ram_gb=128.0,
    gpu_backend="cuda",
    gpu_name="NVIDIA RTX 4090",
    gpu_vram_gb=24.0,
  )


def _fixture_provenance() -> Provenance:
  return Provenance(
    llamastash_version="llamastash 0.2.0",
    llama_server_version="version: 3705 (b6e7c5a)",
    llama_cpp_commit="b6e7c5a",
    ollama_version="ollama version 0.3.10",
    lmstudio_version="0.3.5 (build 22)",
    python_version="3.12.4",
  )


def _fixture_cell() -> Cell:
  return Cell(
    tool="llamastash",
    model=ModelSpec(
      size_class="mid",
      hf_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
      hf_file="qwen2.5-7b-instruct-q4_k_m.gguf",
      sha256="a" * 64,
      bytes=4_700_000_000,
    ),
    mode="normalized",
    workload="chat_turn",
    argv_recorded=[
      "llama-server",
      "--host",
      "127.0.0.1",
      "-m",
      "/m/qwen.gguf",
      "-c",
      "4096",
      "--n-gpu-layers",
      "99",
    ],
    reps=[
      Rep(rep_index=0, is_warmup=True, ttft_ms=120.0, decode_tps=50.0),
      Rep(rep_index=1, is_warmup=False, ttft_ms=90.0, decode_tps=55.0),
      Rep(rep_index=2, is_warmup=False, ttft_ms=88.0, decode_tps=56.0),
    ],
    summary=Summary(
      ttft_ms_mean=89.0,
      ttft_ms_stddev_pct=1.1,
      decode_tps_mean=55.5,
      decode_tps_stddev_pct=0.9,
      measured_rep_count=2,
    ),
    determinism=Determinism(
      prompt_sha256="b" * 64,
      first_n_token_ids_sha256="c" * 64,
      n_compared_tokens=64,
      determinism_mismatch=False,
    ),
  )


def _fixture_report() -> RunReport:
  return RunReport(
    suite="end_to_end",
    host=_fixture_host(),
    provenance=_fixture_provenance(),
    started_at_utc="2026-05-21T10:00:00+00:00",
    finished_at_utc="2026-05-21T10:42:00+00:00",
    git_sha="abc123def456",
    cells=[_fixture_cell()],
  )


# ---- Happy path: round-trip --------------------------------------


def test_run_report_round_trip_byte_equal() -> None:
  report = _fixture_report()
  serialized = report.model_dump_json()
  restored = RunReport.model_validate_json(serialized)
  reserialized = restored.model_dump_json()
  assert reserialized == serialized


def test_run_report_round_trip_through_python_json() -> None:
  """`json.loads` ↔ `model_validate` keeps the same data the
  renderer will read off disk."""
  report = _fixture_report()
  raw = json.loads(report.model_dump_json())
  restored = RunReport.model_validate(raw)
  assert restored == report


def test_schema_version_default_pins_to_one() -> None:
  report = _fixture_report()
  assert report.schema_version == 1


# ---- Validation rejections ---------------------------------------


def test_schema_rejects_unknown_top_level_field() -> None:
  """`extra='forbid'` keeps the wire format honest — a stray field
  is a producer bug, not a silent passthrough."""
  raw = json.loads(_fixture_report().model_dump_json())
  raw["bogus_field"] = "leaked"
  with pytest.raises(ValidationError):
    RunReport.model_validate(raw)


def test_schema_rejects_bumped_schema_version() -> None:
  raw = json.loads(_fixture_report().model_dump_json())
  raw["schema_version"] = 2
  with pytest.raises(ValidationError) as excinfo:
    RunReport.model_validate(raw)
  assert "schema_version" in str(excinfo.value)


def test_cell_requires_tool_model_mode() -> None:
  with pytest.raises(ValidationError) as excinfo:
    Cell.model_validate({"summary": {"measured_rep_count": 0}})
  msg = str(excinfo.value)
  assert "tool" in msg
  assert "model" in msg
  assert "mode" in msg


def test_cell_rejects_unknown_tool_value() -> None:
  raw = json.loads(_fixture_cell().model_dump_json())
  raw["tool"] = "kobold_cpp"
  with pytest.raises(ValidationError):
    Cell.model_validate(raw)


def test_model_spec_rejects_short_sha() -> None:
  with pytest.raises(ValidationError):
    ModelSpec(
      size_class="mid",
      hf_repo="x/y",
      hf_file="x.gguf",
      sha256="abc",
      bytes=1,
    )


def test_rep_rejects_negative_metric() -> None:
  with pytest.raises(ValidationError):
    Rep(rep_index=1, ttft_ms=-5.0)


def test_summary_measured_rep_count_required() -> None:
  with pytest.raises(ValidationError):
    Summary()  # type: ignore[call-arg]
