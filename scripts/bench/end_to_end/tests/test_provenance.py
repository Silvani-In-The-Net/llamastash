"""Provenance capture tests.

The contract: best-effort capture, never raises. Missing binaries
land as `None` on the relevant Provenance fields. Host capture
always returns a populated Host (some optional fields may be None,
but the required ones — host_id, os, cpu — are always present).

We avoid mocking subprocess.run wholesale; instead we shadow PATH
via tmp_path to provide / withhold fake binaries. That exercises
the same shutil.which → subprocess.run pipeline the real call uses.
"""
from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from scripts.bench.end_to_end.provenance import (
  _extract_llama_cpp_commit,
  capture_host,
  capture_provenance,
)


# ---- Fake-binary helpers -----------------------------------------


def _write_fake_bin(path: Path, stdout: str, exit_code: int = 0) -> None:
  """Write an executable Python script that prints `stdout` and
  exits with `exit_code`. Using Python (rather than bash) keeps the
  fake binaries self-contained — we can wipe the PATH down to just
  the temp dir without losing the interpreter the fakes need to run.
  `sys.executable` is the venv Python, which is always available in
  the test environment."""
  py = sys.executable
  body = f"#!{py}\nimport sys\nsys.stdout.write({stdout!r} + '\\n')\nsys.exit({exit_code})\n"
  path.write_text(body)
  path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def isolated_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """Replace PATH with a temp-only directory so only the fake
  binaries we plant are discoverable. Tools the bench cares about
  (`llamastash`, `llama-server`, `ollama`, `lms`) must NOT leak in
  from the dev machine's real install, or the "no tools" assertions
  fail nondeterministically."""
  fake_bin = tmp_path / "bin"
  fake_bin.mkdir()
  monkeypatch.setenv("PATH", str(fake_bin))
  return fake_bin


# ---- Provenance: tool versions -----------------------------------


def test_capture_provenance_returns_none_when_no_tools(isolated_path: Path) -> None:
  prov = capture_provenance()
  assert prov.llamastash_version is None
  assert prov.llama_server_version is None
  assert prov.llama_cpp_commit is None
  assert prov.ollama_version is None
  assert prov.lmstudio_version is None
  # python_version always populates from sys.version.
  assert prov.python_version is not None


def test_capture_provenance_populates_when_all_tools_present(isolated_path: Path) -> None:
  _write_fake_bin(isolated_path / "llamastash", "llamastash 0.2.0")
  _write_fake_bin(
    isolated_path / "llama-server",
    "version: 3705 (b6e7c5a)\nbuilt with cc (GCC) 14.2.1",
  )
  _write_fake_bin(isolated_path / "ollama", "ollama version is 0.3.10")
  _write_fake_bin(isolated_path / "lms", "0.3.5 (build 22)")

  prov = capture_provenance()
  assert prov.llamastash_version == "llamastash 0.2.0"
  assert prov.llama_server_version == "version: 3705 (b6e7c5a)"
  assert prov.llama_cpp_commit == "b6e7c5a"
  assert prov.ollama_version == "ollama version is 0.3.10"
  assert prov.lmstudio_version == "0.3.5 (build 22)"


def test_capture_provenance_handles_partial_tool_set(isolated_path: Path) -> None:
  _write_fake_bin(isolated_path / "llama-server", "version: 1000 (abc1234)")
  prov = capture_provenance()
  assert prov.llamastash_version is None
  assert prov.llama_server_version == "version: 1000 (abc1234)"
  assert prov.llama_cpp_commit == "abc1234"
  assert prov.ollama_version is None
  assert prov.lmstudio_version is None


def test_capture_provenance_does_not_raise_on_tool_failure(
  isolated_path: Path,
) -> None:
  """A misbehaving binary (exits non-zero, prints garbage) must not
  abort the capture — the field just lands as a best-effort string
  or `None`."""
  _write_fake_bin(isolated_path / "ollama", "garbage output line", exit_code=7)

  # Should not raise. On non-zero exit `_run` still returns stdout
  # (best-effort); the resulting string is whatever the tool said.
  prov = capture_provenance()
  assert prov.ollama_version == "garbage output line"


# ---- llama.cpp commit extraction ---------------------------------


@pytest.mark.parametrize(
  "raw,expected",
  [
    ("version: 3705 (b6e7c5a)", "b6e7c5a"),
    ("llama-server version 3705 (deadbee)\nbuilt ...", "deadbee"),
    ("ollama version is 0.3.10\nllama.cpp: 1a2b3c4d5e", "1a2b3c4d5e"),
    ("ollama version is 0.3.10", None),  # no commit anywhere
    ("", None),
  ],
)
def test_extract_llama_cpp_commit(raw: str, expected: str | None) -> None:
  assert _extract_llama_cpp_commit(raw) == expected


# ---- Host capture -------------------------------------------------


def test_capture_host_always_populates_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
  """Even on a stripped PATH, host capture must produce something
  non-empty for the required fields. Optional fields may be None."""
  monkeypatch.setenv("PATH", "")
  host = capture_host()
  assert host.host_id
  assert host.os
  assert host.cpu
  assert host.cpu_threads >= 1
  assert host.gpu_backend in ("cuda", "rocm", "metal", "vulkan", "cpu")


def test_capture_host_honours_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("LLAMASTASH_BENCH_HOST_ID", "Bench-Host.Lan")
  monkeypatch.setenv("LLAMASTASH_BENCH_GPU_BACKEND", "ROCm")
  host = capture_host()
  # host_id lowercased + sanitized to alnum/-_.
  assert host.host_id == "bench-host"
  # backend forced via env; backend label is lowercased.
  assert host.gpu_backend == "rocm"


def test_capture_host_sanitizes_unsafe_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("LLAMASTASH_BENCH_HOST_ID", "weird/host name!!")
  host = capture_host()
  # Slashes and spaces become hyphens; trailing punctuation collapsed.
  assert "/" not in host.host_id
  assert " " not in host.host_id
  assert host.host_id  # still non-empty
