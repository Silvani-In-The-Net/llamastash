"""Renderer + variance-gate tests.

Synthetic in-memory fixtures construct `RunReport`s covering OK /
flagged / dropped cells, plus a determinism mismatch scenario. The
real `docs/benchmarks/runs/` discovery path is exercised via a
tmp-path fixture so we don't depend on a real Suite-B run existing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.bench.end_to_end.render import (
  CellStatus,
  DROP_PCT,
  FLAG_PCT,
  classify_cell,
  discover_runs,
  load_runs,
  main as render_main,
  render_cell_table,
  render_determinism_callouts,
  render_dropped_footer,
  render_results_page,
  update_index,
)
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


def _host(host_id: str = "dev-box", backend: str = "cuda") -> Host:
  return Host(
    host_id=host_id,
    os="Linux 6.6.0",
    cpu="Test CPU",
    cpu_threads=8,
    ram_gb=32.0,
    gpu_backend=backend,  # type: ignore[arg-type]
    gpu_name="Test GPU",
    gpu_vram_gb=24.0,
  )


def _prov() -> Provenance:
  return Provenance(python_version="3.12.0")


def _cell(
  tool: str,
  workload: str,
  decode_mean: float,
  decode_stddev_pct: float,
  ttft_mean: float = 100.0,
  ttft_stddev_pct: float = 5.0,
  size: str = "mid",
  mode: str = "normalized",
  unfair_knobs: list[str] | None = None,
  determinism_mismatch: bool = False,
) -> Cell:
  return Cell(
    tool=tool,  # type: ignore[arg-type]
    model=ModelSpec(
      size_class=size,  # type: ignore[arg-type]
      hf_repo="x/y",
      hf_file="m.gguf",
      sha256="a" * 64,
      bytes=1,
    ),
    mode=mode,  # type: ignore[arg-type]
    workload=workload,  # type: ignore[arg-type]
    reps=[Rep(rep_index=1, decode_tps=decode_mean)],
    summary=Summary(
      decode_tps_mean=decode_mean,
      decode_tps_stddev_pct=decode_stddev_pct,
      ttft_ms_mean=ttft_mean,
      ttft_ms_stddev_pct=ttft_stddev_pct,
      measured_rep_count=4,
    ),
    unfair_knobs=unfair_knobs or [],
    determinism=Determinism(determinism_mismatch=determinism_mismatch),
  )


def _report(cells: list[Cell]) -> RunReport:
  return RunReport(
    suite="end_to_end",
    host=_host(),
    provenance=_prov(),
    started_at_utc="2026-05-21T10:00:00+00:00",
    finished_at_utc="2026-05-21T10:30:00+00:00",
    cells=cells,
  )


# ---- classify_cell -----------------------------------------------


def test_classify_clean_under_flag_threshold() -> None:
  c = _cell("llamastash", "chat_turn", 50.0, 5.0)
  assert classify_cell(c) == CellStatus.CLEAN


def test_classify_flag_at_boundary_just_above_10pct() -> None:
  c = _cell("llamastash", "chat_turn", 50.0, 10.5)
  assert classify_cell(c) == CellStatus.FLAGGED


def test_classify_clean_at_flag_threshold_exact() -> None:
  # 10.0% is on the boundary; the gate uses strict `>` so this is CLEAN.
  c = _cell("llamastash", "chat_turn", 50.0, 10.0)
  assert classify_cell(c) == CellStatus.CLEAN


def test_classify_drop_above_25pct() -> None:
  c = _cell("llamastash", "chat_turn", 50.0, 27.0)
  assert classify_cell(c) == CellStatus.DROPPED


def test_classify_uses_worst_metric_across_summary() -> None:
  # decode stddev clean (5%), ttft stddev catastrophic (30%) → DROPPED.
  c = _cell("llamastash", "chat_turn", 50.0, 5.0, ttft_stddev_pct=30.0)
  assert classify_cell(c) == CellStatus.DROPPED


# ---- discover_runs + load_runs ----------------------------------


def test_discover_runs_returns_empty_on_missing_dir(tmp_path: Path) -> None:
  assert discover_runs(tmp_path / "nope") == []


def test_discover_runs_finds_jsons_recursively(tmp_path: Path) -> None:
  (tmp_path / "host-a").mkdir()
  (tmp_path / "host-b" / "subdir").mkdir(parents=True)
  (tmp_path / "host-a" / "2026-01-01-abc.json").write_text("{}")
  (tmp_path / "host-b" / "subdir" / "2026-01-02-def.json").write_text("{}")
  (tmp_path / "host-a" / "README.md").write_text("not json")
  paths = discover_runs(tmp_path)
  assert len(paths) == 2
  assert all(p.suffix == ".json" for p in paths)


def test_load_runs_raises_on_schema_mismatch(tmp_path: Path) -> None:
  bad = tmp_path / "bad.json"
  bad.write_text(json.dumps({"schema_version": 2, "suite": "end_to_end"}))
  with pytest.raises(ValidationError):
    load_runs([bad])


def test_load_runs_succeeds_on_valid_report(tmp_path: Path) -> None:
  good = tmp_path / "good.json"
  rep = _report([_cell("llamastash", "chat_turn", 50.0, 5.0)])
  good.write_text(rep.model_dump_json())
  runs = load_runs([good])
  assert len(runs) == 1
  assert runs[0].report.cells[0].summary.decode_tps_mean == 50.0


# ---- render_cell_table ------------------------------------------


def test_render_cell_table_includes_clean_and_flagged_omits_dropped() -> None:
  cells = [
    _cell("llamastash", "chat_turn", 50.0, 5.0),  # clean
    _cell("llamacpp", "chat_turn", 48.0, 15.0),  # flagged
    _cell("ollama", "chat_turn", 40.0, 30.0),  # dropped
  ]
  table = render_cell_table(cells)
  assert "LlamaStash" in table
  assert "llama-server (raw)" in table
  assert "Ollama" not in table  # dropped → not in table
  assert "±15%" in table  # flagged → inline ±


def test_render_cell_table_surfaces_unfair_knobs_per_row() -> None:
  cells = [
    _cell("lmstudio", "chat_turn", 30.0, 5.0, unfair_knobs=["flash_attn", "ubatch_size"]),
  ]
  table = render_cell_table(cells)
  assert "flash_attn" in table
  assert "ubatch_size" in table


def test_render_dropped_footer_lists_each_dropped_cell() -> None:
  cells = [
    _cell("llamastash", "chat_turn", 50.0, 5.0),
    _cell("ollama", "agent_decode", 40.0, 30.0),
    _cell("lmstudio", "parallel_4", 25.0, 27.0),
  ]
  footer = render_dropped_footer(cells)
  assert "Re-run needed" in footer
  assert "agent_decode" in footer
  assert "parallel_4" in footer
  assert "chat_turn" not in footer  # only dropped ones


def test_render_dropped_footer_empty_when_no_drops() -> None:
  cells = [_cell("llamastash", "chat_turn", 50.0, 5.0)]
  assert render_dropped_footer(cells) == ""


# ---- determinism callouts ---------------------------------------


def test_render_determinism_callouts_surfaces_mismatches() -> None:
  cells = [
    _cell("llamastash", "chat_turn", 50.0, 5.0, determinism_mismatch=True),
    _cell("llamacpp", "chat_turn", 50.0, 5.0, determinism_mismatch=False),
  ]
  out = render_determinism_callouts(cells)
  assert "Determinism mismatches" in out
  assert "LlamaStash" in out
  assert "llama-server" not in out  # the matching one


def test_render_determinism_callouts_empty_when_all_match() -> None:
  cells = [_cell("llamastash", "chat_turn", 50.0, 5.0)]
  assert render_determinism_callouts(cells) == ""


# ---- update_index -----------------------------------------------


def test_update_index_creates_file_when_missing(tmp_path: Path) -> None:
  idx = tmp_path / "index.md"
  update_index(idx, "2026-05-21", tmp_path / "results-2026-05-21.md")
  body = idx.read_text()
  assert "[2026-05-21](results-2026-05-21.md)" in body


def test_update_index_prepends_under_results_heading(tmp_path: Path) -> None:
  idx = tmp_path / "index.md"
  idx.write_text(
    "# benchmarks\n\n## Results\n\n- [2026-04-01](results-2026-04-01.md)\n"
  )
  update_index(idx, "2026-05-21", Path("results-2026-05-21.md"))
  body = idx.read_text()
  assert body.index("2026-05-21") < body.index("2026-04-01"), "newer entry must come first"


def test_update_index_idempotent_on_same_entry(tmp_path: Path) -> None:
  idx = tmp_path / "index.md"
  idx.write_text("# x\n\n## Results\n\n")
  update_index(idx, "2026-05-21", Path("results-2026-05-21.md"))
  update_index(idx, "2026-05-21", Path("results-2026-05-21.md"))
  body = idx.read_text()
  # The full link entry (`- [2026-05-21](results-2026-05-21.md)`)
  # must appear exactly once even after a re-run; the date string
  # itself appears twice per entry (label + path).
  assert body.count("- [2026-05-21](results-2026-05-21.md)") == 1


def test_update_index_drops_placeholder_line(tmp_path: Path) -> None:
  idx = tmp_path / "index.md"
  idx.write_text(
    "# x\n\n## Results\n\n*(no results page yet — first one lands soon.)*\n"
  )
  update_index(idx, "2026-05-21", Path("results-2026-05-21.md"))
  body = idx.read_text()
  assert "no results page yet" not in body.lower()


# ---- end-to-end render -----------------------------------------


def test_render_results_page_writes_charts_and_table(tmp_path: Path) -> None:
  charts_dir = tmp_path / "charts"
  rep = _report(
    [
      _cell("llamastash", "chat_turn", 55.0, 4.0),
      _cell("llamacpp", "chat_turn", 56.0, 5.0),
      _cell("ollama", "chat_turn", 42.0, 15.0),  # flagged
      _cell("lmstudio", "chat_turn", 38.0, 27.0),  # dropped
    ]
  )
  loaded = load_runs([_dump(rep, tmp_path / "host-x" / "2026-05-21-abc.json")])
  body = render_results_page(loaded, "2026-05-21", charts_dir)

  # Headline section per (model, workload).
  assert "## mid — chat_turn" in body
  # Inline ± on the flagged cell.
  assert "±15%" in body
  # Dropped cell isn't in the main table, but is in the footer.
  assert "LM Studio" not in render_cell_table(rep.cells)
  assert "Re-run needed" in body
  # Charts written to disk.
  assert (charts_dir / "mid-chat_turn-decode.svg").exists()
  assert (charts_dir / "mid-chat_turn-ttft.svg").exists()


def test_main_dry_run_writes_nothing(tmp_path: Path) -> None:
  runs_dir = tmp_path / "runs"
  runs_dir.mkdir()
  rep = _report([_cell("llamastash", "chat_turn", 50.0, 5.0)])
  (runs_dir / "host-x").mkdir()
  (runs_dir / "host-x" / "2026-05-21-abc.json").write_text(rep.model_dump_json())
  out_dir = tmp_path / "out"
  rc = render_main(
    [
      "--date",
      "2026-05-21",
      "--runs-dir",
      str(runs_dir),
      "--out-dir",
      str(out_dir),
      "--index",
      str(out_dir / "index.md"),
      "--dry-run",
    ]
  )
  assert rc == 0
  assert not out_dir.exists()


def test_main_writes_results_page_and_updates_index(tmp_path: Path) -> None:
  runs_dir = tmp_path / "runs"
  runs_dir.mkdir()
  rep = _report([_cell("llamastash", "chat_turn", 50.0, 5.0)])
  (runs_dir / "host-x").mkdir()
  (runs_dir / "host-x" / "2026-05-21-abc.json").write_text(rep.model_dump_json())
  out_dir = tmp_path / "out"
  index = out_dir / "index.md"
  rc = render_main(
    [
      "--date",
      "2026-05-21",
      "--runs-dir",
      str(runs_dir),
      "--out-dir",
      str(out_dir),
      "--index",
      str(index),
    ]
  )
  assert rc == 0
  results_file = out_dir / "results-2026-05-21.md"
  assert results_file.exists()
  assert "## mid — chat_turn" in results_file.read_text()
  assert "[2026-05-21]" in index.read_text()


# ---- helpers ----------------------------------------------------


def _dump(report: RunReport, path: Path) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(report.model_dump_json())
  return path
