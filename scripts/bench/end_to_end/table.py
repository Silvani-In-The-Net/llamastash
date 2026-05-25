"""Comprehensive bench summary table.

Pivots the bench JSONs into a model x tool x mode x engine x workload
grid that's easier to scan than the per-page tables the renderer
produces. Single-host data without engine variants degenerates to a
simple tool x mode table; multi-host data with per-engine `host_id`
suffixes (e.g. `<base>-vulkan`, `<base>-rocm`) surfaces the engine
column automatically.

Usage:

  python -m scripts.bench.end_to_end.table          # all hosts
  python -m scripts.bench.end_to_end.table --host deepu-flowz13-arch
  python -m scripts.bench.end_to_end.table --runs-dir /custom/path
  python -m scripts.bench.end_to_end.table --json > pivot.json

Outputs a per-model GitHub-flavoured markdown table to stdout. The
column set is (per workload): decode tok/s, TTFT ms.

The `host_id -> engine` mapping is opinionated to the maintainer's
naming convention (suffixes like `-vulkan`, `-rocm`, `-hip-rocwmma-on`).
Override via `--engine-map host_id=label,host_id=label,...`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .render import discover_runs, load_runs

# Default host_id-suffix → engine label, derived empirically on the
# 2026-05-24 AMD-APU sweep. Add more as new host conventions appear.
DEFAULT_ENGINE_MAP: dict[str, str] = {
  "rocm": "HIP/ROCm",
  "clean70w": "HIP/ROCm",
  "vulkan": "Vulkan",
  "lms-vulkan": "Vulkan",
  "hip-rocwmma-on": "HIP+rocWMMA",
  "hip-rocwmma-off": "HIP/ROCm",
}


def engine_for_host(host_id: str, overrides: dict[str, str]) -> str:
  """Pick an engine label from `host_id`. Exact override wins; then
  suffix match against DEFAULT_ENGINE_MAP; then 'HIP/ROCm' for any
  host with no engine hint in its name."""
  if host_id in overrides:
    return overrides[host_id]
  for suffix, label in DEFAULT_ENGINE_MAP.items():
    if host_id.endswith(suffix):
      return label
  return "HIP/ROCm"


def _fmt(v: Optional[float], width: int, decimals: int = 1) -> str:
  if v is None:
    return f"{'—':>{width}s}"
  if v >= 10_000:
    return f"{v:>{width},.0f}"
  return f"{v:>{width}.{decimals}f}"


def _mean(values: list[float]) -> Optional[float]:
  vals = [v for v in values if v is not None]
  return sum(vals) / len(vals) if vals else None


SIZE_ORDER = ["small", "mid", "large_dense", "large_moe"]
TOOL_ORDER = ["llamastash", "llamacpp", "ollama", "lmstudio"]
MODE_ORDER = ["defaults", "normalized"]
WL_ORDER = ["chat_turn", "agent_decode", "rag_prefill", "parallel_4"]
TOOL_PRETTY = {
  "llamastash": "LlamaStash",
  "llamacpp": "raw llama-server",
  "ollama": "Ollama",
  "lmstudio": "LM Studio",
}
WL_LABEL = {
  "chat_turn": "chat_turn (50p/64d)",
  "agent_decode": "agent_decode (50p/256d)",
  "rag_prefill": "rag_prefill (8157p/64d)",
  "parallel_4": "parallel_4 (4× chat_turn)",
}


def parse_engine_map(s: Optional[str]) -> dict[str, str]:
  if not s:
    return {}
  out: dict[str, str] = {}
  for pair in s.split(","):
    if "=" not in pair:
      continue
    k, _, v = pair.partition("=")
    out[k.strip()] = v.strip()
  return out


def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    prog="bench-table",
    description="Pivot bench JSONs into a per-model summary table.",
  )
  p.add_argument(
    "--runs-dir",
    type=Path,
    default=Path("docs/benchmarks/runs"),
    help="Root of `<host_id>/<date>-<hms>-<sha>.json` files.",
  )
  p.add_argument(
    "--host",
    default=None,
    help=(
      "Only include cells whose host_id starts with this prefix. Useful "
      "for filtering a multi-host runs dir to one machine."
    ),
  )
  p.add_argument(
    "--engine-map",
    default=None,
    help=(
      "Comma-separated `host_id=label` overrides, e.g. "
      "`my-host-foo=Metal,my-host-bar=CUDA`. Overrides take precedence "
      "over the built-in suffix mapping."
    ),
  )
  p.add_argument(
    "--json",
    action="store_true",
    help="Emit the pivot as JSON to stdout (one record per cell) instead of markdown.",
  )
  return p


def collect(
  runs_dir: Path,
  host_prefix: Optional[str],
  overrides: dict[str, str],
) -> dict[tuple[str, str, str, str, str], list[tuple[Optional[float], Optional[float]]]]:
  """Return {(size, tool, mode, engine, workload): [(decode, ttft), ...]}."""
  by_key: dict[tuple[str, str, str, str, str], list[tuple[Optional[float], Optional[float]]]] = (
    defaultdict(list)
  )
  for loaded in load_runs(discover_runs(runs_dir)):
    host = loaded.report.host.host_id
    if host_prefix and not host.startswith(host_prefix):
      continue
    engine = engine_for_host(host, overrides)
    for c in loaded.report.cells:
      if c.summary.measured_rep_count < 1:
        continue
      key = (c.model.size_class, c.tool, c.mode, engine, c.workload)
      by_key[key].append(
        (c.summary.decode_tps_mean, c.summary.ttft_ms_mean)
      )
  return by_key


def emit_markdown(
  by_key: dict[tuple[str, str, str, str, str], list[tuple[Optional[float], Optional[float]]]],
) -> None:
  """Pretty-print the pivot as one markdown table per model."""
  # Engines actually present across all cells, in a stable order.
  present_engines = sorted({k[3] for k in by_key.keys()})
  preferred = ["HIP/ROCm", "Vulkan", "HIP+rocWMMA", "Metal", "CUDA"]
  engine_order = [e for e in preferred if e in present_engines] + sorted(
    e for e in present_engines if e not in preferred
  )

  for size in SIZE_ORDER:
    size_keys = [k for k in by_key if k[0] == size]
    if not size_keys:
      continue
    print(f"\n### {size}\n")
    hdr = "| Tool | Mode | Engine |"
    sep = "|---|---|---|"
    for wl in WL_ORDER:
      hdr += f" {WL_LABEL[wl]} decode tok/s | TTFT ms |"
      sep += "---:|---:|"
    print(hdr)
    print(sep)
    for tool in TOOL_ORDER:
      for mode in MODE_ORDER:
        for engine in engine_order:
          # Ollama uses its own bundled engine; squash to "bundled"
          # rather than the host's suffix-derived label so the table
          # doesn't duplicate rows for what's physically one config.
          row_engine_label = "bundled" if tool == "ollama" else engine
          if tool == "ollama" and engine != engine_order[0]:
            continue
          has_any = any(
            (size, tool, mode, engine, wl) in by_key for wl in WL_ORDER
          )
          if not has_any:
            continue
          row = f"| {TOOL_PRETTY.get(tool, tool)} | {mode} | {row_engine_label} |"
          for wl in WL_ORDER:
            samples = by_key.get((size, tool, mode, engine, wl), [])
            dec = _mean([s[0] for s in samples])
            ttft = _mean([s[1] for s in samples])
            row += f" {_fmt(dec, 6)} | {_fmt(ttft, 7, 0)} |"
          print(row)


def emit_json(
  by_key: dict[tuple[str, str, str, str, str], list[tuple[Optional[float], Optional[float]]]],
) -> None:
  records = []
  for (size, tool, mode, engine, wl), samples in sorted(by_key.items()):
    dec = _mean([s[0] for s in samples])
    ttft = _mean([s[1] for s in samples])
    records.append({
      "size": size,
      "tool": tool,
      "mode": mode,
      "engine": engine,
      "workload": wl,
      "decode_tps_mean": dec,
      "ttft_ms_mean": ttft,
      "n_runs": len(samples),
    })
  json.dump(records, sys.stdout, indent=2)
  sys.stdout.write("\n")


def main(argv: Optional[list[str]] = None) -> int:
  args = build_arg_parser().parse_args(argv)
  overrides = parse_engine_map(args.engine_map)
  by_key = collect(args.runs_dir, args.host, overrides)
  if not by_key:
    print("no cells found", file=sys.stderr)
    return 1
  if args.json:
    emit_json(by_key)
  else:
    emit_markdown(by_key)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
