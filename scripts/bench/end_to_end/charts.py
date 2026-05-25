"""Matplotlib SVG renderer for the headline charts (R138).

SVG only — no JS, no interactive widgets. Any reader can re-render
the same chart from the same JSON to verify it wasn't doctored.

Matplotlib's "Agg" / "svg" backends are headless and don't need a
display, so this works in CI / over SSH without a setup line.

We pull matplotlib in lazily so the rest of the harness (drivers,
workloads, schema) doesn't pay the import cost on every CLI tab-
completion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .schema import Cell


def _setup_matplotlib():
  import matplotlib

  matplotlib.use("svg", force=True)
  import matplotlib.pyplot as plt

  return plt


TOOL_COLORS = {
  "llamastash": "#e69138",
  "llamacpp": "#6aa84f",
  "ollama": "#3d85c6",
  "lmstudio": "#a64d79",
}


def _tool_label(tool: str) -> str:
  return {
    "llamastash": "LlamaStash",
    "llamacpp": "llama-server (raw)",
    "ollama": "Ollama",
    "lmstudio": "LM Studio",
  }.get(tool, tool)


def render_decode_tps_bar(
  cells: Iterable[Cell],
  model_label: str,
  backend_label: str,
  workload: str,
  out_path: Path,
) -> Path:
  """One bar per tool — `decode_tps_mean` in normalized mode for the
  given (model, backend, workload). Cells whose summary lacks a mean
  are skipped silently (the renderer's variance gate already handled
  the user-visible explanation). Returns the written path so callers
  can embed it."""
  plt = _setup_matplotlib()
  rows = [
    (c.tool, c.summary.decode_tps_mean or 0.0, c.summary.decode_tps_stddev_pct or 0.0)
    for c in cells
    if c.workload == workload and c.mode == "normalized" and c.summary.decode_tps_mean
  ]
  rows.sort(key=lambda r: -r[1])  # highest tps first

  fig, ax = plt.subplots(figsize=(7.0, 3.5))
  if not rows:
    ax.text(
      0.5,
      0.5,
      "no measured cells",
      ha="center",
      va="center",
      transform=ax.transAxes,
      color="#666",
    )
    ax.set_axis_off()
  else:
    tools = [_tool_label(t) for t, _, _ in rows]
    means = [m for _, m, _ in rows]
    err = [(m * (s / 100.0)) if s else 0.0 for _, m, s in rows]
    colors = [TOOL_COLORS.get(t, "#cccccc") for t, _, _ in rows]
    bars = ax.bar(tools, means, yerr=err, capsize=4, color=colors, edgecolor="#222")
    for bar, m in zip(bars, means):
      ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height(),
        f" {m:,.1f} tok/s",
        ha="center",
        va="bottom",
        fontsize=9,
      )
    ax.set_ylabel("decode tok/s (mean ± stddev)")
    ax.set_title(f"{model_label} — {workload} — {backend_label}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.tight_layout()
  fig.savefig(out_path, format="svg")
  plt.close(fig)
  return out_path


def render_ttft_bar(
  cells: Iterable[Cell],
  model_label: str,
  backend_label: str,
  workload: str,
  out_path: Path,
) -> Path:
  """One bar per tool — `ttft_ms_mean` in normalized mode. Same
  shape as the decode_tps chart; lower is better."""
  plt = _setup_matplotlib()
  rows = [
    (c.tool, c.summary.ttft_ms_mean or 0.0, c.summary.ttft_ms_stddev_pct or 0.0)
    for c in cells
    if c.workload == workload and c.mode == "normalized" and c.summary.ttft_ms_mean
  ]
  rows.sort(key=lambda r: r[1])  # lower TTFT first

  fig, ax = plt.subplots(figsize=(7.0, 3.5))
  if not rows:
    ax.text(0.5, 0.5, "no measured cells", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
  else:
    tools = [_tool_label(t) for t, _, _ in rows]
    means = [m for _, m, _ in rows]
    err = [(m * (s / 100.0)) if s else 0.0 for _, m, s in rows]
    colors = [TOOL_COLORS.get(t, "#cccccc") for t, _, _ in rows]
    bars = ax.bar(tools, means, yerr=err, capsize=4, color=colors, edgecolor="#222")
    for bar, m in zip(bars, means):
      ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height(),
        f" {m:,.0f} ms",
        ha="center",
        va="bottom",
        fontsize=9,
      )
    ax.set_ylabel("TTFT (ms, lower is better)")
    ax.set_title(f"{model_label} — {workload} — {backend_label}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.tight_layout()
  fig.savefig(out_path, format="svg")
  plt.close(fig)
  return out_path


__all__ = ["render_decode_tps_bar", "render_ttft_bar"]
