# Changelog

All notable changes to llamastash will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project intends to follow [SemVer](https://semver.org/spec/v2.0.0.html) starting with the first stable release.

Entries are one-line summaries of noteworthy changes; follow the linked commit or PR for the full story.

## [Unreleased]

_No changes yet._

## [0.0.1] — [Unreleased]

First publicly-installable release. Single `llamastash` binary acts as TUI, CLI, and daemon; distribution lands across Cargo, a Homebrew tap, and a GitHub-hosted install script, with a marketing site at [llamastash.cli.rs](https://llamastash.cli.rs).

- Daemon-on-demand over a `0600` Unix socket with peercred auth; supervises `llama-server` children through `Launching → Loading → Ready / Error → Stopping → Stopped` with three-factor orphan re-adoption.
- GGUF header parser and async scanner for HuggingFace / Ollama / LM Studio caches; model identity is `(canonical path, BLAKE3 of header)`.
- TUI with grouped list + favorites + filter, launch picker, advanced flag panel, clipboard yank, streaming Chat / Embed / Rerank / Logs right pane, five themes (Catppuccin Macchiato default).
- CLI: `list` / `start` / `stop` / `status` / `logs` / `presets` / `favorites` / `daemon` — every read+mutation command supports `--json`, with documented exit codes and an auto-spawn-daemon flow (`--no-spawn` to opt out).
- `llamastash init` first-run wizard (R48): detect → install `llama-server` per OS×GPU → recommend + pull a GGUF → write `config.yaml` with `arch_defaults` → smoke launch → TUI handoff. Per-step `--install` / `--model` / `--config-step` overrides, `--recommended` / `--json` / `--offline` modes, and `--revision <SHA>` to pin HF commits.
- TUI HuggingFace pull dialog (`d` from the model list) — three-stage Search → File picker → Confirm modal backed by HF Hub's `/api/models` (debounced via `FetchClient`, fit-aware ✓/⚠/✗/— glyph column, sharded-set collapse, byte-accurate progress strip, FIFO queue with one active pull). `Ctrl+X` cancels the active pull mid-chunk; `Ctrl+D` deletes the focused GGUF on idle rows only (HF-cache layout deletes the whole repo dir, constrained to the `~/.cache/huggingface/hub` tree). CLI `--offline` / `LLAMASTASH_OFFLINE` flows through every spawned HF task ([`#4`](../../pull/4)).
- Custom theme via the `custom_theme` config block — user-defined palette accepting `#RRGGBB` hex or ANSI names, inheriting unspecified slots from `base:`; joins the `t:theme` cycle once `theme: custom` is set.
- Custom keybindings via the `keybindings:` config block — every TUI `Action` accepts a Kdash-style key-spec override (`ctrl+q`, `shift+tab`, `f1`, …); overrides flow through to live HF-dialog / confirm-popup labels.
- `llamastash doctor` read-only diagnostic with stable finding ids under `--json` (R74); `llamastash pull <owner/repo[:filename.gguf]>` on `hf-hub` (R65); `llamastash recommend` shortcut for hardware-aware GGUF picks ([`adfef21`](../../commit/adfef21)).
- Path-A recommender with VRAM-fit hard filter and composite ranking (benchmark × tok/s × params × recency), backed by a bundled benchmark snapshot refreshed by daily CI and vendored [`whichllm`](https://github.com/Andyyyy64/whichllm) catalog discovery ([`ae94ee3`](../../commit/ae94ee3)).
- Colored CLI output across every human-readable surface with `--no-colors` / `NO_COLOR` / non-TTY off-conditions; padded TTY tables for report commands; `--json` byte-stable regardless ([`96fed70`](../../commit/96fed70)).
- TUI `Ctrl+R` restarts the daemon preserving the parent dispatcher's resolved options; `Ctrl+Q` kills it; both stay discoverable via `?` only ([`adfef21`](../../commit/adfef21), [`0b6fc77`](../../commit/0b6fc77)).
- `LLAMASTASH_STATE_DIR` / `LLAMASTASH_CONFIG_DIR` / `LLAMASTASH_CACHE_DIR` env overrides for side-by-side daemons (alongside the existing `LLAMASTASH_SOCKET`).

## How to read this file

Tagged releases land under their version heading; in-flight work accumulates under **Unreleased** until the next tag promotes it. llamastash is pre-1.0 / WIP; the entire pre-release history is bundled under the first publishable tag, [0.0.1], rather than backfilled into a series of synthetic tags. The ledger starts there.
