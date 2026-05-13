# llamatui

A fast, keyboard-driven TUI for launching and managing local `llama-server` (llama.cpp) instances.

> **Status: early development.** The v1 scope is in [`docs/brainstorms/llamatui-requirements.md`](docs/brainstorms/llamatui-requirements.md). The implementation plan is in [`docs/plans/2026-05-13-001-feat-llamatui-v1-launcher-plan.md`](docs/plans/2026-05-13-001-feat-llamatui-v1-launcher-plan.md).

## What it does (v1, in progress)

- Discovers GGUF models on disk — including HuggingFace, Ollama, and LM Studio caches — and groups them by directory.
- Surfaces GGUF metadata (architecture, quantization, native context, KV-cache-aware memory estimates) so you can pick smart defaults.
- Launches `llama-server` with a tweakable launch picker (context length, reasoning, advanced flags), per-model named presets, favorites, and a filter.
- Manages multiple concurrent models with a health-probed status state machine; logs and a smoke-test prompt panel are one tab away.
- Pulls GGUFs directly from HuggingFace from inside the TUI or from the CLI.
- Exposes the same primitives as non-interactive `llamatui` subcommands so shell scripts and AI agents can drive it.

## Install

Coming soon: `cargo install llamatui`, Homebrew tap, and pre-built release binaries.

## License

MIT © Deepu K Sasidharan
