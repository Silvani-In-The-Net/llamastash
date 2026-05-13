---
date: 2026-05-13
topic: llamatui-v1
---

# llamatui — v1 Requirements

## Problem Frame

Developers and enthusiasts running local LLMs today face a forced choice between heavy abstractions (Ollama, LM Studio) that hide llama.cpp behind their own model registries and conventions, and raw CLI use of `llama-server` that requires remembering long flag combinations and managing processes manually. There is no fast, keyboard-driven, *transparent* launcher that treats llama.cpp as a first-class backend and is usable both by a human at a terminal and by autonomous agents that drive it through shell tools.

llamatui v1 fills that gap: a Rust TUI that discovers GGUF models already on disk (including those in existing Ollama/LM Studio/HuggingFace caches), launches them via `llama-server` with sensible per-model defaults, surfaces health/status/resources, and exposes the same capabilities to external agents and scripts via a clean non-interactive CLI.

Audience:
- Primary: developers, ML practitioners, and power-user enthusiasts comfortable at the terminal who want full control over llama.cpp.
- Secondary: agent runtimes (Claude Code, Codex, custom scripts) that drive llamatui through CLI subcommands invoked from skills/tools.

## Requirements

### Discovery & Scanning

- **R1.** Recursively scan one or more roots for `*.gguf` files and group results by parent directory in the model list.
- **R2.** Auto-include well-known local model caches in the default scan set, using each tool's configured model directory: HuggingFace hub (`~/.cache/huggingface/hub`), Ollama (`~/.ollama/models`), and LM Studio (its configured model directory, which varies by version and OS; resolve via the user's LM Studio config when present, otherwise probe the conventional locations). The user can disable any of these individually.
- **R3.** Custom scan paths come from a config file, the env var `LLAMATUI_MODEL_PATHS` (OS path-separator list), and the repeatable flag `--model-path <dir>`.
- **R4.** A single env var / flag (`LLAMATUI_NO_SCAN=1` / `--no-scan`) disables filesystem scanning entirely. Custom paths from flags/env always merge with the default scan set unless `--no-scan` is set, in which case only user-provided paths are used.
- **R5.** Detect split GGUFs (e.g., `*-00001-of-00003.gguf`) and group siblings as a single logical model; launches target the first shard.
- **R6.** Detect symlinks and Ollama's content-addressed blob layout (`manifests/` + `blobs/`) and present models by their human-readable name rather than blob hash where possible.
- **R7.** Parse each GGUF file header for metadata: architecture, parameter count, quantization, native max context, embedded chat template name, embedded tokenizer hints, and any reasoning-format hint. Surface these in the model list.
- **R8.** Estimate model RAM/VRAM requirement from GGUF metadata, the active quantization, *and the chosen context length* (KV-cache cost grows with context × layers × model-dim and is a substantial part of total memory for long contexts), and warn (non-blocking) before launch when the estimate exceeds available memory.
- **R9.** Scanning is asynchronous and non-blocking; the TUI is usable while scanning is in progress, and discovered models stream into the list incrementally.
- **R46.** HuggingFace pull: a hotkey (e.g., `Ctrl+D`) opens an input for a HuggingFace repo ID (`owner/repo` or `owner/repo:filename.gguf`). On submit, llamatui downloads the GGUF into a configured target directory, surfaces download progress in the side panel, and once complete the new file appears in the scanned list. Failures are visible and recoverable.

### Launching & Lifecycle

- **R10.** Locate the `llama-server` binary in this priority order: `--llama-server <path>` flag → `LLAMATUI_LLAMA_SERVER` env var → `$PATH` lookup. If none resolve, fail closed with a clear, actionable error message that names both the flag and the env var.
- **R11.** Launch picker offers: context length (preset list + "GGUF native max" + "Custom…"), reasoning toggle (default ON for models whose GGUF metadata identifies as reasoning models, OFF otherwise), and access to an Advanced panel for any other `llama-server` flag. The Advanced panel also exposes an explicit port override; otherwise the port is auto-allocated (R15).
- **R12.** Default context-length presets: 2048, 4096, 8192, 16384, 32768, 65536, 131072, "GGUF native max", and "Custom…" (which opens a numeric input). Values above the GGUF native max are flagged as risky but allowed.
- **R13.** Reasoning toggle, when ON, sets `--reasoning-format deepseek --jinja` on the server *and* collapses `<think>` blocks in the smoke-test chat. The Advanced panel lets the user unbundle these so each side can be overridden independently.
- **R14.** Advanced panel exposes the full surface of `llama-server` flags, with discoverable autocompletion / suggestions for commonly used ones (`-ngl`, `--n-cpu-moe`, `--cache-type-k/v`, `--threads`, `--flash-attn`, `--mlock`, `--no-mmap`, `--parallel`, etc.). Free-form flag entry is always available as the escape hatch.
- **R15.** Each launched model gets a unique port. Ports are allocated by probing the configured range for the next free port; collisions outside that range are detected so the same model never gets a port that is already in use.
- **R16.** A model is shown as `loading` until the daemon successfully probes its `/v1/models` (or `/health`) endpoint, then `ready`. If the process exits, crashes, or fails to become healthy within a configurable timeout, the state becomes `error` and the failure cause is visible in the side panel.
- **R17.** Multiple models can be running concurrently. The status panel shows port, uptime, current state, and live RAM/VRAM/CPU usage per running model.
- **R18.** Stop is initiated with a single keystroke on a running model; the daemon issues SIGTERM with a 5-second grace period, then SIGKILL. A "stop all" hotkey exists and prompts for confirmation.
- **R19.** Embedding models (started with `--embeddings`) and reranker models (started with `--reranking`) are recognised at launch time. Their status row and smoke-test panel are tailored to their mode (embed text → vector display; rerank pair → score) rather than chat.
- **R20.** llamatui remembers the last successful launch parameters per unique model file (identified by canonical path + a stable hash of the GGUF *header* — never the full file). On the next launch, those parameters are pre-populated.
- **R21.** Users can save multiple **named presets** per model (e.g., `coding`, `long-ctx`, `fast`) and choose one at launch. Presets are editable from the TUI and stored in the persisted state directory.
- **R22.** Filesystem changes (new GGUFs appearing in scanned roots) are picked up automatically without a restart, via debounced filesystem watching.

### UI & Interaction

- **R23.** Layout: a model-list pane (primary, left), a right pane that holds the side panel content, and a contextual help bar that always shows the keybindings valid in the current focus.
- **R24.** Favorites: any model can be marked/unmarked favorite with a single key; favorites render at the top of the list, above the directory groupings.
- **R25.** Fuzzy/substring filter over the model list, activated with `/`, filtering on filename, directory, architecture, quantization, and any user-applied labels.
- **R26.** Built-in themes: Catppuccin Macchiato (default), Catppuccin Latte, Gruvbox Dark, Solarized Dark, and Monochrome. Theme selectable from config and from a runtime hotkey.
- **R27.** All actions reachable from the keyboard. Mouse support is optional polish, not required for any flow.
- **R28.** Status indicators use both colour *and* a glyph/shape, so the UI is legible on monochrome terminals and for users with colour-vision deficiencies.
- **R29.** Perceived warm-attach (daemon already running) to first interactive frame is under 200 ms on a typical developer machine; keyboard-input-to-redraw latency stays under 16 ms (60 fps target). Cold first-launch (daemon must spawn) is allowed up to ~1 second; scanning happens after first paint either way.
- **R30.** Clipboard actions: yank the running model's endpoint URL, yank a ready-to-paste `curl` command, and yank the canonical model path.

### Smoke-Test Side Panel

- **R31.** The right pane is tab-driven. Tabs include **Logs** (default) and, when a model with focus is `ready`, **Chat** (or **Embed**/**Rerank** depending on the model's mode — see R33). A hotkey cycles tabs; tab state is per-focused-model.
- **R32.** Smoke-test chat is explicitly a smoke test, not a daily-driver chat: single-shot prompts, streaming token output, no conversational history, no markdown rendering required, no system-prompt management beyond a single inline field. It uses the same OpenAI-compatible endpoint as external clients, so a successful smoke test proves the model is also usable from any external client.
- **R33.** For embedding models, the Chat tab is replaced by an **Embed** tab (single text → vector summary, optional second-text cosine similarity). For reranker models, by a **Rerank** tab (query + candidate list → ranked scores).

### Agent-Driveable CLI

- **R34.** *(Deferred to v2)* HTTP API and MCP server. v1 does not expose llamatui over HTTP or MCP. Agent integration in v1 is via the CLI subcommands below, which agents invoke from their own tool/skill layer.
- **R35.** Non-interactive CLI subcommands drive the daemon end-to-end:
  - `llamatui list [--json] [--filter PATTERN]`
  - `llamatui start <model-ref> [--preset NAME] [--ctx N] [--port N] [--reasoning on|off] [--mode chat|embedding|rerank] [-- ...extra llama-server flags]`
  - `llamatui stop <model-id-or-port>` and `llamatui stop --all`
  - `llamatui status [--json]`
  - `llamatui logs <model-id> [--follow] [-n N]`
  - `llamatui presets <model-ref> {list|save|delete} ...`
  - `llamatui pull <hf-repo-id-or-url>` (HuggingFace pull from CLI; same target dir as R46)
  - `llamatui daemon {start|stop|status}` (manual daemon control; the TUI auto-spawns the daemon on attach but users can run `daemon start` for headless setups)
  - All read subcommands support `--json` for structured output; failures use distinct exit codes.

### Persistence

- **R37.** Favorites, presets, last-launch parameters per model, filter state, theme, and the running-model snapshot are persisted to disk between sessions.
- **R38.** State, config, and logs follow XDG Base Directory conventions on Linux and the analogous macOS conventions, and are stored separately from each other (state is mutable runtime data; config is user-authored; logs are append-only).

### Architecture (key product decision)

- **R39.** The system is split into a **daemon** (owns `llama-server` child processes, the persisted state, the file watcher, and the IPC endpoint) and a **TUI / CLI client** (thin frontends that attach to the daemon over a local Unix-domain socket). The TUI and the non-interactive CLI subcommands speak the same internal protocol to the daemon.
- **R40.** The TUI auto-spawns the daemon on first attach if it is not running, and shuts it down only on explicit user request (TUI hotkey or `llamatui daemon stop`) — *not* when the TUI exits. Running models survive TUI close.
- **R41.** Only one daemon instance per user is permitted; subsequent daemon starts detect the existing instance via a PID/lock file and exit cleanly. Multiple TUI/CLI clients can attach to the same daemon concurrently.
- **R42.** Orphan handling on daemon restart: the daemon reads its persisted running-model snapshot and re-adopts only processes it spawned itself (matched by recorded PID, port, and canonical model path). Other `llama-server` processes on the system, if discovered, are surfaced **read-only** in a separate "external" section showing their port and PID — they can be stopped from llamatui but their launch parameters are not editable.
- **R47.** IPC authentication: the daemon listens on a Unix socket created with mode `0600` and verifies connecting peers via `SO_PEERCRED` (Linux) / `getpeereid` (macOS) so only the owning user's processes can drive it. No token files needed for v1; the HTTP/MCP token scheme is deferred to v2 with R34.

### Distribution & Cross-Platform

- **R43.** First-class platforms: Linux (x86_64 + aarch64) and macOS (Apple Silicon + Intel). Windows is explicitly post-v1.
- **R44.** GPU detection on Linux probes NVIDIA (NVML), AMD (ROCm-SMI), and Vulkan-capable devices; on macOS it detects Apple Silicon Metal. On Intel macOS, GPU detection is a best-effort no-op (CPU-only assumed). Detected GPU memory feeds the launch-time RAM/VRAM warning (R8).
- **R45.** Distribution: a **single binary** `llamatui` shipped via `cargo install`, pre-built release binaries on GitHub Releases (per platform), and Homebrew. All daemon, TUI, and CLI behavior is accessed through subcommands of that one binary. Linux distro packaging (AUR, deb, rpm) is desirable but not required for v1.

## Success Criteria

- A new user on a fresh machine that already has GGUFs in any of the well-known caches can launch the TUI, see their models grouped and annotated, pick one, hit `Enter`, and have a healthy OpenAI-compatible endpoint within ~10 seconds of model load time — *without reading docs*.
- An external agent (Claude Code, Codex, or a shell script) can `llamatui list --json` → `llamatui start <model> --preset coding` → query the resulting endpoint → `llamatui stop` with no human in the loop, in a stable, documented way.
- Warm-attach cold-start to first interactive frame is under 200 ms; cold first-launch (daemon spawn) under ~1 s. The UI remains responsive (<16 ms input-to-redraw) while a large-tree scan is in progress.
- Killing the TUI does not interrupt running models; reopening the TUI re-attaches and shows correct status. Killing the daemon and restarting it cleanly re-adopts its own processes and surfaces external `llama-server` processes read-only.
- A reasonable subset of users prefer llamatui to manually invoking `llama-server` and to Ollama when they want llama.cpp behaviour directly. (Measured informally via GitHub stars, issue volume, and qualitative feedback for the first 3 months post-release.)

## Scope Boundaries

**In v1, deliberately out of scope:**
- HTTP API and MCP server. Both are tracked for v2 (see R34). Agent integration in v1 happens via the CLI surface (R35).
- Daily-driver chat experience (markdown rendering, conversation history, multi-turn UI). Users continue to use Open WebUI, IDE plugins, etc. for real chat.
- Model downloading other than HuggingFace (R46). No Ollama-registry pull, no civitai-style sources.
- Quantizing or converting models. llamatui consumes existing GGUFs; it does not produce them.
- Backends other than `llama-server`. No vLLM, no MLX, no Ollama serve passthrough. The product is *transparent about llama.cpp*; making it generic dilutes that.
- Multi-user / remote / network-exposed daemon. v1 is single-user, local-socket only.
- Windows. Tracked for a later milestone; not gating v1.
- Telemetry of any kind.
- Auto-update of llamatui itself.
- Notifications outside the TUI (system notifications, webhooks).

**Explicit non-feature:**
- llamatui does **not** maintain its own model registry or manifest format. Models are files on disk; that is the source of truth.

## Key Decisions

- **Identity: launcher + smoke-test side panel + agent-driveable CLI.** Full chat is rejected to avoid scope creep and keep llamatui's value as a transparent llama.cpp launcher sharp. HTTP API and MCP server are deferred to v2 — v1's agent story is a clean CLI that agents call from skills/tools.
- **Daemon-on-demand architecture.** Models and the protocol must be queryable when no TUI is open (so the CLI can drive things). TUI auto-spawns daemon on attach; daemon survives TUI exit.
- **Single binary with subcommands.** One `llamatui` artifact handles TUI, CLI, and daemon. Simpler install, simpler packaging.
- **Unix-socket peercred auth.** Single-user local tool. No token files in v1.
- **GGUF-native intelligence is part of v1.** Parsing GGUF metadata for arch, context, quant, chat template, and RAM/VRAM-with-KV-cache estimates is a differentiator no other TUI offers and a small implementation cost.
- **HF pull and cache auto-discovery are part of v1.** Closing the "discover → run" loop on day one is the biggest popularity driver for the workbench identity.
- **Reasoning toggle is opinionated by default.** A single combined switch handles 95% of use; the Advanced panel lets power users unbundle it.
- **Orphan policy: own-only adoption, external read-only.** The daemon never silently takes ownership of `llama-server` processes it didn't start, but it does *show* them so the user always sees the full picture.
- **Right pane is tab-driven** (Logs / Chat / Embed / Rerank), per-focused-model. Avoids modals and a third pane.
- **MIT licensed**, matching the author's other TUIs (kdash, jwt-ui), to maximise reach.

## Dependencies / Assumptions

- `llama-server` from llama.cpp ≥ a recent release (must support `--jinja`, `--reasoning-format`, `--embeddings`, `--reranking`, `/v1/models`, `/health`). v1 documents a tested minimum version.
- GGUF format is stable enough for header parsing; older/odd files are handled gracefully or skipped with a clear message.
- The user's OS provides Unix-domain sockets and `SO_PEERCRED`/`getpeereid` (true on Linux + macOS, which are the v1 targets).
- Reference UX patterns are taken from `kdash` and `jwt-ui` (both authored by the same maintainer): collapsible side panel, contextual help bar, Macchiato palette, themable.

## Outstanding Questions

### Resolve Before Planning

*(none)*

### Deferred to Planning

- [Affects R7][Needs research] Which GGUF-parsing crate to adopt vs. hand-rolling a minimal header parser — driven by which fields we actually surface and how much of the format we care about (just header, or also tensor metadata for tighter RAM/VRAM estimates).
- [Affects R8, R17, R44][Needs research] Exact mechanism for GPU memory detection on each platform (NVML wrapper crate, ROCm SMI shell out, `ioreg` parse on macOS) and how to keep the dependency footprint small.
- [Affects R10, R45][Needs research] When multiple `llama-server` binaries exist in `$PATH` (e.g., a CUDA build and a Vulkan build), how should llamatui disambiguate? Pick the first, prompt the user once and cache, or expose via config? Should llamatui *help install* `llama-server` when it's missing, rather than only erroring?
- [Affects R15][Needs research] Default auto-allocation port range. 8080–8200 is conventional but heavily contested with dev servers; a higher unprivileged range (e.g., 41100–41300) collides less. Settle during planning after a quick survey.
- [Affects R22, R46][Technical] Filesystem watcher strategy for very large model trees (HuggingFace hub layout is deeply nested) and how the watcher interacts with HF downloads-in-progress so partial GGUFs don't surface as launchable models.
- [Affects R29][Technical] Async runtime topology between scanning, IPC, the file watcher, and the UI render loop — a planning concern, not a brainstorm concern.
- [Affects R34][Strategic] v2 surface (HTTP + MCP) should be designed so the internal IPC protocol can host both without a rewrite. Sketch the v2 shape during v1 planning enough to not paint ourselves into a corner.
- [Affects R26][Scope] Five themes is generous for a v1 launch. Planning may choose to ship 2–3 (Macchiato default + Latte + Mono) and add the others post-v1 if it shortens the path to release.

## Next Steps

`-> /ce:plan` for structured implementation planning.
