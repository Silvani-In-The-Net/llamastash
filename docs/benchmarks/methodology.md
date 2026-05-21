# Methodology

This document is the *contract* the benchmark pages refer back to. Two suites share this contract:

- **Suite A — overhead regression.** Does `llamastash start <model>` add measurable overhead on top of raw `llama-server` for the same effective argv? Architecturally, LlamaStash spawns the unmodified upstream binary; this suite proves that claim quantitatively per release.
- **Suite B — cross-tool comparison.** Same model, same hardware, four tools (LlamaStash, raw `llama-server`, Ollama, LM Studio) driven through their OpenAI-compatible HTTP endpoints. Answers "how does LlamaStash-as-shipped compare to the alternatives?"

Both suites share the harness under `scripts/bench/`. Output JSONs live under `docs/benchmarks/runs/<host-id>/` (Suite B) and `docs/benchmarks/overhead/<host-id>/` (Suite A); the renderer reads them to build dated results pages.

## What we measure

Per cell (one tool × one model × one mode × one workload), the renderer reports the mean across measured reps (the first rep is a warmup and is excluded):

| Metric | What it captures |
|--------|------------------|
| `ttft_ms` | Time from request-send to the first SSE chunk |
| `ttft_ms_first_request` | TTFT including any lazy-load on Ollama / LM Studio (cold) |
| `ttft_ms_post_load` | TTFT after the model is warm — engine-comparable across tools |
| `prompt_tps` | Prompt-evaluation tokens per second |
| `decode_tps` | Generation tokens per second |
| `e2e_latency_ms` | Total request wall-clock |
| `rss_peak_mb` | Process RSS peak captured during the rep |
| `gpu_mem_peak_mb` | Backend-appropriate GPU memory peak |

For each metric the cell also records the stddev as a percentage of the mean. The renderer's variance gate flags cells with stddev > 10% (rendered with `±` and the percentage inline; excluded from the headline chart but kept in the detail table) and drops cells with stddev > 25% (replaced with a "re-run needed" placeholder in a footer section).

## Matched-pair settings policy (R130)

Every model runs in two modes per tool:

- **Defaults mode** — invoke each tool exactly as a new user would. No tuning knobs supplied. This is the comparison most prospective users actually care about.
- **Normalized mode** — supply the same effective knobs across every tool to the extent the tool's CLI / API allows: `ctx`, `n_gpu_layers`, `flash_attn`, `kv_cache_type`, `batch_size`, `ubatch_size`. Where a tool refuses to expose a knob, the cell records it under `unfair_knobs` and the renderer prints the gap in the published table.

The `rag_prefill` workload overrides `ctx` to `8192` regardless of mode so the 8k-token prompt fits.

## Per-tool fairness notes

The methodology page is updated post-Unit-8 with the actual list of knobs each tool refuses to expose. Until the first end-to-end run lands, the placeholders below describe the *intent*, not yet the *observed*.

- **`llamastash` / raw `llama-server`** — full control. Normalized mode for LlamaStash uses `LLAMASTASH_BENCH_DISABLE_DEFAULTS=1` so the resolver collapses to "user knobs only," producing argv byte-identical to the raw `llama-server` invocation. The Suite A overhead check asserts this byte-equality (after stripping `--port`).
- **Ollama** — driven through `/v1/chat/completions`. Each test GGUF is imported once via `ollama create <bench-name> -f <Modelfile>`; the harness verifies the content-addressed blob's SHA matches the source. The Ollama driver runs `ollama rm <bench-name>` in `stop()` to bound the imported-blob store growth. *Q2 — Modelfile `PARAMETER` vs OpenAI shim parameter precedence — is resolved post-first-run.*
- **LM Studio** — driven through `lms load` + `lms server start` + `/v1/chat/completions`. Normalized mode passes `--context-length`, `--gpu`, and any other knob `lms` exposes; un-exposable knobs land in `unfair_knobs`. *Q1 — the exact `lms` normalization ceiling — is resolved post-first-run.* The MLX, vLLM, mlc-llm, and exllamav2 engines LM Studio can also drive are explicit non-goals (R147); normalized mode forces the llama.cpp path.

## Variance gate (R140)

- Per-cell stddev computed across measured reps (warmup excluded).
- `stddev / mean * 100`:
  - `≤ 10%` — clean; cell is published unconditionally.
  - `10% < x ≤ 25%` — flagged; rendered with `±<pct>%` inline, excluded from the headline chart but kept in the detail table.
  - `> 25%` — dropped; the cell becomes a footer note with "re-run needed."

Threshold defaults live in `scripts/bench/overhead/thresholds.json` and are tunable. The Suite A overhead orchestrator uses a separate two-tier threshold (`catastrophic` exits non-zero; `advisory` exits zero with a banner) keyed to the same JSON file.

## Cross-backend determinism caveat (R141 as edited)

Token-ID identity is asserted only *within* a backend (e.g., two CUDA-backed tools should produce the same token sequence on the same prompt and same sampling settings). Across backends, floating-point variance in CUDA / Metal / ROCm / CPU kernels is real and not a bug; the harness logs the divergence but never fails. The `Determinism` block in each cell records `prompt_sha256`, `first_n_token_ids_sha256`, and `determinism_mismatch: bool` so readers can see the check happened.

## Suite-A two-tier threshold (R123)

Suite A produces two scalar deltas per metric (LlamaStash mean − raw mean):

| Metric | Catastrophic (exit 1) | Advisory (exit 0 with banner) |
|--------|-----------------------|-------------------------------|
| `ttft_ms` delta | ≥ 200 ms | ≥ 30 ms |
| `decode_tps` delta percentage | ≥ 2.0% slower | ≥ 0.5% slower |
| Daemon idle RSS | ≥ 64 MiB extra | ≥ 48 MiB extra |

The thresholds are first-cut estimates; we tune them after the first cross-host runs land observed variance.

## Repeatability contract (R135)

Every published results page is reproducible from:

- the harness commit recorded in `git_sha` of the source JSON,
- the four tool versions recorded in `Provenance`,
- the four GGUF SHA-256s recorded per `ModelSpec`,
- the matched-pair settings declared in this doc.

Anyone with the same hardware class can re-run `scripts/bench/end_to_end/run.sh` and compare against the published JSON. Differences are evidence of a real change (driver bump, llama.cpp commit, OS scheduler, etc.), not measurement noise — the variance gate (above) bounds the noise.

## Re-running

Prerequisites:

- Linux or macOS. Windows is out of scope (mirrors LlamaStash's own platform coverage; R149).
- A llama.cpp `llama-server` binary on PATH (or `LLAMASTASH_LLAMA_SERVER`).
- For Suite B, additionally: the `ollama` and `lms` CLIs on PATH. The harness exits with a one-line install hint per tool when a binary is missing (R137); it does not auto-install.
- Disk budget: each test GGUF is ~4 GiB; Ollama imports duplicate the bytes into its content-addressed store. Plan for ~50 GiB per backend host with all four model sizes. `LLAMASTASH_BENCH_KEEP_IMPORTS=1` skips the per-cell Ollama cleanup for debugging.

Commands:

```sh
make bench-end-to-end -- --dry-run   # print the planned matrix
make bench-end-to-end                 # run Suite B
make bench-overhead                   # run Suite A
make bench-test                       # unit tests for the harness itself
```

Env vars honored by `run.sh`:

- `LLAMASTASH_BENCH_HOST_ID` — override the short hostname used as the runs/ subdir.
- `LLAMASTASH_BENCH_GPU_BACKEND` — force the backend autodetect; useful when a host has multiple.
- `LLAMASTASH_BENCH_PORT_BASE` — first free port to probe (default `18000`).
- `LLAMASTASH_BENCH_READY_TIMEOUT_S` — driver readiness timeout (default `180`).
- `LLAMASTASH_BENCH_KEEP_IMPORTS` — keep Ollama-imported models on `stop()`.
- `LLAMASTASH_BENCH_MODELS_{SMALL,MID,LARGE_DENSE,LARGE_MOE}` — per-slot model overrides (`<hf_repo>/<hf_file>`).

## Conflict-of-interest disclaimer

This is a first-party benchmark. LlamaStash maintainers picked the workloads, the matched-pair policy, and the rendering. Three guardrails exist to keep that honest:

1. Raw `llama-server` is in Suite B. If LlamaStash is doing anything more than passing knobs through to upstream, the gap shows up.
2. Suite A asserts argv byte-equality between LlamaStash and raw `llama-server` for the same explicit knobs — there's no place for a hidden tweak to hide.
3. Every JSON is checked into the repo; every chart is deterministic SVG (matplotlib SVG backend, no JS). Any reader can re-render the same chart from the same JSON, or re-run the harness on their hardware and compare.

Unflattering numbers ship truthfully (R150). If a future run finds LlamaStash slower or more memory-hungry than an alternative on a real workload, the page lands the same way the favorable ones do.

## Non-goals

Carrying forward from the brainstorm (R145–R150):

- Model quality (HumanEval / MMLU / Aider) — speed and resource cost only.
- GUI / UX comparison — LM Studio's GUI vs LlamaStash's TUI is a separate brainstorm.
- Native non-llama.cpp engines (MLX, vLLM, mlc-llm, exllamav2) — normalized mode forces LM Studio's llama.cpp path; MLX may become a separate "Apple Silicon engine comparison" later.
- Cloud / hosted endpoints — local-only.
- Windows.
- Tools beyond the four named — Jan, GPT4All, llamafile, KoboldCpp are follow-ups, not v1.

## Open questions tracked here

- **Q1** — LM Studio normalization ceiling. Resolved post-first-Suite-B-run.
- **Q2** — Ollama Modelfile vs OpenAI API parameter precedence. Resolved post-first-Suite-B-run.
- **Q3** — Run both `large_dense` and `large_moe`, or pick one per host? Decided after the first NVIDIA + Apple Metal runs.
- **Q4** — Per-tool `llama.cpp` commit recording. Best-effort capture in `scripts/bench/end_to_end/provenance.py`; `None` when not extractable.
- **Q6** — TTFT "cold launch" definition. Recorded as **both** `ttft_ms_first_request` (with lazy-load) and `ttft_ms_post_load` (engine-comparable). The renderer chooses which to chart per workload.
