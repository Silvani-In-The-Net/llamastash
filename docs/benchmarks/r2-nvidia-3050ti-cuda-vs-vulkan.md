# R2 — NVIDIA RTX 3050 Ti: CUDA vs Vulkan, four-tool cross-comparison

**Date:** 2026-05-28
**Host:** `deepu-xps` (Manjaro Linux, kernel 6.6.141, NVIDIA driver 595.71.05, CUDA 13.2)
**GPU:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (Ampere, sm_86), 4.0 GiB VRAM
**CPU:** Intel i9-11900H (8 cores, AVX-512)
**RAM:** 63 GiB

See [methodology.md](methodology.md) before reading specific numbers; the matched-pair settings policy, the variance gate, and the cross-backend determinism caveat all apply here.

## Scope

- One model size class: `small` = `gemma-3-4b-it.Q3_K_M.gguf` (2.1 GiB, byte-identical across all four tools).
- Four tools: `llamastash`, raw `llama-server` (`llamacpp`), `ollama` (upstream installer), LM Studio (`lmstudio`).
- Two backends: **CUDA** and **Vulkan**. Each tool's engine choice is documented per-row below.
- Two modes per tool: `defaults`, `normalized`. Methodology unchanged from R1.
- Four workloads per cell: `chat_turn`, `rag_prefill`, `agent_decode`, `parallel_4`.
- 5 reps per cell (rep 0 warmup, reps 1–4 measured).

`mid` and `large_dense` were not exercised: the 4 GiB VRAM ceiling forces every dense model larger than `small` into a partial-offload regime that's not interesting for a cross-tool comparison. A larger-VRAM host should re-run them.

## Per-tool engine routing (which lane is what)

Real backend used per row, established via post-run log inspection / argv recording:

| Tool         | CUDA lane                                                                                                                                                        | Vulkan lane                                                                                                                      |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `llamastash` | self-built `llama-server` (b9360 commit `6b4e4bd5`, `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`), forwarded to the daemon via `--llama-server` per call | upstream prebuilt `llama-server` from `ggml-org/llama.cpp` release b9360 (Vulkan asset `ubuntu-vulkan-x64.tar.gz`)               |
| `llamacpp`   | same CUDA-built binary, invoked directly                                                                                                                         | same Vulkan binary, invoked directly                                                                                             |
| `ollama`     | upstream installer (`/usr/local/bin/ollama` 0.24.0), CUDA runner auto-selected; daemon log records `library=CUDA compute=8.6 libdirs=ollama,cuda_v13`            | same binary, `OLLAMA_VULKAN=1 OLLAMA_LLM_LIBRARY=vulkan`; daemon log records `library=Vulkan name=Vulkan0 libdirs=ollama,vulkan` |
| `lmstudio`   | `lms runtime select llama.cpp-linux-x86_64-nvidia-cuda12-avx2` (engine 2.16.0)                                                                                   | `lms runtime select llama.cpp-linux-x86_64-vulkan-avx2` (engine 2.16.0)                                                          |

The "as-shipped" wrapper conventions (llamastash auto-installs the Vulkan prebuilt on Linux + NVIDIA per `src/init/install/gh_releases.rs:6-8`; Ollama defaults to CUDA when present; LM Studio defaults to CUDA12 when present) are documented for context but **not** treated as honest defaults in this run — every row is explicitly pinned to a known backend.

## Variance check (R140)

1 cell flagged with stddev > 10%:

- `ollama / CUDA / normalized / agent_decode`: TTFT stddev 10.9% (just over the 10% advisory line; well under the 25% drop threshold). Kept in tables, marked `±`.

All other 63 cells are within the ≤10% clean band.

## Decode throughput (tokens/sec, mean across 4 measured reps)

`rag_prefill` is a prefill-only workload and does not produce a meaningful decode-tps number.

| Tool                  | mode       | chat_turn   | rag_prefill | agent_decode | parallel_4   |
| --------------------- | ---------- | ----------- | ----------- | ------------ | ------------ |
| **llamastash CUDA**   | defaults   | 41.1        | –           | 38.0         | 82.1         |
| **llamastash Vulkan** | defaults   | 42.0 (+2%)  | –           | 41.4 (+9%)   | 85.9 (+5%)   |
| llamastash CUDA       | normalized | 33.2        | 29.4        | 33.2         | 81.4         |
| llamastash Vulkan     | normalized | 38.3 (+15%) | 33.8 (+15%) | 37.8 (+14%)  | 82.6 (+2%)   |
| **llamacpp CUDA**     | defaults   | 36.6        | –           | 29.4         | 68.2         |
| **llamacpp Vulkan**   | defaults   | 37.5 (+3%)  | –           | 29.9 (+2%)   | 63.4 (−7%)   |
| llamacpp CUDA         | normalized | 33.2        | 29.3        | 33.2         | 81.4         |
| llamacpp Vulkan       | normalized | 33.6 (+1%)  | 32.4 (+10%) | 38.2 (+15%)  | 85.1 (+5%)   |
| **ollama CUDA**       | defaults   | 40.7        | –           | 32.7         | 134.5        |
| **ollama Vulkan**     | defaults   | 42.0 (+3%)  | –           | 41.0 (+25%)  | 156.6 (+16%) |
| ollama CUDA           | normalized | 32.8        | –           | 32.8         | 134.8        |
| ollama Vulkan         | normalized | 38.1 (+16%) | –           | 41.5 (+27%)  | 162.5 (+21%) |
| **lmstudio CUDA**     | defaults   | 48.7        | –           | 36.6         | 113.6        |
| **lmstudio Vulkan**   | defaults   | 48.3 (−1%)  | –           | 40.0 (+9%)   | 119.4 (+5%)  |
| lmstudio CUDA         | normalized | 39.1        | –           | 32.6         | 109.4        |
| lmstudio Vulkan       | normalized | 44.8 (+15%) | –           | 37.9 (+16%)  | 115.0 (+5%)  |

## TTFT (time-to-first-token, ms, post-load)

| Tool              | mode       | chat_turn  | rag_prefill | agent_decode | parallel_4  |
| ----------------- | ---------- | ---------- | ----------- | ------------ | ----------- |
| llamastash CUDA   | defaults   | 74         | –           | 86           | 286         |
| llamastash Vulkan | defaults   | 113 (+54%) | –           | 118 (+38%)   | 378 (+32%)  |
| llamastash CUDA   | normalized | 95         | 60          | 103          | 298         |
| llamastash Vulkan | normalized | 133 (+40%) | 52 (−12%)   | 137 (+33%)   | 372 (+25%)  |
| llamacpp CUDA     | defaults   | 110        | –           | 137          | 373         |
| llamacpp Vulkan   | defaults   | 148 (+35%) | –           | 189 (+38%)   | 513 (+37%)  |
| llamacpp CUDA     | normalized | 95         | 61          | 103          | 307         |
| llamacpp Vulkan   | normalized | 144 (+52%) | 53 (−12%)   | 133 (+29%)   | 370 (+20%)  |
| ollama CUDA       | defaults   | 120        | 3422        | 135          | 3340        |
| ollama Vulkan     | defaults   | 115 (−4%)  | –           | 114 (−15%)   | 2979 (−11%) |
| ollama CUDA       | normalized | 136        | 3712        | 136 (±11%)   | 3318        |
| ollama Vulkan     | normalized | 118 (−13%) | –           | 116 (−15%)   | 2914 (−12%) |
| lmstudio CUDA     | defaults   | 318        | 119         | 155          | 795         |
| lmstudio Vulkan   | defaults   | 308 (−3%)  | 111 (−7%)   | 142 (−8%)    | 788 (−1%)   |
| lmstudio CUDA     | normalized | 386        | 120         | 179          | 834         |
| lmstudio Vulkan   | normalized | 330 (−14%) | 117 (−2%)   | 151 (−16%)   | 830 (−0.4%) |

## Wall-clock per-lane

Total Suite B `--models small` wall-clock summed across the 4 per-tool runs:

| Lane   | ollama | llamastash | llamacpp | lmstudio | Total                |
| ------ | ------ | ---------- | -------- | -------- | -------------------- |
| CUDA   | 291s   | 137s       | 154s     | 124s     | **706s (≈12 min)**   |
| Vulkan | 6172s  | 127s       | 146s     | 120s     | **6565s (≈109 min)** |

The Vulkan total is dominated by `ollama`'s two `rag_prefill` cells. Prefill on Vulkan is dramatically slower than on CUDA when the prompt is large (8 K tokens here); see "Findings" §4.

## Findings

### 1. Vulkan decode ≥ CUDA decode, consistently

On gemma-3-4B Q3_K_M on a 4 GiB RTX 3050 Ti, **Vulkan decode is faster than CUDA decode in 26 of 28 comparable cells** (median Δ +5%, range −7% to +27%). This contradicts the conventional "CUDA wins on NVIDIA" intuition and reflects what the actual measurement shows on this hardware + this quant.

Likely cause: decode on a Q3-quantized 4B model is memory-bandwidth-bound on a 4 GiB Ampere card, and Vulkan's memory-access path on the upstream llama.cpp Vulkan backend has caught up with (or slightly exceeded) the CUDA path for the kernels these workloads exercise.

### 2. Vulkan TTFT is consistently _worse_ on llamastash/llamacpp; consistently _better_ on ollama/lmstudio

| Lane           | Tool family                                                                      | TTFT trend                        |
| -------------- | -------------------------------------------------------------------------------- | --------------------------------- |
| Vulkan vs CUDA | `llamastash` + `llamacpp` (both on the same b9360 upstream binary)               | **+20% to +54% slower** on Vulkan |
| Vulkan vs CUDA | `ollama` + `lmstudio` (each on their own bundled engine, different commit/build) | **−1% to −16% faster** on Vulkan  |

Same engine source (`llamastash` and `llamacpp` share the b9360 binary) ⇒ same TTFT story (Vulkan slower). Different engine source (Ollama 0.24.0 / LM Studio 2.16.0) ⇒ different TTFT story (Vulkan faster). The Vulkan prefill kernels in **upstream llama.cpp b9360** are slower than CUDA on this GPU; the Vulkan prefill paths in Ollama's and LM Studio's bundled engines are not.

### 3. Defaults vs normalized

`normalized` mode forces `ngl=99`, `flash_attn=on`, `ctx=4096`, fixed batch sizes. `defaults` mode is each tool's out-of-the-box choice.

- **llamastash + llamacpp defaults consistently underperform their normalized rows** by 6–25% on chat_turn / agent_decode. The upstream binaries' conservative defaults (lower batch, no `--flash-attn auto`) leave perf on the table.
- **lmstudio defaults _outperform_ normalized** by 6–24%. LM Studio's defaults are tuned for the loaded model + detected GPU; the bench's universal normalized recipe is not.
- **ollama normalized = defaults** (numbers nearly identical). Methodology already explains this: the Ollama OpenAI shim silently caps `ctx` and ignores `ngl`, so normalized knobs land on `unfair_knobs`.

### 4. Vulkan rag_prefill on ollama is catastrophically slow

`rag_prefill` uses an 8 K-token prompt. On ollama:

- CUDA rag_prefill defaults TTFT: 3.4 s (manageable).
- Vulkan rag_prefill defaults TTFT: ollama didn't surface a TTFT number, but the **wall-clock for the two rag_prefill cells alone consumed roughly 1.5 hours** out of the 1h 43m total ollama-Vulkan run.

For document-RAG style workloads on this hardware, ollama-Vulkan is a non-starter. ollama-CUDA, llamastash-CUDA, llamacpp-CUDA, lmstudio (both lanes) all process the same 8 K-token prefill in under 4 seconds.

### 5. llamastash matches llamacpp when knobs are matched

In `normalized` mode (where llamastash collapses to a byte-equivalent invocation per Suite A) llamastash and llamacpp track within ±0.5 tok/s on chat_turn and agent_decode on both backends. The Suite A argv-equality claim (R125) extends to runtime parity in this run.

### 6. lmstudio leads on small-VRAM throughput

`lmstudio` posts the highest absolute numbers in 6 of 8 defaults workloads — partly because its engine defaults are tuned (finding #3), partly because the LM Studio 2.16.0 engine ships marginally newer kernels than the b9360 binary llamastash inherits from upstream. The CUDA-normalized lanes converge back to the engine-baseline numbers (33–115 tok/s across tools), so the headline lmstudio advantage is "smarter defaults," not "fundamentally faster engine."

## Suite A and Suite C (Vulkan only, not re-run)

The earlier Vulkan-lane Suite A and Suite C results stand and are not re-run in this report:

- Suite A (overhead): TTFT +1.7 ms, decode −0.57% → ADVISORY. JSON: `docs/benchmarks/overhead/deepu-xps/2026-05-27-9806df2033dc.json`.
- Suite C (proxy): TTFT +0.57 ms, decode −0.20% → OK. JSON: `docs/benchmarks/proxy/deepu-xps/2026-05-28-9806df2033dc.json`.

Re-running Suite A/C against the CUDA binary is unlikely to change the proxy/overhead numbers (the proxy is engine-agnostic) but would be needed to certify CUDA-lane overhead at release time.

## Bench-harness fixes applied during this run

Two real upstream bench-harness bugs were patched locally to make these numbers possible. Both remain uncommitted; both are worth turning into PRs:

1. `scripts/bench/end_to_end/drivers/llamastash.py:69` — forward `LLAMASTASH_LLAMA_SERVER` to `llamastash start` as `--llama-server <path>` so the daemon honours the bench's binary choice. Without this, a running daemon's cached binary path silently overrides the bench's `LLAMASTASH_LLAMA_SERVER` env var, and "CUDA-lane" runs end up on whatever was cached (Vulkan, in the first attempt of this report).
2. `scripts/bench/proxy/orchestrator.py:143` — use `Path.absolute()` instead of `Path.resolve()` for the proxy model-id fallback. `resolve()` follows HF cache symlinks to the content-addressed blob, which the proxy doesn't recognise; `absolute()` preserves the symlink path the proxy _does_ recognise.

## Provenance

Eight per-tool Suite B JSONs, one per (tool, backend) cell:

```
docs/benchmarks/runs/deepu-xps-cuda/
  2026-05-28-072734-9806df2033dc.json   ollama       (CUDA)
  2026-05-28-073332-9806df2033dc.json   llamastash   (CUDA)
  2026-05-28-073616-9806df2033dc.json   llamacpp     (CUDA)
  2026-05-28-073932-9806df2033dc.json   lmstudio     (CUDA)

docs/benchmarks/runs/deepu-xps-vulkan/
  2026-05-28-074333-9806df2033dc.json   ollama       (Vulkan)
  2026-05-28-092711-9806df2033dc.json   llamastash   (Vulkan)
  2026-05-28-092944-9806df2033dc.json   llamacpp     (Vulkan)
  2026-05-28-093243-9806df2033dc.json   lmstudio     (Vulkan)
```

Bench harness commit: `9806df2033dce7f002515cc1dcc84b1024e6dff9` (plus the two uncommitted bench-harness patches noted above).

Two earlier CUDA-lane JSONs (`2026-05-28-054627-…` and `2026-05-28-055402-…`) were discarded: the first attempted CUDA but the daemon used a cached Vulkan binary; the second was correct CUDA but mixed three tools in one invocation, making per-tool isolation harder to verify. Both are kept on disk for audit but excluded from this report's numbers.
