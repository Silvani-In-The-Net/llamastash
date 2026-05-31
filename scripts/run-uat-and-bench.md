# Steps to run UAT and benchmarks

1. Install Rust if you don't have it: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`. (You also need `git`, `make`, and a C/C++ compiler — standard build tools on mac and Linux.)
2. Get the repo and run it:

   ```bash
   git clone https://github.com/llamastash/llamastash.git && cd llamastash
   ./scripts/run-uat-and-bench.sh
   ```

3. For the full cross-tool comparison (you chose this), before running also:
   - Install Ollama and make sure it's running (`ollama list` works).
   - Install LM Studio, run `~/.lmstudio/bin/lms bootstrap` so the `lms` CLI is on PATH, and inside the app download `Qwen2.5-0.5B-Instruct-GGUF (Q4_K_M)` so it's in the library.
   - If you skip these, the script still runs — it just leaves those two tools out.
4. Send back the `handoff-results-*.tar.gz` it prints at the end.
