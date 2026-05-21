# Suite A — `llamastash` vs raw `llama-server` overhead JSONs

One subdirectory per host. Files are named `<YYYY-MM-DD>-<commit-sha>.json` and validate against the v1 schema (`scripts/bench/end_to_end/schema.py`) with `suite: "overhead"`.

Suite A is the architectural regression check: every release runs it on the maintainer's primary hardware and compares the LlamaStash-vs-raw deltas against the two-tier thresholds documented in [../methodology.md#suite-a-two-tier-threshold-r123](../methodology.md#suite-a-two-tier-threshold-r123).

- **Catastrophic** — orchestrator exits non-zero. Block the release until investigated.
- **Advisory** — exits zero with a banner. Worth a look, not a blocker.
- **OK** — exits silently.

See `scripts/bench/overhead/thresholds.json` for the current numeric bounds (with per-backend overrides allowed).
