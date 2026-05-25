# Suite B — cross-tool run JSONs

One subdirectory per host (e.g. `runs/dev-box/`, `runs/m3-max/`). Inside each subdirectory, files are named `<YYYY-MM-DD>-<commit-sha>.json` and validate against the v1 schema in `scripts/bench/end_to_end/schema.py`.

These are the raw inputs the renderer reads to produce `docs/benchmarks/results-<DATE>.md`. They're checked into the repo so:

- Every published chart is regenerable from source.
- Community contributors can drop a new host directory and get rendered into the next results page.
- Future drift (driver bump, llama.cpp commit change) is auditable against the prior run on the same host.

See [../methodology.md](../methodology.md) for the schema's semantics and the variance-gate rules the renderer applies.
