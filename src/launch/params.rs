//! Compose `llama-server` argv from the user's launch choices.
//!
//! Order matters: `--host 127.0.0.1` and `--port` come first so the
//! command line reads well in logs; then `-m <path>`, then mode flags
//! (`--embeddings` / `--reranking`), then reasoning bundle
//! (`--jinja --reasoning-format deepseek`), then `-c <ctx>`, then any
//! user-supplied advanced flags. Advanced flags land *last* so they
//! always trump bundled ones — that's the contract documented on the
//! TUI's "Advanced" panel.

use std::ffi::OsString;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::launch::mode::LaunchMode;

/// All launch knobs the supervisor reads. Persisted under
/// `last_params: HashMap<ModelId, LaunchParams>` in `state.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchParams {
  /// Absolute path to the GGUF the user picked (or shard 1 for split
  /// sets).
  pub model_path: PathBuf,
  /// Chosen launch mode (chat / embedding / rerank).
  pub mode: LaunchMode,
  /// Context length. `None` lets `llama-server` use the GGUF's
  /// native value (no `-c` flag).
  pub ctx: Option<u32>,
  /// Listening port. `None` leaves port allocation to the supervisor.
  pub port: Option<u16>,
  /// Reasoning bundle on/off. When `true`, supervisor appends
  /// `--jinja --reasoning-format deepseek` to the argv.
  pub reasoning: bool,
  /// Free-form pass-through flags. The TUI's advanced panel and the
  /// CLI's `-- ...` tail both flow into here.
  pub advanced: Vec<OsString>,
}

impl LaunchParams {
  pub fn new(model_path: PathBuf, mode: LaunchMode) -> Self {
    Self {
      model_path,
      mode,
      ctx: None,
      port: None,
      reasoning: false,
      advanced: Vec::new(),
    }
  }
}

/// Materialise the argv `Command::args(...)` will hand to
/// `llama-server`. Caller passes the resolved listening port
/// separately because allocation happens in the supervisor, not in
/// `LaunchParams`.
pub fn compose(params: &LaunchParams, allocated_port: u16) -> Vec<OsString> {
  let mut argv: Vec<OsString> = Vec::with_capacity(16 + params.advanced.len());
  argv.push("--host".into());
  argv.push("127.0.0.1".into());
  argv.push("--port".into());
  argv.push(allocated_port.to_string().into());
  argv.push("-m".into());
  argv.push(params.model_path.clone().into());
  match params.mode {
    LaunchMode::Chat => {}
    LaunchMode::Embedding => argv.push("--embeddings".into()),
    LaunchMode::Rerank => argv.push("--reranking".into()),
  }
  if params.reasoning {
    argv.push("--jinja".into());
    argv.push("--reasoning-format".into());
    argv.push("deepseek".into());
  }
  if let Some(ctx) = params.ctx {
    argv.push("-c".into());
    argv.push(ctx.to_string().into());
  }
  argv.extend(params.advanced.iter().cloned());
  argv
}

#[cfg(test)]
mod tests {
  use super::*;

  fn strs(args: &[OsString]) -> Vec<String> {
    args
      .iter()
      .map(|s| s.to_string_lossy().into_owned())
      .collect()
  }

  fn base_params() -> LaunchParams {
    LaunchParams::new(PathBuf::from("/m/model.gguf"), LaunchMode::Chat)
  }

  #[test]
  fn chat_mode_emits_canonical_argv_prefix() {
    let p = base_params();
    let argv = strs(&compose(&p, 41100));
    let head: Vec<&str> = argv.iter().map(String::as_str).take(6).collect();
    assert_eq!(
      head,
      vec![
        "--host",
        "127.0.0.1",
        "--port",
        "41100",
        "-m",
        "/m/model.gguf"
      ]
    );
    // Chat mode adds no embedding/rerank flag.
    assert!(!argv
      .iter()
      .any(|a| a == "--embeddings" || a == "--reranking"));
  }

  #[test]
  fn embedding_mode_adds_embeddings_flag() {
    let mut p = base_params();
    p.mode = LaunchMode::Embedding;
    let argv = strs(&compose(&p, 41100));
    assert!(argv.iter().any(|a| a == "--embeddings"));
    assert!(!argv.iter().any(|a| a == "--reranking"));
  }

  #[test]
  fn rerank_mode_adds_reranking_flag() {
    let mut p = base_params();
    p.mode = LaunchMode::Rerank;
    let argv = strs(&compose(&p, 41100));
    assert!(argv.iter().any(|a| a == "--reranking"));
  }

  #[test]
  fn reasoning_bundles_jinja_and_deepseek() {
    let mut p = base_params();
    p.reasoning = true;
    let argv = strs(&compose(&p, 41100));
    assert!(argv.iter().any(|a| a == "--jinja"));
    let i = argv.iter().position(|a| a == "--reasoning-format").unwrap();
    assert_eq!(argv[i + 1], "deepseek");
  }

  #[test]
  fn ctx_override_emits_dash_c() {
    let mut p = base_params();
    p.ctx = Some(32768);
    let argv = strs(&compose(&p, 41100));
    let i = argv.iter().position(|a| a == "-c").unwrap();
    assert_eq!(argv[i + 1], "32768");
  }

  #[test]
  fn ctx_unset_omits_dash_c() {
    let p = base_params();
    let argv = strs(&compose(&p, 41100));
    assert!(!argv.iter().any(|a| a == "-c"));
  }

  #[test]
  fn advanced_flags_land_at_the_end_to_override_bundled() {
    let mut p = base_params();
    p.reasoning = true;
    p.advanced = vec![
      // User wants raw reasoning format despite the reasoning bundle.
      OsString::from("--reasoning-format"),
      OsString::from("none"),
      OsString::from("--threads"),
      OsString::from("8"),
    ];
    let argv = strs(&compose(&p, 41100));
    // Last occurrence of `--reasoning-format` wins because
    // `llama-server` honours the right-most flag — that's the basis
    // of the "advanced flags trump bundled" contract.
    let positions: Vec<usize> = argv
      .iter()
      .enumerate()
      .filter(|(_, a)| *a == "--reasoning-format")
      .map(|(i, _)| i)
      .collect();
    assert_eq!(positions.len(), 2, "bundled + override both present");
    let last = *positions.last().unwrap();
    assert_eq!(argv[last + 1], "none", "advanced override is last");
  }

  #[test]
  fn allocated_port_appears_after_port_flag() {
    let p = base_params();
    let argv = strs(&compose(&p, 41200));
    let i = argv.iter().position(|a| a == "--port").unwrap();
    assert_eq!(argv[i + 1], "41200");
  }
}
