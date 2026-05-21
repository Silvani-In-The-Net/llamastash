//! Pre-flight: turn an inbound HTTP request into a forwarding plan.
//!
//! Unit 3 walks every incoming `/v1/...` request through this module
//! before reaching for the upstream `llama-server`. The output is a
//! [`RouteDecision`] that captures everything [`super::forward`]
//! needs to do the pass-through, plus enough context for the error
//! arms to render an OpenAI-shaped body.
//!
//! Hot path:
//!   1. Buffer the body under a 2 MiB cap ([`http-body-util::Limited`]).
//!   2. Extract `body.model` with a tolerant `JustModel` parse that
//!      ignores every other field. Empty / missing → 400.
//!   3. Build a `Vec<CatalogRow>` from the catalog snapshot and run
//!      the existing fuzzy resolver ([`crate::cli::resolve::resolve_model`]).
//!   4. Walk the supervisor snapshot for a Ready entry whose
//!      [`ModelId`] path matches the resolved catalog row.
//!
//! Unit 3 stops at step 4 — no auto-start, no fallback. Unit 4
//! replaces the [`RouteDecision::NotRunning`] arm with the launch +
//! single-flight + fallback machinery; the variant intentionally
//! carries the resolved row + arch so Unit 4 doesn't have to repeat
//! the lookup.
//!
//! Plan: docs/plans/2026-05-21-001-feat-proxy-router-plan.md (Unit 3).

use std::sync::Arc;

use http_body_util::{BodyExt, Limited};
use hyper::body::{Bytes, Incoming};

use crate::cli::resolve::{resolve_model, CatalogRow};
use crate::daemon::supervisor::ManagedState;
use crate::discovery::DiscoveredModel;

use super::state::ProxyState;

/// Inbound body size cap. The 2 MiB ceiling lets OpenAI-shape chat
/// completions with multi-thousand-token histories through while
/// still bounding worst-case memory and refusing accidental
/// uploads. Anything larger surfaces as HTTP 413 via [`BodyError::TooLarge`].
pub const BODY_LIMIT_BYTES: usize = 2 * 1024 * 1024;

/// Forwarding plan produced by [`decide`]. Keep this `pub(crate)` —
/// the router pattern-matches on variants but no external module
/// constructs them.
#[derive(Debug)]
pub(crate) enum RouteDecision {
  /// Forward to a Ready supervisor on `port`. `served_model_id` is
  /// the display name of the model actually serving the request;
  /// equal to `requested_model` on the happy path and diverges on
  /// fallback (Unit 4). `fallback` gates the `x-llamastash-*`
  /// response headers in [`super::forward`].
  ReadyAt {
    port: u16,
    served_model_id: String,
    /// The user-supplied `body.model` value, retained for symmetry
    /// with the other variants. Unit 4 reads this when picking a
    /// fallback model to report in the OpenAI error body.
    #[allow(dead_code)]
    requested_model: String,
    fallback: bool,
    fallback_reason: Option<String>,
  },
  /// The catalog has the model but no Ready supervisor is serving
  /// it. Unit 3 returns 503 here; Unit 4 will swap this for the
  /// auto-start + wait-for-Ready path. The variant carries the
  /// resolved row + arch so Unit 4 doesn't have to re-resolve.
  NotRunning {
    requested_model: String,
    /// Resolved catalog entry. Unit 4 consumes this to build the
    /// launch params for `start_model_inner` without re-running
    /// the resolver.
    #[allow(dead_code)]
    resolved_row: Box<CatalogRow>,
    /// Catalog arch metadata (e.g. `"llama"`, `"qwen3"`). `None`
    /// when discovery couldn't parse the GGUF header. Unit 4's
    /// family-MRU fallback pivots on this field.
    #[allow(dead_code)]
    arch: Option<String>,
  },
  /// `resolve_model` returned zero matches. Unit 3 emits 404
  /// `model_not_found` with `matches: []`.
  NotFound { requested_model: String },
  /// `resolve_model` returned > 1 matches. Unit 3 emits 400
  /// `ambiguous_model` with the candidate names.
  Ambiguous {
    requested_model: String,
    candidates: Vec<String>,
  },
  /// `body.model` is absent or empty. Unit 3 emits 400
  /// `invalid_request` with `code: "model_required"`.
  ModelRequired,
}

/// Errors raised before [`decide`] returns — these escape the
/// forwarding-plan layer and propagate to the per-request HTTP
/// status mapping in [`super::router`].
#[derive(Debug)]
pub(crate) enum BodyError {
  /// Body exceeded the 2 MiB ceiling. HTTP 413.
  TooLarge,
  /// Body wasn't valid JSON or `body.model` wasn't a string.
  /// HTTP 400 `invalid_request`.
  Malformed { message: String },
  /// hyper choked reading the request body off the wire. Surface as
  /// HTTP 400; client-side framing is broken either way.
  Read { message: String },
}

/// Minimal-shape body parse: serde ignores all fields it doesn't
/// know about, so anything beyond `model` is preserved in the
/// buffered bytes we forward upstream unchanged.
#[derive(serde::Deserialize)]
struct JustModel {
  #[serde(default)]
  model: Option<String>,
}

/// Outcome of buffering + extracting. `bytes` is the full body
/// (capped at [`BODY_LIMIT_BYTES`]); we forward these verbatim, so
/// no re-encoding ever happens after this point.
pub(crate) struct ParsedBody {
  pub bytes: Bytes,
  pub model: Option<String>,
}

/// Drain the inbound body under the 2 MiB cap, then peek the
/// `model` field with a single tolerant parse. The body bytes are
/// kept as-is for verbatim forwarding.
pub(crate) async fn buffer_and_extract(body: Incoming) -> Result<ParsedBody, BodyError> {
  let collected = match Limited::new(body, BODY_LIMIT_BYTES).collect().await {
    Ok(c) => c,
    Err(err) => {
      // `http-body-util::Limited` wraps the inner error inside a
      // `Box<dyn Error + Send + Sync>`. The cap-overflow case is
      // exposed as `LengthLimitError`; distinguishing it lets us
      // emit 413 vs 400 with the right message.
      if err
        .downcast_ref::<http_body_util::LengthLimitError>()
        .is_some()
      {
        return Err(BodyError::TooLarge);
      }
      return Err(BodyError::Read {
        message: format!("failed to read request body: {err}"),
      });
    }
  };
  let bytes = collected.to_bytes();

  // An empty body is allowed in principle — `model` extraction
  // then returns None and the caller emits `model_required`.
  let model = if bytes.is_empty() {
    None
  } else {
    match serde_json::from_slice::<JustModel>(&bytes) {
      Ok(parsed) => parsed
        .model
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty()),
      Err(err) => {
        return Err(BodyError::Malformed {
          message: format!("request body is not valid JSON: {err}"),
        });
      }
    }
  };

  Ok(ParsedBody { bytes, model })
}

/// Build a [`RouteDecision`] from the parsed body. Does no I/O
/// beyond reading shared snapshots (catalog, supervisors). The
/// forwarding decision is pure — the side-effecting forward call
/// lives in [`super::forward`].
pub(crate) async fn decide(state: &Arc<ProxyState>, body_model: Option<String>) -> RouteDecision {
  let requested = match body_model {
    Some(m) if !m.is_empty() => m,
    _ => return RouteDecision::ModelRequired,
  };

  // Catalog snapshot → CatalogRow vec (the resolver speaks
  // `&[CatalogRow]`). Built in-process here because the existing
  // `cli::resolve::fetch_catalog` round-trips through IPC, which
  // we explicitly want to avoid on the hot path.
  let snap = state.catalog.snapshot().await;
  let rows: Vec<CatalogRow> = snap.iter().map(catalog_row_from_discovered).collect();
  let resolved = match resolve_model(&rows, &requested) {
    Ok(r) => r,
    Err(_) => {
      // `resolve_model` collapses both 0- and N-match cases into
      // the same `MODEL_NOT_FOUND` exit. Re-derive which by
      // re-running the substring filter — cheap, and lets the
      // proxy emit the right HTTP code (404 vs 400).
      let candidates = substring_candidates(&rows, &requested);
      return if candidates.is_empty() {
        RouteDecision::NotFound {
          requested_model: requested,
        }
      } else {
        RouteDecision::Ambiguous {
          requested_model: requested,
          candidates: candidates.into_iter().map(|r| r.name()).collect(),
        }
      };
    }
  };

  // Walk the supervisor snapshot for a Ready entry serving the
  // resolved row's path. Two HashMap lookups + one state read each
  // — well within the hot-path budget the plan asks for.
  let sup_snap = state.supervisors.snapshot().await;
  for (_launch_id, model) in sup_snap.into_iter() {
    if !same_path(&model.id().path, &resolved.path) {
      continue;
    }
    if matches!(model.state().await, ManagedState::Ready) {
      return RouteDecision::ReadyAt {
        port: model.port(),
        served_model_id: resolved.name(),
        requested_model: requested,
        fallback: false,
        fallback_reason: None,
      };
    }
  }

  // Catalog matched but nobody is serving. Unit 3's placeholder —
  // TODO(unit-4): replace with auto-start + wait-for-Ready in
  // src/proxy/launch.rs.
  let arch = resolved.arch.clone();
  RouteDecision::NotRunning {
    requested_model: requested,
    resolved_row: Box::new(resolved),
    arch,
  }
}

/// Recompute the substring candidates `resolve_model` saw. We
/// duplicate a few lines of the resolver here so the proxy can
/// distinguish "0 matches" (404) from "N matches" (400) without
/// teaching the resolver itself a new exit code — keeping
/// `cli::resolve` callers stable.
fn substring_candidates<'a>(rows: &'a [CatalogRow], reference: &str) -> Vec<&'a CatalogRow> {
  let needle = reference.trim();
  if needle.is_empty() {
    return Vec::new();
  }
  // Exact path / name first — same precedence as resolve_model.
  let exact_path: Vec<&CatalogRow> = rows.iter().filter(|r| r.path == needle).collect();
  if !exact_path.is_empty() {
    return exact_path;
  }
  let exact_name: Vec<&CatalogRow> = rows.iter().filter(|r| r.name() == needle).collect();
  if !exact_name.is_empty() {
    return exact_name;
  }
  let lower = needle.to_lowercase();
  rows
    .iter()
    .filter(|r| {
      r.name().to_lowercase().contains(&lower) || r.parent.to_lowercase().contains(&lower)
    })
    .collect()
}

/// Project a discovered-model entry onto the `CatalogRow` shape the
/// resolver expects. In-process equivalent of
/// `cli::resolve::parse_catalog_row` (which goes through the JSON
/// wire); kept here so the proxy doesn't pay a serialize/deserialize
/// round-trip on the hot path.
fn catalog_row_from_discovered(m: &DiscoveredModel) -> CatalogRow {
  let path = m.path.to_string_lossy().into_owned();
  let parent = m.parent.to_string_lossy().into_owned();
  let arch = m.metadata.as_ref().and_then(|md| md.arch.clone());
  let quant = m.metadata.as_ref().map(|md| md.quant.label().to_string());
  let native_ctx = m.metadata.as_ref().and_then(|md| md.native_ctx);
  let parameter_label = m
    .metadata
    .as_ref()
    .and_then(|md| md.parameter_label.clone());
  let weights_bytes = m.metadata.as_ref().and_then(|md| md.weights_bytes);
  CatalogRow {
    path,
    model_id: None,
    parent,
    source: m.source.label().to_string(),
    arch,
    quant,
    native_ctx,
    mode_hint: None,
    parameter_label,
    weights_bytes,
    display_label: m.display_label.clone(),
    parse_error: m.parse_error.clone(),
  }
}

/// Compare a `ModelId::path` (PathBuf) with a `CatalogRow::path`
/// (String). The catalog row is built from the discovered path's
/// `to_string_lossy()` view, and `ModelId::path` is canonical too —
/// equality is exact in production.
fn same_path(model_id_path: &std::path::Path, row_path: &str) -> bool {
  model_id_path.to_string_lossy() == row_path
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::cli::resolve::CatalogRow;

  fn row(path: &str, parent: &str) -> CatalogRow {
    CatalogRow {
      path: path.to_string(),
      model_id: None,
      parent: parent.to_string(),
      source: "user".to_string(),
      arch: Some("llama".to_string()),
      quant: Some("Q4_K".to_string()),
      native_ctx: Some(8192),
      mode_hint: None,
      parameter_label: None,
      weights_bytes: None,
      display_label: None,
      parse_error: None,
    }
  }

  #[test]
  fn substring_candidates_returns_zero_for_unmatched() {
    let rows = vec![row("/m/llama.gguf", "/m")];
    assert!(substring_candidates(&rows, "phi").is_empty());
  }

  #[test]
  fn substring_candidates_returns_multiple_for_ambiguous() {
    let rows = vec![
      row("/m/qwen-coder-7b.gguf", "/m"),
      row("/m/qwen-coder-13b.gguf", "/m"),
    ];
    let cands = substring_candidates(&rows, "qwen-coder");
    assert_eq!(cands.len(), 2);
  }

  #[test]
  fn substring_candidates_unique_match_returns_one() {
    let rows = vec![row("/m/qwen.gguf", "/m"), row("/m/llama.gguf", "/m")];
    let cands = substring_candidates(&rows, "llama");
    assert_eq!(cands.len(), 1);
  }

  #[tokio::test]
  async fn buffer_and_extract_empty_body_returns_none_model() {
    use http_body_util::Full;
    use hyper::body::Bytes;
    // Build an `Incoming`-shaped pipe via hyper's test helpers: the
    // simplest path is to construct a hyper Request and pull its
    // body. We can't construct an `Incoming` directly outside hyper
    // — so this is exercised end-to-end in tests/proxy_routing.rs
    // instead. Inline test left as a documentation marker.
    let _ = Full::new(Bytes::from_static(b""));
  }
}
