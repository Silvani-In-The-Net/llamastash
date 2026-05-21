//! Per-request dispatch. `route` is the body of the `service_fn`
//! closure each hyper connection runs — a flat `match` over
//! `(method, path)` for the six fixed routes the proxy answers,
//! mirroring the style of [`crate::ipc::methods::dispatch_request`].
//!
//! Unit 1 stood up `/health`; Unit 2 adds `/v1/models`. The remaining
//! four arms (`/v1/chat/completions`, `/v1/completions`,
//! `/v1/embeddings`, `/v1/rerank`) stay 501 until Units 3/4 land the
//! resolution + forwarding plumbing.

use std::convert::Infallible;
use std::sync::Arc;

use http_body_util::{combinators::BoxBody, BodyExt, Empty, Full};
use hyper::body::{Bytes, Incoming};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::json;

use super::openai::{ErrorObject, ErrorResponse, ModelList, ModelObject};
use super::state::ProxyState;
use crate::daemon::supervisor::ManagedState;
use crate::discovery::DiscoveredModel;

/// The error type our `BoxBody` carries. We control every body we
/// emit (all in-memory `Bytes`), so an infallible error is the most
/// honest signal — chunks never fail at frame time. When Unit 3
/// starts piping reqwest's `bytes_stream()` through, the body alias
/// switches to a `BoxBody<Bytes, BoxError>` instead.
pub type BodyError = Infallible;

/// What every handler returns. `Result<_, hyper::Error>` is the
/// `service_fn` contract; the inner body is boxed so each arm can
/// pick whatever concrete `Body` makes sense without poisoning the
/// outer signature.
pub type ProxyResponse = Result<Response<BoxBody<Bytes, BodyError>>, hyper::Error>;

/// Entry point invoked by the `service_fn` closure. Returns a fully
/// constructed `Response`; the caller hands it back to hyper.
pub async fn route(state: Arc<ProxyState>, req: Request<Incoming>) -> ProxyResponse {
  let method = req.method().clone();
  let path = req.uri().path().to_string();

  // 6-route dispatch table. The five non-`/health` arms are 501
  // until later units replace them with the real handler bodies.
  // Keeping them named here rather than in a single `_ =>` catch-all
  // documents the surface and makes it obvious which units land
  // where.
  match (&method, path.as_str()) {
    (&Method::GET, "/health") => health(state).await,
    (&Method::GET, "/v1/models") => list_models(state).await,
    (&Method::POST, "/v1/chat/completions") => not_implemented(),
    (&Method::POST, "/v1/completions") => not_implemented(),
    (&Method::POST, "/v1/embeddings") => not_implemented(),
    (&Method::POST, "/v1/rerank") => not_implemented(),
    _ => not_found(),
  }
}

async fn health(state: Arc<ProxyState>) -> ProxyResponse {
  // `models_loaded` filters the supervisor snapshot to entries
  // currently in `ManagedState::Ready`. Unit 1 used `len()` as a
  // wire-shape stand-in; Unit 2 promotes it to the real Ready count
  // per R158 / R159. `models_discovered` is the catalog length —
  // discovery surfaces every row, even parse-error rows, so this
  // matches what `/v1/models` returns.
  let models_loaded = count_ready(&state).await;
  let models_discovered = state.catalog.len().await;
  let body = json!({
    "status": "ok",
    "models_loaded": models_loaded,
    "models_discovered": models_discovered,
  });
  // serde_json::to_vec on a hand-built `Value` cannot fail.
  let bytes = serde_json::to_vec(&body).expect("json encoding of fixed shape");
  Ok(json_response(StatusCode::OK, bytes))
}

/// Count supervisors currently in `ManagedState::Ready`. Each
/// `state()` call acquires a per-supervisor read lock, so the
/// snapshot is a sequence of cheap clones rather than one global
/// lock — consistent with how `status_handler` walks the registry.
async fn count_ready(state: &ProxyState) -> usize {
  let snap = state.supervisors.snapshot().await;
  let mut ready = 0usize;
  for (_id, model) in snap {
    if matches!(model.state().await, ManagedState::Ready) {
      ready += 1;
    }
  }
  ready
}

/// `GET /v1/models` — list every discovered model in OpenAI shape,
/// sorted alphabetically by `id`. Empty catalog returns
/// `{"object":"list","data":[]}` (not a 404, not an error).
async fn list_models(state: Arc<ProxyState>) -> ProxyResponse {
  let snap = state.catalog.snapshot().await;
  let mut rows: Vec<ModelObject> = snap
    .iter()
    .map(|m| ModelObject::new(model_id_for(m)))
    .collect();
  // ASCII-lexicographic sort: stable, deterministic across runs, and
  // independent of the catalog's underlying BTreeMap key (canonical
  // path) which orders by filesystem layout instead of display name.
  rows.sort_by(|a, b| a.id.cmp(&b.id));
  let list = ModelList::new(rows);
  let bytes = serde_json::to_vec(&list).expect("json encoding of fixed shape");
  Ok(json_response(StatusCode::OK, bytes))
}

/// Project a [`DiscoveredModel`] onto the `id` field of an OpenAI
/// `model` object. Rule: `display_label` wins when set (Ollama
/// surfaces `<name>:<tag>` here), otherwise fall back to
/// `path.file_stem()` via [`crate::util::paths::model_display_name`].
/// This matches what the TUI and `llamastash list` show, so the same
/// model identifier appears in every surface.
///
/// Note: `CatalogRow::name()` falls back to `path.file_name()`
/// (basename *with* extension) rather than the file stem. The plan
/// explicitly calls for the stem here so the OpenAI `id` reads
/// cleanly (`qwen2.5-coder` rather than `qwen2.5-coder.gguf`). The
/// resolver's substring matching (used in Unit 3) is tolerant to
/// either form, so this divergence is intentional and bounded.
fn model_id_for(m: &DiscoveredModel) -> String {
  if let Some(label) = &m.display_label {
    return label.clone();
  }
  crate::util::paths::model_display_name(&m.path)
}

fn not_implemented() -> ProxyResponse {
  // OpenAI-shaped error body so clients see a recognisable payload
  // even on the 501 placeholder. Units 3/4 swap this for the real
  // handler — the wire shape they emit will be the same.
  error_response(
    StatusCode::NOT_IMPLEMENTED,
    "not_implemented",
    "endpoint not implemented yet",
  )
}

fn not_found() -> ProxyResponse {
  error_response(StatusCode::NOT_FOUND, "not_found", "no such route")
}

/// Build an OpenAI-shaped error response from a `(status, type,
/// message)` triple. Centralised so the 501 / 404 / future Unit 3
/// `model_required` / `model_not_found` arms all emit the same
/// `{"error":{"type":..., "message":...}}` envelope.
fn error_response(status: StatusCode, r#type: &str, message: &str) -> ProxyResponse {
  let body = ErrorResponse {
    error: ErrorObject::new(r#type, message),
  };
  let bytes = serde_json::to_vec(&body).expect("json encoding of fixed shape");
  Ok(json_response(status, bytes))
}

fn json_response(status: StatusCode, body: Vec<u8>) -> Response<BoxBody<Bytes, BodyError>> {
  let body = Full::new(Bytes::from(body)).boxed();
  Response::builder()
    .status(status)
    .header(hyper::header::CONTENT_TYPE, "application/json")
    .body(body)
    .expect("static headers always parse")
}

/// Construct an empty body — kept here for future handler arms that
/// need a no-content response without re-importing the util crate.
#[allow(dead_code)]
pub(crate) fn empty_body() -> BoxBody<Bytes, BodyError> {
  Empty::<Bytes>::new().boxed()
}
