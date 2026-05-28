//! Interactive cliclack pickers for `start` and `stop` when called
//! without a positional argument.
//!
//! Both pickers refuse non-interactive contexts up front (no TTY or
//! `--json`) so a piped / CI invocation gets an actionable error
//! instead of a hung prompt. The cliclack call itself runs inside
//! `spawn_blocking` because the underlying terminal reads are sync.

use std::io::IsTerminal;

use crate::cli::exit_codes::{CliExit, MODEL_NOT_FOUND, USAGE};
use crate::cli::resolve::{CatalogRow, RunningRow};

/// Refuse the picker up front when the caller can't drive it. Returns
/// `Ok(())` when an interactive picker is allowed.
fn ensure_interactive(json: bool, what: &str) -> Result<(), CliExit> {
  if json {
    return Err(CliExit::new(
      USAGE,
      format!("interactive {what} picker is disabled with --json; pass an explicit argument"),
    ));
  }
  if !std::io::stdin().is_terminal() {
    return Err(CliExit::new(
      USAGE,
      format!("interactive {what} picker requires a TTY; pass an explicit argument"),
    ));
  }
  Ok(())
}

/// Open a cliclack picker over the catalog and return the chosen row.
/// Wired into `start` when the user omits the model reference.
pub async fn pick_catalog_row(rows: &[CatalogRow], json: bool) -> Result<CatalogRow, CliExit> {
  ensure_interactive(json, "start")?;
  if rows.is_empty() {
    return Err(CliExit::new(MODEL_NOT_FOUND, "no models discovered"));
  }
  let owned: Vec<CatalogRow> = rows.to_vec();
  let chosen: CatalogRow = tokio::task::spawn_blocking(move || {
    let mut select = cliclack::select::<usize>("Pick a model to start").initial_value(0);
    for (i, r) in owned.iter().enumerate() {
      let label = r.name();
      let hint = catalog_hint(r);
      select = select.item(i, label, hint);
    }
    let idx = select.interact()?;
    Ok::<_, std::io::Error>(
      owned
        .into_iter()
        .nth(idx)
        .expect("select returns a valid idx"),
    )
  })
  .await
  .map_err(|e| CliExit::new(USAGE, format!("start picker join: {e}")))?
  .map_err(|e| CliExit::new(USAGE, format!("start picker: {e}")))?;
  Ok(chosen)
}

/// Open a cliclack picker over running supervisors and return the
/// chosen `launch_id`. Wired into `stop` when the user omits both
/// `<target>` and `--all`.
pub async fn pick_running_target(rows: &[RunningRow], json: bool) -> Result<String, CliExit> {
  ensure_interactive(json, "stop")?;
  if rows.is_empty() {
    return Err(CliExit::new(MODEL_NOT_FOUND, "no managed launches to stop"));
  }
  let owned: Vec<RunningRow> = rows.to_vec();
  let chosen: String = tokio::task::spawn_blocking(move || {
    let mut select = cliclack::select::<usize>("Pick a launch to stop").initial_value(0);
    for (i, r) in owned.iter().enumerate() {
      let label = format!("{lid} {name}", lid = r.launch_id, name = r.name());
      let hint = format!(":{port} {state}", port = r.port, state = r.state);
      select = select.item(i, label, hint);
    }
    let idx = select.interact()?;
    Ok::<_, std::io::Error>(
      owned
        .into_iter()
        .nth(idx)
        .expect("select returns a valid idx")
        .launch_id,
    )
  })
  .await
  .map_err(|e| CliExit::new(USAGE, format!("stop picker join: {e}")))?
  .map_err(|e| CliExit::new(USAGE, format!("stop picker: {e}")))?;
  Ok(chosen)
}

/// Hint line for a catalog picker row — mirrors the columns the
/// `list` table shows so the user sees the same identifying info in
/// both surfaces.
fn catalog_hint(r: &CatalogRow) -> String {
  let arch = r.arch.as_deref().unwrap_or("?");
  let quant = r.quant.as_deref().unwrap_or("?");
  let ctx = r
    .native_ctx
    .map(|n| n.to_string())
    .unwrap_or_else(|| "?".to_string());
  format!("{arch} {quant} ctx={ctx}")
}
