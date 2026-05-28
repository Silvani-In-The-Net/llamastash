//! `llamastash show <model> [--json]`.
//!
//! One-stop projection of everything LlamaStash knows about a single
//! model: catalog row, GGUF metadata, on-disk size (summed across
//! split shards), the yaml + built-in `arch_defaults` that would
//! feed a launch, and the last `start_model` params recorded for
//! this file. Reuses the same resolver `start` and `/v1/...` use, so
//! a reference that works on one surface works here.

use serde_json::{json, Value};

use crate::cli::cli_args::{Cli, ShowArgs};
use crate::cli::client::connect_or_spawn;
use crate::cli::colors;
use crate::cli::exit_codes::{CliExit, CliResult};
use crate::cli::output::pretty_json;
use crate::cli::resolve::{fetch_catalog, resolve_model, CatalogRow};
use crate::config::Config;
use crate::daemon::host_metrics::GpuFlavor;
use crate::launch::defaults_table;

pub async fn handle(args: ShowArgs, cli: &Cli, config: &Config) -> CliResult {
  let mut client = connect_or_spawn(cli, config).await?;
  let catalog = fetch_catalog(&mut client).await?;
  let row = resolve_model(&catalog, &args.model)?;

  // Pull last-params for this model_path. The IPC handler keys by
  // ModelId; `model_path` is part of the JSON wire shape (`entry.id.path`)
  // and is unique within the catalog, so filtering by string equality
  // is sufficient here.
  let last_params_body = client
    .call("last_params_list", None)
    .await
    .map_err(CliExit::from_client_error)?;
  let last_params = last_params_body
    .get("last_params")
    .and_then(Value::as_array)
    .and_then(|rows| {
      rows.iter().find_map(|r| {
        let p = r.get("model_path").and_then(Value::as_str)?;
        if p == row.path {
          r.get("params").cloned()
        } else {
          None
        }
      })
    });

  // GPU backend from the daemon's host-metrics sampler — keys the
  // built-in arch_defaults lookup so the values we display match
  // what `start_model` would resolve.
  let status_body = client
    .call("status", None)
    .await
    .map_err(CliExit::from_client_error)?;
  let backend_label = status_body
    .get("host")
    .and_then(|h| h.get("gpu_backend"))
    .and_then(Value::as_str)
    .unwrap_or("");
  let backend = GpuFlavor::from_label(backend_label);

  // Built-in arch defaults for this (arch, backend) pair — the same
  // values that ship under `LayerLabel::ArchDefault` in the launch
  // resolver. Yaml arch_defaults sit on the same layer and win
  // per-field; surface both so the user sees where each field comes
  // from.
  let arch_key = row.arch.as_deref().unwrap_or("");
  let builtin_arch_defaults = defaults_table::lookup(arch_key, backend);
  let yaml_arch_defaults = row
    .arch
    .as_deref()
    .and_then(|a| config.arch_defaults.get(a))
    .cloned();

  let bytes = on_disk_total(&row);

  let envelope = json!({
    "name": row.name(),
    "path": row.path,
    "parent": row.parent,
    "source": row.source,
    "model_id": row.model_id,
    "display_label": row.display_label,
    "parse_error": row.parse_error,
    "metadata": {
      "arch": row.arch,
      "quant": row.quant,
      "native_ctx": row.native_ctx,
      "mode_hint": row.mode_hint,
      "parameter_label": row.parameter_label,
      "total_parameters": row.total_parameters,
      "tokenizer_kind": row.tokenizer_kind,
      "has_chat_template": row.has_chat_template,
      "has_reasoning_hint": row.has_reasoning_hint,
    },
    "size": {
      "weights_bytes": row.weights_bytes,
      "shard_count": 1 + row.split_siblings.len(),
      "on_disk_total_bytes": bytes,
      "split_siblings": row.split_siblings,
    },
    "arch_defaults": {
      "gpu_backend": format!("{backend:?}"),
      "yaml": yaml_arch_defaults,
      "builtin": builtin_arch_defaults,
    },
    "last_params": last_params,
  });

  if args.json {
    println!("{}", pretty_json(&envelope));
  } else {
    print!("{}", render_human(&row, &envelope));
  }
  Ok(())
}

/// Sum `path` + every sibling's on-disk size. `0` for any missing
/// path so a broken sibling shows up in the envelope (`shard_count`)
/// but doesn't error out the whole `show`.
fn on_disk_total(row: &CatalogRow) -> u64 {
  let primary = std::fs::metadata(&row.path).map(|m| m.len()).unwrap_or(0);
  let siblings: u64 = row
    .split_siblings
    .iter()
    .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
    .sum();
  primary.saturating_add(siblings)
}

fn render_human(row: &CatalogRow, env: &Value) -> String {
  use std::fmt::Write;
  let mut out = String::new();
  let kv = |buf: &mut String, key: &str, val: &str| {
    let _ = writeln!(buf, "  {}  {}", colors::dim(&format!("{key:<18}")), val);
  };

  let _ = writeln!(out, "{}", bold(&row.name()));
  kv(&mut out, "path", &row.path);
  kv(&mut out, "parent", &row.parent);
  kv(&mut out, "source", &row.source);
  if let Some(id) = &row.model_id {
    kv(&mut out, "model_id", id);
  }
  if let Some(lbl) = &row.display_label {
    kv(&mut out, "display_label", lbl);
  }
  if let Some(err) = &row.parse_error {
    kv(&mut out, "parse_error", &colors::warning(err));
  }

  let _ = writeln!(out, "\n{}", bold("metadata"));
  kv(&mut out, "arch", row.arch.as_deref().unwrap_or("—"));
  kv(&mut out, "quant", row.quant.as_deref().unwrap_or("—"));
  kv(
    &mut out,
    "native_ctx",
    &row
      .native_ctx
      .map(|n| n.to_string())
      .unwrap_or_else(|| "—".into()),
  );
  kv(
    &mut out,
    "mode_hint",
    row.mode_hint.as_deref().unwrap_or("—"),
  );
  kv(
    &mut out,
    "parameter_label",
    row.parameter_label.as_deref().unwrap_or("—"),
  );
  kv(
    &mut out,
    "tokenizer_kind",
    row.tokenizer_kind.as_deref().unwrap_or("—"),
  );
  kv(
    &mut out,
    "has_chat_template",
    if row.has_chat_template { "yes" } else { "no" },
  );
  kv(
    &mut out,
    "has_reasoning_hint",
    if row.has_reasoning_hint { "yes" } else { "no" },
  );

  let shard_count = 1 + row.split_siblings.len();
  let on_disk = env
    .get("size")
    .and_then(|s| s.get("on_disk_total_bytes"))
    .and_then(Value::as_u64)
    .unwrap_or(0);
  let _ = writeln!(out, "\n{}", bold("size"));
  if let Some(wb) = row.weights_bytes {
    kv(&mut out, "weights_bytes", &format_bytes(wb));
  } else {
    kv(&mut out, "weights_bytes", "—");
  }
  kv(&mut out, "shard_count", &shard_count.to_string());
  kv(&mut out, "on_disk_total", &format_bytes(on_disk));
  if !row.split_siblings.is_empty() {
    for (i, p) in row.split_siblings.iter().enumerate() {
      kv(&mut out, &format!("shard {}", i + 2), p);
    }
  }

  let backend = env
    .get("arch_defaults")
    .and_then(|a| a.get("gpu_backend"))
    .and_then(Value::as_str)
    .unwrap_or("");
  let _ = writeln!(
    out,
    "\n{} ({})",
    bold("arch_defaults"),
    colors::dim(backend),
  );
  let yaml = env.get("arch_defaults").and_then(|a| a.get("yaml"));
  let builtin = env.get("arch_defaults").and_then(|a| a.get("builtin"));
  kv(&mut out, "yaml", &knobs_one_line(yaml));
  kv(&mut out, "builtin", &knobs_one_line(builtin));

  let _ = writeln!(out, "\n{}", bold("last_params"));
  match env.get("last_params") {
    Some(Value::Null) | None => kv(&mut out, "(none)", "launch it once to populate"),
    Some(v) => kv(&mut out, "ctx", &fmt_field(v.get("ctx"))),
  }
  if let Some(v) = env.get("last_params") {
    if !v.is_null() {
      kv(&mut out, "mode", &fmt_field(v.get("mode")));
      kv(&mut out, "reasoning", &fmt_field(v.get("reasoning")));
      kv(&mut out, "knobs", &knobs_one_line(v.get("knobs")));
    }
  }

  out
}

fn fmt_field(v: Option<&Value>) -> String {
  match v {
    Some(Value::Null) | None => "—".into(),
    Some(Value::String(s)) => s.clone(),
    Some(other) => other.to_string(),
  }
}

fn knobs_one_line(value: Option<&Value>) -> String {
  let Some(Value::Object(map)) = value else {
    return "—".into();
  };
  let mut pairs: Vec<String> = map
    .iter()
    .filter(|(_, val)| !val.is_null())
    .map(|(key, val)| match val {
      Value::String(s) => format!("{key}={s}"),
      _ => format!("{key}={val}"),
    })
    .collect();
  pairs.sort();
  if pairs.is_empty() {
    "—".into()
  } else {
    pairs.join(", ")
  }
}

fn bold(s: &str) -> String {
  console::style(s).bold().to_string()
}

fn format_bytes(n: u64) -> String {
  const KIB: f64 = 1024.0;
  const MIB: f64 = KIB * 1024.0;
  const GIB: f64 = MIB * 1024.0;
  let nf = n as f64;
  if nf >= GIB {
    format!("{:.2} GiB", nf / GIB)
  } else if nf >= MIB {
    format!("{:.1} MiB", nf / MIB)
  } else if nf >= KIB {
    format!("{:.0} KiB", nf / KIB)
  } else {
    format!("{n} B")
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use serde_json::json;

  fn fake_row(path: &str) -> CatalogRow {
    CatalogRow {
      path: path.into(),
      model_id: Some("deadbeef".into()),
      parent: "/m".into(),
      source: "user".into(),
      arch: Some("qwen3".into()),
      quant: Some("Q5_K".into()),
      native_ctx: Some(32768),
      mode_hint: Some("chat".into()),
      parameter_label: Some("80B".into()),
      weights_bytes: Some(40_000_000_000),
      display_label: None,
      parse_error: None,
      split_siblings: vec![format!("{path}.part2"), format!("{path}.part3")],
      has_chat_template: true,
      has_reasoning_hint: false,
      tokenizer_kind: Some("qwen2".into()),
      total_parameters: Some(80_000_000_000),
    }
  }

  #[test]
  fn on_disk_total_includes_every_shard_when_files_exist() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("m.gguf");
    std::fs::write(&p, b"1234567890").unwrap();
    let s2 = dir.path().join("m.gguf-2");
    std::fs::write(&s2, b"abcdef").unwrap();
    let row = CatalogRow {
      path: p.display().to_string(),
      split_siblings: vec![s2.display().to_string()],
      ..fake_row("/m/x.gguf")
    };
    assert_eq!(on_disk_total(&row), 10 + 6);
  }

  #[test]
  fn on_disk_total_skips_missing_siblings_without_panicking() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("present.gguf");
    std::fs::write(&p, b"0123").unwrap();
    let row = CatalogRow {
      path: p.display().to_string(),
      split_siblings: vec!["/does/not/exist.gguf-2".into()],
      ..fake_row("/m/x.gguf")
    };
    assert_eq!(on_disk_total(&row), 4);
  }

  #[test]
  fn format_bytes_rolls_through_units() {
    assert_eq!(format_bytes(0), "0 B");
    assert_eq!(format_bytes(1023), "1023 B");
    assert_eq!(format_bytes(1024), "1 KiB");
    assert!(format_bytes(2 * 1024 * 1024).starts_with("2.0 MiB"));
    assert!(format_bytes(3 * 1024 * 1024 * 1024).starts_with("3.00 GiB"));
  }

  #[test]
  fn knobs_one_line_sorts_keys_and_drops_nulls() {
    let v = json!({
      "ctx": 8192,
      "reasoning": null,
      "n_gpu_layers": 99,
      "flash_attn": true,
    });
    let line = knobs_one_line(Some(&v));
    assert!(!line.contains("reasoning"));
    assert!(line.contains("ctx=8192"));
    assert!(line.contains("flash_attn=true"));
    assert!(line.contains("n_gpu_layers=99"));
    // Sorted alphabetically: ctx < flash_attn < n_gpu_layers.
    let ctx_idx = line.find("ctx=").unwrap();
    let flash_idx = line.find("flash_attn=").unwrap();
    let ngl_idx = line.find("n_gpu_layers=").unwrap();
    assert!(ctx_idx < flash_idx && flash_idx < ngl_idx);
  }

  #[test]
  fn knobs_one_line_returns_dash_for_empty_or_null() {
    assert_eq!(knobs_one_line(None), "—");
    assert_eq!(knobs_one_line(Some(&Value::Null)), "—");
    assert_eq!(knobs_one_line(Some(&json!({}))), "—");
    // All-null map collapses to dash too.
    assert_eq!(
      knobs_one_line(Some(&json!({"ctx": null, "reasoning": null}))),
      "—"
    );
  }
}
