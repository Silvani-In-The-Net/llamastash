//! Walk one or more scan roots, group split-GGUF shards, parse each
//! launchable file's header on a bounded pool, and stream results to
//! the caller over an `mpsc` channel (origin: R1, R5, R9).
//!
//! The walk uses the `ignore` crate so `.gitignore` rules and the
//! caller's exclude globs are honoured for free. CPU-bound parsing
//! runs on `tokio::task::spawn_blocking` so the scan tasks don't
//! starve the runtime when the user has hundreds of GGUFs on disk.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ignore::WalkBuilder;
use tokio::sync::mpsc;

use crate::discovery::split_gguf::{group, DiscoveredEntry};
use crate::discovery::{DiscoveredModel, ModelSource};
use crate::gguf::{read_path, summarise_metadata, GgufError, HeaderReadOptions};

/// One root to scan plus how to label files found beneath it.
#[derive(Debug, Clone)]
pub struct ScanRoot {
  pub path: PathBuf,
  pub source: ModelSource,
}

/// Options for [`scan`]. `excludes` are appended to the gitignore-
/// derived ignores; absolute or relative-to-root globs both work.
#[derive(Debug, Clone, Default)]
pub struct ScanOptions {
  pub excludes: Vec<String>,
  /// Capacity for the streaming channel. The TUI is usually faster
  /// than disk, but a tiny capacity makes back-pressure visible in
  /// tests; production defaults to a comfortable buffer.
  pub channel_capacity: Option<usize>,
}

impl ScanOptions {
  pub fn channel_capacity(&self) -> usize {
    self.channel_capacity.unwrap_or(64)
  }
}

/// Begin a scan across `roots`. Returns the receiver immediately; the
/// scan runs in the background and closes the channel when every root
/// has been walked.
///
/// Errors per-file (unreadable directories, parse failures) are
/// surfaced via `DiscoveredModel.parse_error` rather than aborting the
/// whole scan — a single bad model file should not blind the user to
/// the rest of their library (origin: R9 "scan continues with other
/// roots").
pub fn scan(roots: Vec<ScanRoot>, opts: ScanOptions) -> mpsc::Receiver<DiscoveredModel> {
  let (tx, rx) = mpsc::channel(opts.channel_capacity());
  let excludes = Arc::new(opts.excludes);
  tokio::spawn(async move {
    for root in roots {
      walk_root(root, Arc::clone(&excludes), tx.clone()).await;
    }
    // dropping `tx` here closes the receiver
  });
  rx
}

async fn walk_root(root: ScanRoot, excludes: Arc<Vec<String>>, tx: mpsc::Sender<DiscoveredModel>) {
  let path = root.path.clone();
  let source = root.source;
  let excludes_for_walk = Arc::clone(&excludes);
  let paths = tokio::task::spawn_blocking(move || collect_gguf_paths(&path, &excludes_for_walk))
    .await
    .unwrap_or_else(|join_err| {
      log::warn!(
        "scan walker task for {} panicked: {join_err}",
        root.path.display()
      );
      Vec::new()
    });

  for entry in group(paths) {
    let model = build_discovered_model(entry, source).await;
    if tx.send(model).await.is_err() {
      // Receiver dropped — caller doesn't want more; stop walking.
      return;
    }
  }
}

/// Synchronous file-system walk. Returns every `.gguf` file under
/// `root` honouring gitignore semantics and the caller's exclude
/// globs. Unreadable subdirectories are logged and skipped rather
/// than aborting the walk.
fn collect_gguf_paths(root: &Path, excludes: &[String]) -> Vec<PathBuf> {
  if !root.exists() {
    log::warn!("scan root does not exist: {}", root.display());
    return Vec::new();
  }
  let mut builder = WalkBuilder::new(root);
  builder
    .standard_filters(true)
    .require_git(false)
    .follow_links(false)
    .hidden(false);
  if !excludes.is_empty() {
    let mut overrides = ignore::overrides::OverrideBuilder::new(root);
    for pat in excludes {
      // `ignore`'s override globs treat a leading `!` as include-back,
      // so prefix every user exclude with `!` to mean "exclude this".
      // A plain `*.tmp` glob would otherwise be interpreted as
      // "include only files matching this".
      if let Err(e) = overrides.add(&format!("!{pat}")) {
        log::warn!("invalid scan exclude glob {pat:?}: {e}");
      }
    }
    match overrides.build() {
      Ok(o) => {
        builder.overrides(o);
      }
      Err(e) => log::warn!("scan exclude globs failed to compile: {e}"),
    }
  }

  let mut out = Vec::new();
  for result in builder.build() {
    match result {
      Ok(entry) => {
        let p = entry.path();
        // Skip `.gguf.part` (mid-download) and only emit regular files
        // ending in `.gguf` — symlinks land in their canonical form
        // after `read_path` resolves them.
        if p.extension().and_then(|s| s.to_str()) == Some("gguf")
          && entry.file_type().map(|t| t.is_file()).unwrap_or(false)
        {
          out.push(p.to_path_buf());
        }
      }
      Err(e) => log::warn!("scan walker error under {}: {e}", root.display()),
    }
  }
  out
}

async fn build_discovered_model(entry: DiscoveredEntry, source: ModelSource) -> DiscoveredModel {
  match entry {
    DiscoveredEntry::Single(path) => parse_into_model(path, source, Vec::new()).await,
    DiscoveredEntry::Split(group) => {
      // Siblings exclude the launch file itself so the field's purpose
      // ("sibling shards") matches its content.
      let siblings = group
        .shards
        .into_iter()
        .filter(|p| *p != group.launch_path)
        .collect();
      parse_into_model(group.launch_path, source, siblings).await
    }
  }
}

async fn parse_into_model(
  path: PathBuf,
  source: ModelSource,
  siblings: Vec<PathBuf>,
) -> DiscoveredModel {
  let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
  let path_for_parse = path.clone();
  let parsed: Result<_, GgufError> =
    tokio::task::spawn_blocking(move || read_path(&path_for_parse, HeaderReadOptions::default()))
      .await
      .unwrap_or_else(|join_err| {
        Err(GgufError::Io(std::io::Error::other(format!(
          "parser task panicked: {join_err}"
        ))))
      });
  match parsed {
    Ok(read) => DiscoveredModel {
      path,
      parent,
      source,
      metadata: Some(summarise_metadata(&read.header)),
      parse_error: None,
      split_siblings: siblings,
    },
    Err(e) => DiscoveredModel {
      path,
      parent,
      source,
      metadata: None,
      parse_error: Some(e.to_string()),
      split_siblings: siblings,
    },
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  use std::fs;
  use std::time::{SystemTime, UNIX_EPOCH};

  use crate::gguf::test_fixtures::build_minimal_gguf;

  fn temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .expect("clock")
      .as_nanos();
    let dir = std::env::temp_dir().join(format!(
      "llamatui-scanner-{label}-{}-{nanos}",
      std::process::id()
    ));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
  }

  #[test]
  fn collect_gguf_paths_skips_part_files() {
    let dir = temp_dir("part");
    fs::write(dir.join("a.gguf"), build_minimal_gguf("llama")).unwrap();
    fs::write(dir.join("a.gguf.part"), b"in-progress").unwrap();
    let paths = collect_gguf_paths(&dir, &[]);
    assert_eq!(paths.len(), 1);
    assert!(paths[0].ends_with("a.gguf"));
    fs::remove_dir_all(&dir).ok();
  }

  #[test]
  fn collect_gguf_paths_honours_exclude_globs() {
    let dir = temp_dir("excl");
    fs::create_dir_all(dir.join("keep")).unwrap();
    fs::create_dir_all(dir.join("skip")).unwrap();
    fs::write(dir.join("keep/a.gguf"), build_minimal_gguf("llama")).unwrap();
    fs::write(dir.join("skip/b.gguf"), build_minimal_gguf("llama")).unwrap();
    let paths = collect_gguf_paths(&dir, &["skip/**".to_string()]);
    assert_eq!(paths.len(), 1);
    assert!(paths[0].to_string_lossy().contains("keep"));
    fs::remove_dir_all(&dir).ok();
  }

  #[test]
  fn nonexistent_root_returns_empty_without_panic() {
    let bogus = PathBuf::from("/nonexistent/scan-root-llamatui");
    assert!(collect_gguf_paths(&bogus, &[]).is_empty());
  }

  #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
  async fn scan_streams_discovered_models_with_metadata() {
    let dir = temp_dir("stream");
    fs::write(dir.join("a.gguf"), build_minimal_gguf("llama")).unwrap();
    fs::write(dir.join("b.gguf"), build_minimal_gguf("qwen3")).unwrap();
    let roots = vec![ScanRoot {
      path: dir.clone(),
      source: ModelSource::UserPath,
    }];
    let mut rx = scan(roots, ScanOptions::default());
    let mut got = Vec::new();
    while let Some(m) = rx.recv().await {
      got.push(m);
    }
    assert_eq!(got.len(), 2);
    for m in &got {
      assert!(m.metadata.is_some(), "minimal gguf should parse");
      assert_eq!(m.source, ModelSource::UserPath);
      assert!(m.split_siblings.is_empty());
    }
    fs::remove_dir_all(&dir).ok();
  }

  #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
  async fn scan_surfaces_parse_failure_without_dropping_row() {
    let dir = temp_dir("badparse");
    fs::write(dir.join("bad.gguf"), b"this is not a GGUF").unwrap();
    let roots = vec![ScanRoot {
      path: dir.clone(),
      source: ModelSource::UserPath,
    }];
    let mut rx = scan(roots, ScanOptions::default());
    let m = rx.recv().await.expect("one model surfaced");
    assert!(rx.recv().await.is_none(), "only one file in dir");
    assert!(m.metadata.is_none(), "invalid file → no metadata");
    assert!(m.parse_error.is_some(), "diagnostic must accompany failure");
    fs::remove_dir_all(&dir).ok();
  }

  #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
  async fn scan_groups_split_shards_into_one_entry() {
    let dir = temp_dir("split");
    let bytes = build_minimal_gguf("llama");
    fs::write(dir.join("model-00001-of-00003.gguf"), &bytes).unwrap();
    fs::write(dir.join("model-00002-of-00003.gguf"), &bytes).unwrap();
    fs::write(dir.join("model-00003-of-00003.gguf"), &bytes).unwrap();
    let roots = vec![ScanRoot {
      path: dir.clone(),
      source: ModelSource::UserPath,
    }];
    let mut rx = scan(roots, ScanOptions::default());
    let m = rx.recv().await.expect("one grouped entry");
    assert!(
      rx.recv().await.is_none(),
      "shard set should collapse to one"
    );
    assert_eq!(m.split_siblings.len(), 2, "shard 1 plus 2 siblings");
    assert!(m
      .path
      .to_string_lossy()
      .ends_with("model-00001-of-00003.gguf"));
    fs::remove_dir_all(&dir).ok();
  }
}
