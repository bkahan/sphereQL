//! Doc-consistency drift check.
//!
//! Two checks that keep hand-maintained docs honest:
//!
//! 1. **Crate-table membership** — every workspace member listed in the
//!    root `Cargo.toml` must appear in *both* the README "Workspace
//!    layout" table and the `docs/architecture.md` crate table. These two
//!    tables document the same crate set in two places and drift apart
//!    when a new crate lands (the `scripts/check-versions` crate was
//!    missing from both until this check existed).
//!
//! 2. **Test-count floors** — the docs advertise approximate floors like
//!    "850+ Rust tests plus 200+ pytest tests". This check counts the
//!    real tests and fails if a stated floor is now *above* the actual
//!    count (the docs overstate) or so far *below* it that the floor has
//!    gone stale (bump it).
//!
//! # Usage
//!
//! ```sh
//! cargo run -p check-docs
//! ```
//!
//! Exits 0 when the docs match reality, 1 on drift.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use regex::Regex;
use walkdir::WalkDir;

/// How far a stated floor may trail the real count before it's "stale".
const FLOOR_STALENESS_BUDGET: i64 = 200;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .expect("failed to resolve workspace root")
}

fn read(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn skip_dir(name: &str) -> bool {
    matches!(
        name,
        "target" | ".git" | ".venv" | "node_modules" | "__pycache__" | ".pytest_cache"
    )
}

// ── Check 1: crate-table membership ────────────────────────────────────

/// Crate-shaped names (`sphereql*` / `scripts/*`) that appear in the
/// first column of a markdown table in `doc`.
fn crate_names_in_doc(doc: &str) -> BTreeSet<String> {
    // Match a leading `| `name` |` first column.
    let re = Regex::new(r"(?m)^\|\s*`([A-Za-z0-9_./-]+)`\s*\|").unwrap();
    re.captures_iter(doc)
        .map(|c| c[1].to_string())
        .filter(|n| n.starts_with("sphereql") || n.starts_with("scripts/"))
        .collect()
}

fn check_crate_tables(root: &Path, members: &[String], problems: &mut Vec<String>) {
    let readme = read(&root.join("README.md")).unwrap_or_default();
    let arch = read(&root.join("docs/architecture.md")).unwrap_or_default();
    let readme_set = crate_names_in_doc(&readme);
    let arch_set = crate_names_in_doc(&arch);

    for m in members {
        if !readme_set.contains(m) {
            problems.push(format!(
                "README.md crate table is missing workspace member `{m}`"
            ));
        }
        if !arch_set.contains(m) {
            problems.push(format!(
                "docs/architecture.md crate table is missing workspace member `{m}`"
            ));
        }
    }
}

// ── Check 2: test-count floors ─────────────────────────────────────────

fn count_matches(root: &Path, ext: &str, re: &Regex, subdirs: Option<&[&str]>) -> usize {
    let roots: Vec<PathBuf> = match subdirs {
        Some(dirs) => dirs.iter().map(|d| root.join(d)).collect(),
        None => vec![root.to_path_buf()],
    };
    let mut total = 0;
    for base in roots {
        for entry in WalkDir::new(&base)
            .into_iter()
            .filter_entry(|e| {
                !e.file_type().is_dir() || !skip_dir(&e.file_name().to_string_lossy())
            })
            .filter_map(Result::ok)
        {
            if entry.path().extension().and_then(|e| e.to_str()) != Some(ext) {
                continue;
            }
            if let Some(src) = read(entry.path()) {
                total += re.find_iter(&src).count();
            }
        }
    }
    total
}

fn check_test_counts(root: &Path, problems: &mut Vec<String>) {
    let rust_re = Regex::new(r"#\[(?:tokio::)?test\]").unwrap();
    let rust = count_matches(root, "rs", &rust_re, None) as i64;

    let py_re = Regex::new(r"(?m)^\s*(?:async\s+)?def test_\w+").unwrap();
    let pytest = count_matches(
        root,
        "py",
        &py_re,
        Some(&["sphereql-python/tests", "lingua-spherica/tests"]),
    ) as i64;

    println!("test counts: {rust} Rust (#[test]/#[tokio::test]), {pytest} pytest");

    let rust_floor = Regex::new(r"(\d+)\+\s+Rust tests").unwrap();
    let py_floor = Regex::new(r"(\d+)\+\s+(?:Python binding|pytest) tests").unwrap();

    for doc in ["README.md", "docs/project-status.md"] {
        let Some(txt) = read(&root.join(doc)) else {
            continue;
        };
        check_floor(doc, &txt, &rust_floor, rust, "Rust", problems);
        check_floor(doc, &txt, &py_floor, pytest, "pytest", problems);
    }
}

fn check_floor(
    doc: &str,
    txt: &str,
    re: &Regex,
    actual: i64,
    label: &str,
    problems: &mut Vec<String>,
) {
    for c in re.captures_iter(txt) {
        let floor: i64 = c[1].parse().unwrap_or(0);
        if floor > actual {
            problems.push(format!(
                "{doc}: states \"{floor}+ {label} tests\" but only {actual} exist — the floor overstates the count"
            ));
        } else if actual - floor > FLOOR_STALENESS_BUDGET {
            problems.push(format!(
                "{doc}: states \"{floor}+ {label} tests\" but {actual} exist — bump the floor (stale by {} > {FLOOR_STALENESS_BUDGET})",
                actual - floor
            ));
        }
    }
}

// ── Main ───────────────────────────────────────────────────────────────

fn main() {
    let root = workspace_root();
    let mut problems: Vec<String> = Vec::new();

    let root_toml: toml::Value = read(&root.join("Cargo.toml"))
        .expect("read root Cargo.toml")
        .parse()
        .expect("parse root Cargo.toml");
    let members: Vec<String> = root_toml
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    check_crate_tables(&root, &members, &mut problems);
    check_test_counts(&root, &mut problems);

    println!(
        "checked {} workspace members against the README + architecture crate tables",
        members.len()
    );

    if problems.is_empty() {
        println!("OK: crate tables and test-count floors agree with reality.");
        return;
    }

    eprintln!("\nDOC DRIFT DETECTED:\n");
    for p in &problems {
        eprintln!("  - {p}");
    }
    eprintln!("\nUpdate the crate tables / test-count floors to match, then re-run.\n");
    std::process::exit(1);
}
