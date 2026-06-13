//! Version drift check.
//!
//! Catches the failure mode where a release-version string in one doc,
//! README, or manifest gets bumped while another is forgotten. SphereQL
//! pins its version in two canonical places:
//!
//!   * the **Rust** version — `[workspace.package].version` in the root
//!     `Cargo.toml` (e.g. `0.2.0-alpha`), and
//!   * the **Python** version — `[project].version` in
//!     `sphereql-python/pyproject.toml` (e.g. `0.2.0a0`), which must be the
//!     PEP 440 spelling of the same release.
//!
//! Everything else must agree with whichever canonical its *token shape*
//! implies: a `-alpha`-style token is checked against the Rust canonical,
//! an `a0`-style token against the Python one, and a bare `x.y.z` against
//! the shared base. This token-shape rule means a single line can mix both
//! spellings correctly — e.g. "Pre-1.0 (`0.2.0a0`, tracking the
//! workspace's `0.2.0-alpha`)".
//!
//! # What is checked
//!
//! * **Manifests** (parsed, exact): every workspace member `Cargo.toml`
//!   path-dependency `version = "…"` pin, the two `pyproject.toml` files,
//!   and the `__version__` in any `__init__.py`.
//! * **Docs** (regex scan): every `*.md` token that appears in a
//!   self-version context — `version = "…"`, `` `…` ``, or `v…` — must
//!   equal the matching canonical.
//!
//! Files that legitimately carry historical or target versions
//! (`CHANGELOG.md`, `TODO.md`) and any one-off exceptions live in
//! `.version-drift-ignore.toml`.
//!
//! # Limitations
//!
//! The markdown scan is crate-name-blind: it flags any 3-part version
//! token in a `version = "…"`, backtick, or `v…` context. Today every
//! such token in the docs is sphereQL's own version, so this is exact.
//! If a doc ever pins a *third-party* crate to an `x.y.z` version in one
//! of those contexts (e.g. a `serde = { version = "1.0.200" }` snippet),
//! add an `[[allow]]` entry for it in `.version-drift-ignore.toml`.
//! Real manifests are unaffected — [`check_manifest_pins`] only inspects
//! dependencies that carry a `path` key.
//!
//! # Usage
//!
//! ```sh
//! cargo run -p check-versions
//! ```
//!
//! Exits 0 when every version string agrees, 1 on drift.

use std::path::{Path, PathBuf};

use regex::Regex;
use serde::Deserialize;
use walkdir::WalkDir;

// ── Ignore config ──────────────────────────────────────────────────────

#[derive(Debug, Default, Deserialize)]
struct IgnoreConfig {
    /// Files excluded from the markdown token scan entirely (matched by
    /// repo-relative path or bare file name).
    #[serde(default)]
    ignore_files: Vec<String>,
    /// One-off (file, token) pairs allowed to differ from the canonical.
    #[serde(default)]
    allow: Vec<AllowEntry>,
}

#[derive(Debug, Deserialize)]
struct AllowEntry {
    file: String,
    token: String,
    #[allow(dead_code)] // surfaced only for human review via the TOML source
    reason: String,
}

impl IgnoreConfig {
    fn load(path: &Path) -> IgnoreConfig {
        if !path.exists() {
            return IgnoreConfig::default();
        }
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        toml::from_str(&raw).unwrap_or_else(|e| panic!("invalid TOML at {}: {e}", path.display()))
    }

    fn is_ignored(&self, rel: &str, file_name: &str) -> bool {
        self.ignore_files.iter().any(|f| f == rel || f == file_name)
    }

    fn allows(&self, rel: &str, token: &str) -> bool {
        self.allow
            .iter()
            .any(|a| (a.file == rel || a.file == file_name_of(rel)) && a.token == token)
    }
}

fn file_name_of(rel: &str) -> &str {
    rel.rsplit('/').next().unwrap_or(rel)
}

// ── Version token parsing ──────────────────────────────────────────────

/// Split `x.y.z[suffix]` into the base `x.y.z` and the optional
/// pre-release suffix (with a leading `-` already stripped).
fn split_version(v: &str) -> (String, Option<String>) {
    // Base is the first three dotted numbers; whatever trails is pre-release.
    let re = Regex::new(r"^(\d+\.\d+\.\d+)(.*)$").unwrap();
    match re.captures(v) {
        Some(c) => {
            let base = c[1].to_string();
            let rest = c[2].trim_start_matches('-').to_string();
            (base, if rest.is_empty() { None } else { Some(rest) })
        }
        None => (v.to_string(), None),
    }
}

/// Normalize a pre-release token to its PEP 440 / semver kind: `a`, `b`,
/// `rc`, or `?` for anything unrecognized.
fn pre_kind(pre: &str) -> &'static str {
    let p = pre.to_ascii_lowercase();
    if p.starts_with("rc") {
        "rc"
    } else if p.starts_with("alpha") || p.starts_with('a') {
        "a"
    } else if p.starts_with("beta") || p.starts_with('b') {
        "b"
    } else {
        "?"
    }
}

/// A found version token, with the file + line it came from.
struct Found {
    token: String,
    rel: String,
    line: usize,
}

/// Check one token against the canonical pair, classifying it by shape:
/// a token containing `-` is Rust-flavored, a bare-but-suffixed token
/// (`0.2.0a0`) is Python-flavored, and a plain `x.y.z` is checked on its
/// base only (the two canonicals share a base).
fn check_token(token: &str, rust_canon: &str, py_canon: &str) -> Result<(), String> {
    let (base, pre) = split_version(token);
    match pre {
        None => {
            let canon_base = split_version(rust_canon).0;
            if base == canon_base {
                Ok(())
            } else {
                Err(format!("base does not match canonical `{rust_canon}`"))
            }
        }
        Some(_) if token.contains('-') => {
            if token == rust_canon {
                Ok(())
            } else {
                Err(format!("expected Rust canonical `{rust_canon}`"))
            }
        }
        Some(_) => {
            if token == py_canon {
                Ok(())
            } else {
                Err(format!("expected Python canonical `{py_canon}`"))
            }
        }
    }
}

// ── TOML helpers ───────────────────────────────────────────────────────

fn toml_str<'a>(v: &'a toml::Value, path: &[&str]) -> Option<&'a str> {
    let mut cur = v;
    for key in path {
        cur = cur.get(key)?;
    }
    cur.as_str()
}

fn read(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

// ── Manifest checks (exact) ────────────────────────────────────────────

/// Every dependency that carries a `path` *and* an explicit `version`
/// pin must pin the Rust canonical. These intra-workspace pins are the
/// classic silent-drift site during a version bump.
fn check_manifest_pins(manifest: &Path, rel: &str, rust_canon: &str, problems: &mut Vec<String>) {
    let Some(txt) = read(manifest) else { return };
    let val: toml::Value = match txt.parse() {
        Ok(v) => v,
        Err(e) => {
            problems.push(format!("{rel}: failed to parse TOML: {e}"));
            return;
        }
    };

    for table in ["dependencies", "dev-dependencies", "build-dependencies"] {
        let Some(deps) = val.get(table).and_then(|v| v.as_table()) else {
            continue;
        };
        for (name, spec) in deps {
            let Some(t) = spec.as_table() else { continue };
            if t.get("path").is_none() {
                continue;
            }
            if let Some(ver) = t.get("version").and_then(|v| v.as_str())
                && ver != rust_canon
            {
                problems.push(format!(
                    "{rel}: dependency `{name}` pins version \"{ver}\", expected \"{rust_canon}\""
                ));
            }
        }
    }
}

// ── Markdown scan ──────────────────────────────────────────────────────

fn skip_dir(name: &str) -> bool {
    matches!(
        name,
        "target" | ".git" | ".venv" | ".claude" | "node_modules" | "__pycache__" | ".pytest_cache"
    )
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn scan_markdown(
    root: &Path,
    rust_canon: &str,
    py_canon: &str,
    ignore: &IgnoreConfig,
) -> Vec<Found> {
    // A release-version token: three dotted numbers plus an optional
    // semver (`-alpha`) or PEP 440 (`a0`) pre-release suffix. Two-part
    // numbers (`pyo3 0.28`) and bare integers never match.
    let tok = r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+|(?:a|b|rc)\d+)?";
    let res = [
        Regex::new(&format!(r#"version\s*=\s*"({tok})""#)).unwrap(),
        Regex::new(&format!(r"`({tok})`")).unwrap(),
        Regex::new(&format!(r"\bv({tok})\b")).unwrap(),
    ];

    let mut found = Vec::new();
    let walker = WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !e.file_type().is_dir() || !skip_dir(&e.file_name().to_string_lossy()));

    for entry in walker.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let rel = relative(root, path);
        if ignore.is_ignored(&rel, file_name_of(&rel)) {
            continue;
        }
        let Some(txt) = read(path) else { continue };
        for (i, line) in txt.lines().enumerate() {
            for re in &res {
                for caps in re.captures_iter(line) {
                    let token = caps[1].to_string();
                    if check_token(&token, rust_canon, py_canon).is_err()
                        && !ignore.allows(&rel, &token)
                    {
                        found.push(Found {
                            token,
                            rel: rel.clone(),
                            line: i + 1,
                        });
                    }
                }
            }
        }
    }
    found
}

// ── Main ───────────────────────────────────────────────────────────────

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .expect("failed to resolve workspace root")
}

fn main() {
    let root = workspace_root();
    let mut problems: Vec<String> = Vec::new();

    // ── Canonical sources ──
    let root_manifest = read(&root.join("Cargo.toml")).expect("read root Cargo.toml");
    let root_toml: toml::Value = root_manifest.parse().expect("parse root Cargo.toml");
    let rust_canon = toml_str(&root_toml, &["workspace", "package", "version"])
        .expect("[workspace.package].version missing in root Cargo.toml")
        .to_string();

    let pyproject = read(&root.join("sphereql-python/pyproject.toml"))
        .expect("read sphereql-python/pyproject.toml");
    let py_toml: toml::Value = pyproject.parse().expect("parse pyproject.toml");
    let py_canon = toml_str(&py_toml, &["project", "version"])
        .expect("[project].version missing in pyproject.toml")
        .to_string();

    // ── Canonical consistency (Rust ↔ PEP 440 Python) ──
    let (rb, rpre) = split_version(&rust_canon);
    let (pb, ppre) = split_version(&py_canon);
    if rb != pb {
        problems.push(format!(
            "canonical mismatch: Rust `{rust_canon}` and Python `{py_canon}` have different base versions"
        ));
    }
    match (rpre.as_deref(), ppre.as_deref()) {
        (None, None) => {}
        (Some(r), Some(p)) if pre_kind(r) == pre_kind(p) => {}
        _ => problems.push(format!(
            "canonical mismatch: Rust `{rust_canon}` and Python `{py_canon}` have incompatible pre-release kinds"
        )),
    }

    // ── Manifest pins across every workspace member ──
    let mut manifests: Vec<PathBuf> = Vec::new();
    if let Some(members) = root_toml
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
    {
        for m in members {
            if let Some(dir) = m.as_str() {
                manifests.push(root.join(dir).join("Cargo.toml"));
            }
        }
    }
    for manifest in &manifests {
        let rel = relative(&root, manifest);
        check_manifest_pins(manifest, &rel, &rust_canon, &mut problems);
    }

    // ── Python-side exact checks ──
    // pyproject already read for the canonical; check lingua-spherica's
    // pyproject + every __init__.py `__version__` against the Python canonical.
    if let Some(lingua) = read(&root.join("lingua-spherica/pyproject.toml"))
        && let Ok(v) = lingua.parse::<toml::Value>()
        && let Some(ver) = toml_str(&v, &["project", "version"])
        && ver != py_canon
    {
        problems.push(format!(
            "lingua-spherica/pyproject.toml: version \"{ver}\", expected Python canonical \"{py_canon}\""
        ));
    }
    let dunder = Regex::new(r#"__version__\s*=\s*"([^"]+)""#).unwrap();
    let init_walker = WalkDir::new(&root)
        .into_iter()
        .filter_entry(|e| !e.file_type().is_dir() || !skip_dir(&e.file_name().to_string_lossy()));
    for entry in init_walker.filter_map(Result::ok) {
        if entry.file_name() != "__init__.py" {
            continue;
        }
        let Some(txt) = read(entry.path()) else {
            continue;
        };
        if let Some(c) = dunder.captures(&txt) {
            let ver = &c[1];
            if ver != py_canon {
                problems.push(format!(
                    "{}: __version__ = \"{ver}\", expected Python canonical \"{py_canon}\"",
                    relative(&root, entry.path())
                ));
            }
        }
    }

    // ── Markdown doc scan ──
    let ignore = IgnoreConfig::load(&root.join(".version-drift-ignore.toml"));
    let doc_problems = scan_markdown(&root, &rust_canon, &py_canon, &ignore);
    for f in &doc_problems {
        let reason = check_token(&f.token, &rust_canon, &py_canon).unwrap_err();
        problems.push(format!("{}:{}: `{}` — {reason}", f.rel, f.line, f.token));
    }

    println!(
        "canonical: Rust `{rust_canon}` / Python `{py_canon}` · checked {} manifests + all *.md + __init__.py",
        manifests.len(),
    );

    if problems.is_empty() {
        println!("OK: every release-version string agrees with the canonical.");
        return;
    }

    eprintln!("\nVERSION DRIFT DETECTED — the following strings disagree with the canonical:\n");
    for p in &problems {
        eprintln!("  - {p}");
    }
    eprintln!(
        "\nBump every site to the canonical, or (for a deliberate historical/target\nreference) add it to `.version-drift-ignore.toml`.\n"
    );
    std::process::exit(1);
}
