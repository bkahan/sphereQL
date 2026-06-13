//! Compile-check the Rust snippets embedded in the prose docs.
//!
//! `cargo test --doc` only covers doc-comments *inside* the crates. The
//! hand-written guides under `docs/` (plus the root `README.md`) carry
//! their own ```` ```rust ```` blocks that nothing compiles — so when an
//! API changes underneath them they rot silently. The drift that motivated
//! this check: a quickstart showing `let p = Projection::fit(...)` after
//! `fit` had started returning `Result`.
//!
//! This binary walks the docs, pulls every fenced Rust block, and
//! type-checks it (no codegen, no run) against the freshly built
//! `sphereql` umbrella crate at `--features full`.
//!
//! # Fence conventions
//!
//! | info string        | behaviour                                  |
//! |--------------------|--------------------------------------------|
//! | ```` ```rust ````        | compile-checked (must type-check)    |
//! | ```` ```rust,no_run ```` | compile-checked (we never run anyway) |
//! | ```` ```rust,ignore ```` | skipped — illustrative / pseudo-code  |
//!
//! Any other info string (`text`, `bash`, `toml`, `python`, …) is ignored.
//!
//! A bare-statement snippet (no top-level `fn main`/items) is wrapped in a
//! `fn main() { … }` before checking, matching how `rustdoc` treats doc
//! examples — so quickstart snippets can read as a sequence of statements.
//!
//! # Usage
//!
//! ```sh
//! cargo run -p check-doc-snippets
//! ```
//!
//! Exits 0 when every non-ignored snippet type-checks, 1 otherwise.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

/// Docs (relative to the workspace root) scanned for ```` ```rust ```` blocks.
/// `docs/*.md` is globbed; these two prose entrypoints live at the root.
const ROOT_DOCS: &[&str] = &["README.md", "CONTRIBUTING.md"];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .expect("failed to resolve workspace root")
}

// ── Snippet extraction ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
    /// `rust` / `rust,no_run` — must type-check.
    Check,
    /// `rust,ignore` — illustrative; counted but not compiled.
    Ignore,
}

struct Snippet {
    file: PathBuf,
    /// 1-based line of the opening fence in the source doc.
    fence_line: usize,
    disposition: Disposition,
    code: String,
}

/// Classify a fence info string. Returns `None` for non-Rust fences.
///
/// Only fences whose first comma-separated token is exactly `rust` count.
/// `rust,ignore` -> skip; `rust,no_run` (and any other `rust,*`) -> check.
fn classify(info: &str) -> Option<Disposition> {
    let info = info.trim();
    let mut tokens = info.split(',').map(str::trim);
    if tokens.next() != Some("rust") {
        return None;
    }
    let rest: Vec<&str> = tokens.collect();
    if rest.contains(&"ignore") {
        Some(Disposition::Ignore)
    } else {
        Some(Disposition::Check)
    }
}

/// Pull every fenced Rust block out of one markdown document.
///
/// Handles both ``` and ~~~ fences and respects the CommonMark rule that a
/// closing fence must be at least as long as, and the same character as,
/// the opening one — so a ``` inside a ~~~~ block doesn't end it.
fn extract(file: &Path, text: &str) -> Vec<Snippet> {
    let mut out = Vec::new();
    let mut lines = text.lines().enumerate();

    while let Some((idx, line)) = lines.next() {
        let trimmed = line.trim_start();
        let fence_char = match trimmed.chars().next() {
            Some(c @ ('`' | '~')) => c,
            _ => continue,
        };
        let fence_len = trimmed.chars().take_while(|&c| c == fence_char).count();
        if fence_len < 3 {
            continue;
        }
        let info = &trimmed[fence_len..];
        let disposition = classify(info);

        // Consume the body up to the matching closing fence regardless of
        // whether we care about this block, so an ignored/non-rust block
        // can't swallow a later rust block.
        let mut body = String::new();
        let mut closed = false;
        for (_, inner) in lines.by_ref() {
            let it = inner.trim_start();
            if it.starts_with(fence_char) {
                let close_len = it.chars().take_while(|&c| c == fence_char).count();
                if close_len >= fence_len && it[close_len..].trim().is_empty() {
                    closed = true;
                    break;
                }
            }
            body.push_str(inner);
            body.push('\n');
        }
        let _ = closed; // EOF-terminated block is fine; body is what we have.

        if let Some(disposition) = disposition {
            out.push(Snippet {
                file: file.to_path_buf(),
                fence_line: idx + 1,
                disposition,
                code: body,
            });
        }
    }
    out
}

/// `docs/*.md` (non-recursive, mirroring the prose-guide layer) plus the
/// root prose docs. Sub-crate READMEs and `docs/.prompts/` are out of
/// scope: the former track per-crate APIs, the latter are LLM prompts.
fn doc_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root.join("docs")) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md") {
                files.push(path);
            }
        }
    }
    for name in ROOT_DOCS {
        let path = root.join(name);
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    files
}

// ── Building the dependency surface ────────────────────────────────────

#[derive(Deserialize)]
struct CargoMessage {
    reason: String,
    #[serde(default)]
    target: Option<CargoTarget>,
    #[serde(default)]
    filenames: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct CargoTarget {
    name: String,
    kind: Vec<String>,
}

/// Where the snippets `--extern`/`-L` against: the freshly built
/// `libsphereql.rlib` and the `deps/` directory holding its transitive
/// rlibs.
struct DepSurface {
    sphereql_rlib: PathBuf,
    deps_dir: PathBuf,
}

/// `cargo build -p sphereql --features full`, parsing the JSON artifact
/// stream to locate the `sphereql` rlib. Cargo writes the top-level lib to
/// `target/<profile>/libsphereql.rlib` and every transitive dependency's
/// rlib to the sibling `target/<profile>/deps/` — that `deps/` dir is what
/// snippets need on the `-L dependency=` search path.
fn build_dep_surface(root: &Path) -> Result<DepSurface, String> {
    let output = Command::new("cargo")
        .current_dir(root)
        .args([
            "build",
            "-p",
            "sphereql",
            "--features",
            "full",
            "--message-format=json",
        ])
        .output()
        .map_err(|e| format!("failed to spawn cargo build: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "`cargo build -p sphereql --features full` failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut rlib: Option<PathBuf> = None;
    for line in stdout.lines() {
        let Ok(msg) = serde_json::from_str::<CargoMessage>(line) else {
            continue;
        };
        if msg.reason != "compiler-artifact" {
            continue;
        }
        let Some(target) = msg.target else { continue };
        if target.name != "sphereql" || !target.kind.iter().any(|k| k == "lib") {
            continue;
        }
        if let Some(files) = msg.filenames
            && let Some(found) = files.into_iter().find(|f| f.ends_with(".rlib"))
        {
            rlib = Some(PathBuf::from(found));
        }
    }

    let sphereql_rlib =
        rlib.ok_or_else(|| "cargo build produced no sphereql .rlib artifact".to_string())?;
    let deps_dir = sphereql_rlib
        .parent()
        .map(|p| p.join("deps"))
        .ok_or_else(|| "sphereql rlib has no parent directory".to_string())?;

    Ok(DepSurface {
        sphereql_rlib,
        deps_dir,
    })
}

// ── Snippet compilation ────────────────────────────────────────────────

/// A snippet is a "full program" if it already declares top-level items
/// (`fn main`, other `fn`, `struct`, `use`, `impl`, …). Bare statement
/// snippets get wrapped in `fn main`, matching rustdoc's behaviour. We only
/// need to detect the *unambiguous full-program* markers; everything else
/// is safe to wrap because the wrapper is a valid statement context.
fn needs_main_wrapper(code: &str) -> bool {
    !code.lines().any(|l| {
        let l = l.trim_start();
        l.starts_with("fn main")
    })
}

fn wrap(code: &str) -> String {
    if needs_main_wrapper(code) {
        // `#[allow(...)]` keeps snippet ergonomics (unused imports/vars are
        // normal in illustrative-but-real examples) from failing the check
        // on warnings; we only care about hard type/API errors.
        format!("#![allow(unused, dead_code)]\nfn main() {{\n{code}\n}}\n",)
    } else {
        format!("#![allow(unused, dead_code)]\n{code}\n")
    }
}

struct Outcome {
    snippet_idx: usize,
    passed: bool,
    stderr: String,
}

fn check_snippet(idx: usize, snippet: &Snippet, surface: &DepSurface, tmp_dir: &Path) -> Outcome {
    let src_path = tmp_dir.join(format!("snippet_{idx}.rs"));
    let out_path = tmp_dir.join(format!("snippet_{idx}.meta"));
    let wrapped = wrap(&snippet.code);

    if let Err(e) = std::fs::write(&src_path, &wrapped) {
        return Outcome {
            snippet_idx: idx,
            passed: false,
            stderr: format!("failed to write temp snippet: {e}"),
        };
    }

    let output = Command::new("rustc")
        .args(["--edition", "2024", "--crate-type", "bin"])
        .arg("--extern")
        .arg(format!(
            "sphereql={}",
            surface.sphereql_rlib.to_string_lossy()
        ))
        .arg("-L")
        .arg(format!("dependency={}", surface.deps_dir.to_string_lossy()))
        .arg("--emit=metadata")
        .arg("-o")
        .arg(&out_path)
        .arg(&src_path)
        .output();

    match output {
        Ok(out) => Outcome {
            snippet_idx: idx,
            passed: out.status.success(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        },
        Err(e) => Outcome {
            snippet_idx: idx,
            passed: false,
            stderr: format!("failed to invoke rustc: {e}"),
        },
    }
}

// ── Main ───────────────────────────────────────────────────────────────

fn rel(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn main() {
    let root = workspace_root();

    let mut snippets: Vec<Snippet> = Vec::new();
    for file in doc_files(&root) {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        snippets.extend(extract(&file, &text));
    }

    let to_check: Vec<usize> = snippets
        .iter()
        .enumerate()
        .filter(|(_, s)| s.disposition == Disposition::Check)
        .map(|(i, _)| i)
        .collect();
    let ignored = snippets.len() - to_check.len();

    println!(
        "found {} rust block(s) across the prose docs: {} to compile-check, {} marked ignore",
        snippets.len(),
        to_check.len(),
        ignored,
    );

    if to_check.is_empty() {
        println!("nothing to compile-check.");
        return;
    }

    println!("building `sphereql --features full` to type-check against…");
    let surface = match build_dep_surface(&root) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("\nERROR: {e}");
            std::process::exit(1);
        }
    };

    let tmp_dir = std::env::temp_dir().join("sphereql-doc-snippets");
    if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
        eprintln!(
            "\nERROR: failed to create temp dir {}: {e}",
            tmp_dir.display()
        );
        std::process::exit(1);
    }

    let mut failures: Vec<Outcome> = Vec::new();
    for &idx in &to_check {
        let snippet = &snippets[idx];
        let outcome = check_snippet(idx, snippet, &surface, &tmp_dir);
        let where_ = format!("{}:{}", rel(&root, &snippet.file), snippet.fence_line);
        if outcome.passed {
            println!("  ok   {where_}");
        } else {
            println!("  FAIL {where_}");
            failures.push(outcome);
        }
    }

    if failures.is_empty() {
        println!(
            "\nOK: all {} compile-checked snippet(s) type-check against sphereql --features full.",
            to_check.len()
        );
        return;
    }

    eprintln!(
        "\nDOC SNIPPET DRIFT DETECTED — {} snippet(s) failed:\n",
        failures.len()
    );
    for f in &failures {
        let s = &snippets[f.snippet_idx];
        eprintln!("── {}:{} ──", rel(&root, &s.file), s.fence_line);
        eprintln!("{}", f.stderr.trim_end());
        eprintln!();
    }
    eprintln!(
        "Fix the API drift, or — if the block is illustrative/pseudo-code — \n\
         tag its fence ```rust,ignore so it's documented as non-compiling.\n"
    );
    std::process::exit(1);
}
