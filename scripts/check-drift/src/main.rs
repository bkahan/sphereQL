//! Bindings drift check.
//!
//! Catches the common failure mode where a new public function or type
//! lands in `sphereql-embed` (or `sphereql-layout`) without a matching
//! binding in `sphereql-python` or `sphereql-wasm`. Walks each crate's
//! source with `syn`, collects the sets of exposed names, and fails
//! when an embed/layout public item isn't surfaced in either binding
//! and isn't on the allowlist.
//!
//! # Why not enforce parity for every item?
//!
//! Some embed items are intentionally Rust-only (trait objects that
//! don't map to Python/JS, internal helpers, generic functions). Those
//! live in `.bindings-ignore.toml` with a reason field so the review
//! trail for "why isn't this bound?" is preserved.
//!
//! # Usage
//!
//! ```sh
//! cargo run -p check-drift
//! ```
//!
//! Exits 0 when clean, 1 when drift is detected.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use syn::{Attribute, Item, Meta};
use walkdir::WalkDir;

// ── Allowlist ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Allowlist {
    #[serde(default)]
    exempt: Vec<ExemptEntry>,
}

#[derive(Debug, Deserialize)]
struct ExemptEntry {
    name: String,
    #[allow(dead_code)] // surfaced only for human review via the TOML source
    reason: String,
}

impl Allowlist {
    fn load(path: &Path) -> Allowlist {
        if !path.exists() {
            return Allowlist { exempt: vec![] };
        }
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        toml::from_str(&raw).unwrap_or_else(|e| panic!("invalid TOML at {}: {e}", path.display()))
    }

    fn exempts(&self) -> BTreeSet<&str> {
        self.exempt.iter().map(|e| e.name.as_str()).collect()
    }
}

// ── Name extraction ────────────────────────────────────────────────────

/// Public top-level names in a crate, from `pub fn / pub struct / pub enum / pub type`.
///
/// Impl blocks and items behind `#[cfg(…)]` are deliberately *not* walked —
/// the drift check is about crate-level API surface, not method churn. Nested
/// modules inside `mod {}` blocks are visited.
fn collect_embed_names(root: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let src = match std::fs::read_to_string(entry.path()) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let Ok(file) = syn::parse_file(&src) else {
            continue;
        };
        visit_items(&file.items, &mut names);
    }
    names
}

fn visit_items(items: &[Item], out: &mut BTreeSet<String>) {
    for item in items {
        match item {
            Item::Fn(f) if matches!(f.vis, syn::Visibility::Public(_)) => {
                out.insert(f.sig.ident.to_string());
            }
            Item::Struct(s) if matches!(s.vis, syn::Visibility::Public(_)) => {
                out.insert(s.ident.to_string());
            }
            Item::Enum(e) if matches!(e.vis, syn::Visibility::Public(_)) => {
                out.insert(e.ident.to_string());
            }
            Item::Type(t) if matches!(t.vis, syn::Visibility::Public(_)) => {
                out.insert(t.ident.to_string());
            }
            Item::Mod(m) => {
                if let Some((_, items)) = &m.content {
                    visit_items(items, out);
                }
            }
            _ => {}
        }
    }
}

// ── Binding-side name extraction ───────────────────────────────────────

/// Names exposed by `sphereql-python`: any struct with `#[pyclass]` (using
/// either the explicit `name = "..."` or the Rust struct name stripped of
/// a `Py` prefix) and any `fn` with `#[pyfunction]` (using its
/// `#[pyo3(name = "...")]` override or Rust name with a `py_` prefix
/// stripped).
fn collect_python_names(root: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let src = match std::fs::read_to_string(entry.path()) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let Ok(file) = syn::parse_file(&src) else {
            continue;
        };
        visit_python_items(&file.items, &mut names);
    }
    names
}

fn visit_python_items(items: &[Item], out: &mut BTreeSet<String>) {
    for item in items {
        match item {
            Item::Struct(s) if has_attr(&s.attrs, "pyclass") => {
                let nm = attr_name_override(&s.attrs, "pyclass")
                    .unwrap_or_else(|| strip_py_prefix(&s.ident.to_string()));
                out.insert(nm);
            }
            Item::Fn(f) if has_attr(&f.attrs, "pyfunction") => {
                let nm = attr_name_override(&f.attrs, "pyo3")
                    .unwrap_or_else(|| strip_py_prefix(&f.sig.ident.to_string()));
                out.insert(nm);
            }
            Item::Mod(m) => {
                if let Some((_, items)) = &m.content {
                    visit_python_items(items, out);
                }
            }
            _ => {}
        }
    }
}

/// Names exposed by `sphereql-wasm`: any `#[wasm_bindgen]`-annotated
/// struct or standalone fn, using the explicit `js_name` / `js_class`
/// override when present, otherwise the Rust identifier.
fn collect_wasm_names(root: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let src = match std::fs::read_to_string(entry.path()) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let Ok(file) = syn::parse_file(&src) else {
            continue;
        };
        visit_wasm_items(&file.items, &mut names);
    }
    names
}

fn visit_wasm_items(items: &[Item], out: &mut BTreeSet<String>) {
    for item in items {
        match item {
            Item::Struct(s) => {
                if has_attr(&s.attrs, "wasm_bindgen") {
                    let nm = attr_name_override(&s.attrs, "wasm_bindgen")
                        .unwrap_or_else(|| strip_wasm_prefix(&s.ident.to_string()));
                    out.insert(nm);
                }
                // Tsify-derived structs cross the wasm boundary too,
                // even without an explicit `#[wasm_bindgen]` annotation.
                if has_derive(&s.attrs, "Tsify") {
                    out.insert(strip_wasm_prefix(&s.ident.to_string()));
                }
            }
            Item::Fn(f) if has_attr(&f.attrs, "wasm_bindgen") => {
                let nm = attr_name_override(&f.attrs, "wasm_bindgen")
                    .unwrap_or_else(|| f.sig.ident.to_string());
                out.insert(nm);
            }
            Item::Mod(m) => {
                if let Some((_, items)) = &m.content {
                    visit_wasm_items(items, out);
                }
            }
            _ => {}
        }
    }
}

fn has_derive(attrs: &[Attribute], trait_name: &str) -> bool {
    for attr in attrs {
        if !attr_path_matches(attr, "derive") {
            continue;
        }
        if let Meta::List(list) = &attr.meta {
            let tokens = list.tokens.to_string();
            if tokens
                .split(',')
                .any(|part| part.trim().split("::").last() == Some(trait_name))
            {
                return true;
            }
        }
    }
    false
}

// ── Attribute helpers ─────────────────────────────────────────────────

fn has_attr(attrs: &[Attribute], name: &str) -> bool {
    attrs.iter().any(|a| attr_path_matches(a, name))
}

fn attr_path_matches(attr: &Attribute, name: &str) -> bool {
    attr.path()
        .segments
        .last()
        .is_some_and(|seg| seg.ident == name)
}

/// Pull a `name = "X"`, `js_name = X`, or `js_class = X` override out of an
/// attribute. Best-effort string scraping — `syn`'s `Meta::List` parse path
/// doesn't like our mix of attribute styles across pyo3 versions.
fn attr_name_override(attrs: &[Attribute], attr_name: &str) -> Option<String> {
    for a in attrs {
        if !attr_path_matches(a, attr_name) {
            continue;
        }
        if let Meta::List(list) = &a.meta {
            let tokens = list.tokens.to_string();
            // Look for `name = "X"`, `js_name = X`, `js_class = X`
            for key in &["name", "js_name", "js_class"] {
                if let Some(val) = extract_assign(&tokens, key) {
                    return Some(val);
                }
            }
        }
    }
    None
}

/// Find `key = "value"` or `key = bareword` in a token string. Whitespace-
/// tolerant; ignores trailing commas and surrounding quotes.
fn extract_assign(tokens: &str, key: &str) -> Option<String> {
    let needle = format!("{key} =");
    let idx = tokens.find(&needle)?;
    let rest = &tokens[idx + needle.len()..].trim_start();
    // Value can be quoted or a bareword; terminate at `,` or `)` or whitespace.
    let (quoted, body) = if let Some(stripped) = rest.strip_prefix('"') {
        (true, stripped)
    } else {
        (false, *rest)
    };
    let terminator = if quoted { '"' } else { ',' };
    let end = body.find(terminator).unwrap_or(body.len());
    let val = body[..end].trim().to_string();
    if val.is_empty() { None } else { Some(val) }
}

fn strip_py_prefix(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("Py")
        && rest.chars().next().is_some_and(|c| c.is_ascii_uppercase())
    {
        return rest.to_string();
    }
    name.strip_prefix("py_").unwrap_or(name).to_string()
}

fn strip_wasm_prefix(name: &str) -> String {
    name.strip_prefix("Wasm").unwrap_or(name).to_string()
}

/// Does the embed-side `name` have any binding? Tries the original name
/// plus a handful of common transformations — binding types often trim
/// `Result` / `Summary` suffixes or wrap with `Out`, so a direct equality
/// check would over-flag. Matches case-insensitively.
fn is_bound(name: &str, bound: &BTreeSet<String>, bound_lower: &BTreeSet<String>) -> bool {
    let aliases = name_aliases(name);
    aliases
        .iter()
        .any(|alias| bound.contains(alias) || bound_lower.contains(&alias.to_ascii_lowercase()))
}

fn name_aliases(name: &str) -> Vec<String> {
    let mut out = vec![name.to_string()];
    // WASM wrappers often append `Out`.
    out.push(format!("{name}Out"));
    // Python types often append `Info` (`CategorySummaryInfo`) and
    // WASM wrappers append `Out` — cover both.
    out.push(format!("{name}Info"));
    // Trim common noise suffixes that binding types drop.
    for suffix in ["Result", "Summary", "Info", "Report"] {
        if let Some(base) = name.strip_suffix(suffix)
            && !base.is_empty()
        {
            out.push(base.to_string());
            out.push(format!("{base}Out"));
            out.push(format!("{base}Hit"));
            out.push(format!("{base}Info"));
        }
    }
    // Pipeline-prefixed names — `PipelinePathStep` → `PathStep`.
    if let Some(rest) = name.strip_prefix("Pipeline")
        && !rest.is_empty()
    {
        out.push(rest.to_string());
        out.push(format!("{rest}Out"));
    }
    out
}

// ── Main ───────────────────────────────────────────────────────────────

fn workspace_root() -> PathBuf {
    // The bin runs under Cargo, so CARGO_MANIFEST_DIR is
    // `.../scripts/check-drift`. Pop two to reach the workspace root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .expect("failed to resolve workspace root")
}

fn main() {
    let root = workspace_root();
    let allowlist_path = root.join(".bindings-ignore.toml");
    let allowlist = Allowlist::load(&allowlist_path);
    let exempts = allowlist.exempts();

    let mut embed_names = collect_embed_names(&root.join("sphereql-embed/src"));
    embed_names.extend(collect_embed_names(&root.join("sphereql-layout/src")));

    let python_names = collect_python_names(&root.join("sphereql-python/src"));
    let wasm_names = collect_wasm_names(&root.join("sphereql-wasm/src"));

    let mut bound: BTreeSet<String> = python_names.clone();
    bound.extend(wasm_names.iter().cloned());
    let bound_lower: BTreeSet<String> = bound.iter().map(|s| s.to_ascii_lowercase()).collect();

    let missing: Vec<String> = embed_names
        .iter()
        .filter(|n| !exempts.contains(n.as_str()))
        .filter(|n| !is_bound(n, &bound, &bound_lower))
        .cloned()
        .collect();

    println!(
        "embed/layout pub items: {}, python bindings: {}, wasm bindings: {}, allowlisted: {}",
        embed_names.len(),
        python_names.len(),
        wasm_names.len(),
        exempts.len(),
    );

    if missing.is_empty() {
        println!(
            "OK: every non-allowlisted public item in sphereql-embed / sphereql-layout has a binding."
        );
        return;
    }

    eprintln!("\nDRIFT DETECTED — the following public items are not bound and not allowlisted:\n");
    for name in &missing {
        eprintln!("  - {name}");
    }
    eprintln!(
        "\nFor each item:\n  1. Add a binding in sphereql-python or sphereql-wasm, OR\n  2. Add an entry to `.bindings-ignore.toml` with a reason.\n"
    );
    std::process::exit(1);
}
