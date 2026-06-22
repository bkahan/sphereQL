//! `sphereql-vis-server` binary: load a corpus, build the indexes, and serve
//! the streaming-viewer API.
//!
//! ```sh
//! # API + auto-detected WASM studio front-end (one-liner):
//! cargo run -p sphereql-vis-server -- --corpus stress --open
//!
//! # API only (no studio found / not built):
//! cargo run -p sphereql-vis-server -- --corpus stress --addr 127.0.0.1:8080
//!
//! # Generate a standalone offline viewer and open it:
//! cargo run -p sphereql-vis-server -- --corpus stress --emit-html --open
//! ```

use std::process::ExitCode;
use std::sync::{Arc, RwLock};

use sphereql_embed::ProjectionKind;
use sphereql_vis::Scene;
use sphereql_vis_server::{AppState, StudioAssets, build_router, parse_corpus, parse_projection};

struct Args {
    corpus: String,
    addr: String,
    projection: ProjectionKind,
    emit_html: Option<String>,
    open_browser: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            corpus: "stress".to_string(),
            addr: "127.0.0.1:8080".to_string(),
            projection: ProjectionKind::Pca,
            emit_html: None,
            open_browser: false,
        }
    }
}

/// Minimal `--flag [value]` parser — avoids a clap dependency for a handful of options.
/// Returns `Err` with a usage hint on an unknown flag or missing value.
fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let flag = &argv[i];
        match flag.as_str() {
            "--corpus" | "-c" => {
                i += 1;
                args.corpus = argv.get(i).ok_or("--corpus needs a value")?.clone();
            }
            "--addr" | "-a" => {
                i += 1;
                args.addr = argv.get(i).ok_or("--addr needs a value")?.clone();
            }
            "--projection" | "-p" => {
                i += 1;
                let v = argv.get(i).ok_or("--projection needs a value")?;
                args.projection =
                    parse_projection(v).ok_or_else(|| format!("unknown projection '{v}'"))?;
            }
            "--emit-html" | "-e" => {
                // Path is optional: if the next token is absent or starts with
                // '-' (another flag), fall back to "sphere_viz.html".
                let next = argv.get(i + 1);
                if let Some(v) = next
                    && !v.starts_with('-')
                {
                    i += 1;
                    args.emit_html = Some(v.clone());
                } else {
                    args.emit_html = Some("sphere_viz.html".to_string());
                }
            }
            "--open" | "-o" => {
                args.open_browser = true;
            }
            "--help" | "-h" => return Err("help".to_string()),
            other => return Err(format!("unknown argument '{other}'")),
        }
        i += 1;
    }
    Ok(args)
}

const USAGE: &str = "\
sphereql-vis-server — out-of-core query server for the streaming viewer

USAGE:
    sphereql-vis-server [--corpus <name|path>] [--addr <host:port>] [--projection <kind>]
                        [--emit-html [path]] [--open]

OPTIONS:
    -c, --corpus      Corpus name (stress, hand_crafted, full, dbpedia_500k, …) or
                      a path to a Parquet file [default: stress]
    -a, --addr        Address to bind [default: 127.0.0.1:8080]
    -p, --projection  pca | umap_sphere | laplacian | kernel_pca [default: pca]
                      (O(n²) families are gated to PCA above ~10k points)
    -e, --emit-html [path]
                      Write a standalone offline viewer pre-wired to connect to
                      this server. Path defaults to sphere_viz.html.
    -o, --open        Open in the default browser after the server starts.
                      Opens http://<addr>/ if the WASM studio is found at
                      sphereql-wasm/studio/dist; otherwise opens --emit-html.
    -h, --help        Print this help";

#[tokio::main]
async fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            if msg == "help" {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            eprintln!("error: {msg}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    let corpus = parse_corpus(&args.corpus);
    eprintln!("loading corpus '{}' …", corpus.name());
    let state = match AppState::from_corpus(corpus, args.projection) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "loaded {} points, projection '{}' (EVR {:.1}%), {} categories",
        state.manifest.total_points,
        state.manifest.stats.projection_kind,
        state.manifest.stats.evr * 100.0,
        state.manifest.palette.len(),
    );

    let viewer_host = resolve_viewer_host(&args.addr);
    let server_url = format!("http://{viewer_host}");

    // Optionally write a self-contained offline viewer (no studio deps needed).
    if let Some(ref out_path) = args.emit_html {
        match write_offline_viewer(out_path, &args.corpus, &server_url) {
            Ok(abs) => eprintln!("offline viewer written to {}", abs.display()),
            Err(e) => eprintln!("warning: --emit-html failed: {e}"),
        }
    }

    // Auto-detect the pre-built WASM studio. If found, serve it as the
    // front-end at `GET /` with the auto-connect script injected.
    let studio = find_studio_dir().and_then(|dir| {
        let idx_path = dir.join("index.html");
        match std::fs::read_to_string(&idx_path) {
            Ok(html) => {
                eprintln!("studio found at {} — serving at /", dir.display());
                Some(StudioAssets {
                    index_html: inject_auto_connect(html, &server_url),
                    dir,
                })
            }
            Err(e) => {
                eprintln!("warning: studio dir found but index.html unreadable: {e}");
                None
            }
        }
    });

    // Decide what --open will point at (resolved after bind below).
    let open_url: Option<String> = if args.open_browser {
        if studio.is_some() {
            Some(format!("{server_url}/"))
        } else if args.emit_html.is_some() {
            // Offline file — path is relative to CWD; open via file:// isn't
            // great cross-platform, so just open the server URL instead.
            Some(format!("{server_url}/"))
        } else {
            Some(format!("{server_url}/"))
        }
    } else {
        None
    };

    let app = build_router(Arc::new(RwLock::new(Arc::new(state))), studio);
    let listener = match tokio::net::TcpListener::bind(&args.addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: failed to bind {}: {e}", args.addr);
            return ExitCode::FAILURE;
        }
    };
    eprintln!("serving on {server_url}/ (Ctrl-C to stop)");

    // Open the browser only after the socket is bound so the page can connect.
    if let Some(url) = open_url {
        open_in_browser(&url);
    }

    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        eprintln!("server error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Rewrite a bind address so browsers can actually reach it: `0.0.0.0:PORT`
/// → `127.0.0.1:PORT`, `:::PORT` → `[::1]:PORT`, everything else unchanged.
fn resolve_viewer_host(addr: &str) -> String {
    if addr.starts_with("0.0.0.0:") {
        addr.replacen("0.0.0.0:", "127.0.0.1:", 1)
    } else if addr.starts_with(":::") {
        addr.replacen(":::", "[::1]:", 1)
    } else {
        addr.to_string()
    }
}

/// Inject a `<script>` before `</body>` that calls `connectToServer(url)` when
/// the page loads with no existing hash. The injected script runs after
/// viewer.js (which is inlined earlier), so `connectToServer` is already
/// defined. Existing `#v=` session hashes or explicit `#server=` hashes are
/// preserved — the guard skips auto-connect when a hash is already present.
fn inject_auto_connect(mut html: String, server_url: &str) -> String {
    let script = format!(
        "<script>if(!location.hash||location.hash===\"#\")connectToServer(\"{server_url}\").catch(function(e){{console.warn(\"SphereQL auto-connect:\",e);}});</script>\n"
    );
    html = html.replacen("</body>", &format!("{script}</body>"), 1);
    html
}

/// Look for the pre-built WASM studio in the expected workspace location.
/// Returns `None` when the directory or its `studio.js` sentinel is absent.
fn find_studio_dir() -> Option<std::path::PathBuf> {
    let candidate = std::path::Path::new("sphereql-wasm/studio/dist");
    if candidate.join("studio.js").exists() && candidate.join("index.html").exists() {
        Some(candidate.to_path_buf())
    } else {
        None
    }
}

/// Write a self-contained offline viewer HTML pre-wired to auto-connect to
/// `server_url`. The file inlines three.js so it works without network. Parent
/// directories are created as needed.
fn write_offline_viewer(
    out_path: &str,
    corpus: &str,
    server_url: &str,
) -> std::io::Result<std::path::PathBuf> {
    let title = format!("SphereQL – {corpus}");
    let scene = Scene::builder().title(&title).build();
    let html = inject_auto_connect(scene.to_html(), server_url);

    let path = std::path::Path::new(out_path);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, &html)?;
    std::fs::canonicalize(path)
}

/// Open a URL or file in the OS default application (best-effort; logs on failure).
fn open_in_browser(url: &str) {
    #[cfg(target_os = "windows")]
    let result = std::process::Command::new("cmd")
        .args(["/c", "start", "", url])
        .spawn();
    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("open").arg(url).spawn();
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let result = std::process::Command::new("xdg-open").arg(url).spawn();
    if let Err(e) = result {
        eprintln!("warning: could not open browser: {e}");
    }
}

/// Resolve when the process receives Ctrl-C, for a clean shutdown.
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("\nshutting down …");
}
