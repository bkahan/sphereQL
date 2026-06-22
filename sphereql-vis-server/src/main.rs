//! `sphereql-vis-server` binary: load a corpus, build the indexes, and serve
//! the streaming-viewer API.
//!
//! ```sh
//! cargo run -p sphereql-vis-server -- --corpus stress --addr 127.0.0.1:8080
//! cargo run -p sphereql-vis-server -- --corpus stress --emit-html target/viewer.html --open
//! cargo run -p sphereql-vis-server -- --corpus path/to/corpus.parquet --projection umap_sphere
//! ```

use std::process::ExitCode;
use std::sync::{Arc, RwLock};

use sphereql_embed::ProjectionKind;
use sphereql_vis::Scene;
use sphereql_vis_server::{AppState, build_router, parse_corpus, parse_projection};

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
                        [--emit-html <path>] [--open]

OPTIONS:
    -c, --corpus      Corpus name (stress, hand_crafted, full, dbpedia_500k, …) or
                      a path to a Parquet file [default: stress]
    -a, --addr        Address to bind [default: 127.0.0.1:8080]
    -p, --projection  pca | umap_sphere | laplacian | kernel_pca [default: pca]
                      (O(n²) families are gated to PCA above ~10k points)
    -e, --emit-html [path]
                      Write a viewer HTML pre-wired to connect to this server
                      (auto-sets location.hash). Path defaults to sphere_viz.html.
                      Use with --open for a one-liner start-and-open workflow
    -o, --open        Open the emitted HTML in the default browser after writing
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

    if let Some(ref out_path) = args.emit_html {
        match emit_viewer_html(out_path, &args.corpus, &args.addr) {
            Ok(abs) => {
                eprintln!("viewer HTML written to {}", abs.display());
                if args.open_browser {
                    open_in_browser(&abs);
                }
            }
            Err(e) => {
                eprintln!("warning: --emit-html failed: {e}");
            }
        }
    }

    let app = build_router(Arc::new(RwLock::new(Arc::new(state))));
    let listener = match tokio::net::TcpListener::bind(&args.addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: failed to bind {}: {e}", args.addr);
            return ExitCode::FAILURE;
        }
    };
    eprintln!("serving on http://{} (Ctrl-C to stop)", args.addr);
    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        eprintln!("server error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Generate a viewer HTML pre-wired to connect to `addr` on load.
///
/// Creates a minimal empty scene (no inline points), injects a small `<script>`
/// that sets `location.hash` to `#server=<url>` so the viewer auto-connects,
/// and writes to `out_path`. Returns the canonicalized absolute path.
fn emit_viewer_html(
    out_path: &str,
    corpus: &str,
    addr: &str,
) -> std::io::Result<std::path::PathBuf> {
    // If the server binds to 0.0.0.0 or :: the browser needs a routable host.
    let viewer_host = if addr.starts_with("0.0.0.0:") {
        addr.replacen("0.0.0.0:", "127.0.0.1:", 1)
    } else if addr.starts_with(":::") || addr == "::" {
        addr.replacen("::", "[::1]:", 1)
    } else {
        addr.to_string()
    };
    let server_url = format!("http://{viewer_host}");

    let title = format!("SphereQL – {corpus}");
    let scene = Scene::builder().title(&title).build();
    let mut html = scene.to_html();

    // Inject auto-connect after viewer.js (before </body>). The #server= IIFE
    // in viewer.js has already run by this point, so we call connectToServer()
    // directly — it's a top-level function and reachable from any later script.
    // Guard: skip if a #v= session hash or explicit #server= hash is present
    // (applyViewHash / the IIFE already handled those on boot).
    let connect_script = format!(
        "<script>if(!location.hash||location.hash===\"#\")connectToServer(\"{server_url}\").catch(function(e){{console.warn(\"SphereQL auto-connect:\",e);}});</script>\n"
    );
    html = html.replacen("</body>", &format!("{connect_script}</body>"), 1);

    let path = std::path::Path::new(out_path);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, &html)?;
    std::fs::canonicalize(path)
}

/// Open a file in the OS default application (best-effort; logs on failure).
fn open_in_browser(path: &std::path::Path) {
    let path_str = path.to_string_lossy();
    #[cfg(target_os = "windows")]
    let result = std::process::Command::new("cmd")
        .args(["/c", "start", "", &*path_str])
        .spawn();
    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("open").arg(&*path_str).spawn();
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let result = std::process::Command::new("xdg-open")
        .arg(&*path_str)
        .spawn();
    if let Err(e) = result {
        eprintln!("warning: could not open browser: {e}");
    }
}

/// Resolve when the process receives Ctrl-C, for a clean shutdown.
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("\nshutting down …");
}
