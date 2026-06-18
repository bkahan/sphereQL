//! `sphereql-vis-server` binary: load a corpus, build the indexes, and serve
//! the streaming-viewer API.
//!
//! ```sh
//! cargo run -p sphereql-vis-server -- --corpus stress --addr 127.0.0.1:8080
//! cargo run -p sphereql-vis-server -- --corpus path/to/corpus.parquet --projection umap_sphere
//! ```

use std::process::ExitCode;
use std::sync::Arc;

use sphereql_embed::ProjectionKind;
use sphereql_vis_server::{AppState, build_router, parse_corpus};

struct Args {
    corpus: String,
    addr: String,
    projection: ProjectionKind,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            corpus: "stress".to_string(),
            addr: "127.0.0.1:8080".to_string(),
            projection: ProjectionKind::Pca,
        }
    }
}

fn parse_projection(s: &str) -> Option<ProjectionKind> {
    match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "pca" => Some(ProjectionKind::Pca),
        "kernel_pca" | "kernelpca" => Some(ProjectionKind::KernelPca),
        "laplacian" | "laplacian_eigenmap" => Some(ProjectionKind::LaplacianEigenmap),
        "umap" | "umap_sphere" => Some(ProjectionKind::UmapSphere),
        _ => None,
    }
}

/// Minimal `--flag value` parser — avoids a clap dependency for three options.
/// Returns `Err` with a usage hint on an unknown flag or missing value.
fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--corpus" | "-c" => {
                args.corpus = it.next().ok_or("--corpus needs a value")?;
            }
            "--addr" | "-a" => {
                args.addr = it.next().ok_or("--addr needs a value")?;
            }
            "--projection" | "-p" => {
                let v = it.next().ok_or("--projection needs a value")?;
                args.projection =
                    parse_projection(&v).ok_or_else(|| format!("unknown projection '{v}'"))?;
            }
            "--help" | "-h" => return Err("help".to_string()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(args)
}

const USAGE: &str = "\
sphereql-vis-server — out-of-core query server for the streaming viewer

USAGE:
    sphereql-vis-server [--corpus <name|path>] [--addr <host:port>] [--projection <kind>]

OPTIONS:
    -c, --corpus      Corpus name (stress, hand_crafted, full, dbpedia_500k, …) or
                      a path to a Parquet file [default: stress]
    -a, --addr        Address to bind [default: 127.0.0.1:8080]
    -p, --projection  pca | umap_sphere | laplacian | kernel_pca [default: pca]
                      (O(n²) families are gated to PCA above ~10k points)
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

    let app = build_router(Arc::new(state));
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

/// Resolve when the process receives Ctrl-C, for a clean shutdown.
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("\nshutting down …");
}
