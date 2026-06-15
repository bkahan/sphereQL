//! Render a corpus to an interactive 3D sphere — the end-to-end visualization
//! demo.
//!
//! Loads a corpus, lets the auto-tuner pick a projection (so the picture
//! reflects the metalearning result), builds the pipeline, and emits ONE rich
//! self-contained HTML scene: the projected point cloud colored by category,
//! plus the full overlay set — category centroids, classified bridges,
//! geodesic concept paths, Voronoi territory caps, antipodes, coverage caps,
//! and domain-group spokes.
//!
//! Run with:
//!   cargo run -p sphereql-examples --example visualize_corpus --release
//!
//! Flags:
//!   --corpus <handcrafted|stress>   corpus to load (default: handcrafted)
//!   --out <path>                    output HTML (default: target/sphere_viz.html)
//!   --cdn                           load three.js from a CDN (smaller file)
//!   --open                          open the result in a browser when done
//!
//! The example is CI-safe: it writes the file and prints its path, and only
//! opens a browser when `--open` is passed.

use sphereql::embed::{
    CompositeMetric, PipelineConfig, PipelineInput, ProjectionKind, SearchStrategy, auto_tune,
};
use sphereql_corpus::{CorpusId, embed};
use sphereql_examples::{build_corpus_scene, tuning_params};

struct Args {
    corpus: CorpusId,
    out: String,
    cdn: bool,
    open: bool,
}

fn parse_args() -> Args {
    let mut corpus = CorpusId::HandCrafted;
    let mut out = "target/sphere_viz.html".to_string();
    let mut cdn = false;
    let mut open = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--corpus" => {
                corpus = match it.next().as_deref() {
                    Some("stress") => CorpusId::Stress,
                    Some("handcrafted") | None => CorpusId::HandCrafted,
                    Some(other) => {
                        eprintln!("unknown corpus '{other}', using handcrafted");
                        CorpusId::HandCrafted
                    }
                };
            }
            "--out" => {
                if let Some(p) = it.next() {
                    out = p;
                }
            }
            "--cdn" => cdn = true,
            "--open" => open = true,
            other => eprintln!("ignoring unknown arg '{other}'"),
        }
    }
    Args {
        corpus,
        out,
        cdn,
        open,
    }
}

fn main() {
    let args = parse_args();

    println!("Loading corpus: {}", args.corpus.name());
    let corpus = match args.corpus.load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to load {}: {e}", args.corpus.name());
            std::process::exit(1);
        }
    };
    let categories: Vec<String> = corpus.iter().map(|c| c.category.to_string()).collect();
    let labels: Vec<&str> = corpus.iter().map(|c| c.label).collect();
    let embeddings: Vec<Vec<f64>> = corpus
        .iter()
        .enumerate()
        .map(|(i, c)| embed(&c.features, 1000 + i as u64))
        .collect();
    println!("  {} concepts loaded", corpus.len());

    // Let the tuner pick a projection across all feasible families.
    let all_kinds = [
        ProjectionKind::Pca,
        ProjectionKind::UmapSphere,
        ProjectionKind::LaplacianEigenmap,
        ProjectionKind::KernelPca,
    ];
    let (budget, space) = tuning_params(corpus.len(), &all_kinds);
    let metric = CompositeMetric::default_composite();
    println!(
        "Auto-tuning (budget={budget}) over {:?}...",
        space.projection_kinds
    );

    let (pipeline, report) = auto_tune(
        PipelineInput {
            categories: categories.clone(),
            embeddings,
        },
        &space,
        &metric,
        SearchStrategy::Random {
            budget,
            seed: 0x5EED_C0FFEE,
            max_wall_secs: None,
        },
        &PipelineConfig::default(),
    )
    .expect("auto_tune failed");

    let evr = pipeline.explained_variance_ratio();
    println!(
        "  winner: {} (score {:.4}, EVR {:.1}%)",
        pipeline.projection_kind().name(),
        report.best_score,
        evr * 100.0
    );

    let title = format!("SphereQL — {}", args.corpus.name());
    let scene = build_corpus_scene(&title, &pipeline, &labels, evr);
    println!(
        "  scene: {} points, {} overlays",
        scene.points.len(),
        scene.overlays.len()
    );

    let html = if args.cdn {
        scene.to_html_cdn()
    } else {
        scene.to_html()
    };
    if let Err(e) = std::fs::write(&args.out, &html) {
        eprintln!("failed to write {}: {e}", args.out);
        std::process::exit(1);
    }
    let abs = std::fs::canonicalize(&args.out).unwrap_or_else(|_| args.out.clone().into());
    println!("\nWrote {} ({} KB)", abs.display(), html.len() / 1024);
    if args.cdn {
        println!("(CDN mode — requires network to view)");
    } else {
        println!("(self-contained — opens offline, no network needed)");
    }

    if args.open {
        open_in_browser(&abs.to_string_lossy());
    } else {
        println!("Open it in a browser, or re-run with --open.");
    }
}

/// Best-effort browser open, per platform. Only called with `--open`.
fn open_in_browser(path: &str) {
    let url = format!("file://{path}");
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd")
        .args(["/C", "start", "", &url])
        .spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(&url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
}
