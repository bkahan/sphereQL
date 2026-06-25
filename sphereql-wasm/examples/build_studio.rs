//! Emit the self-contained Studio shell to `sphereql-wasm/studio/dist/index.html`.
//!
//! The page is the ordinary sphereql-vis viewer (three.js + the SHARED
//! `viewer.js` inlined, empty initial scene) with the Studio chrome injected
//! after `<body>` and `studio.js` referenced before `</body>`. Reusing
//! `Scene::to_html()` means the studio and the baked viewer can never drift.
//!
//! Run: `cargo run -p sphereql-wasm --example build_studio`
//! Then build the wasm + copy the worker (see `studio/build.sh`).

// build_studio is a native-only tool: it bakes the demo scene/corpus from the
// (native-only) sphereql-examples + sphereql-corpus crates, which are target-
// gated out of wasm builds. `wasm-pack test` compiles examples for wasm32, so
// give it a trivial no-op main there; the real tool only builds on the host.
#[cfg(all(feature = "scene", not(target_arch = "wasm32")))]
fn main() -> std::io::Result<()> {
    // Bake the 775-point HandCrafted demo scene (auto-tuned, full overlays — the
    // same picture `visualize_corpus` renders) as the studio's opening scene, so
    // it lands on something rich instead of an empty sphere. The live
    // lingua/corpus/compare modes still drive rebuild() from there.
    let chrome = include_str!("../studio/chrome.html");
    let base = sphereql_examples::demo_scene(sphereql_corpus::CorpusId::HandCrafted).to_html();

    // Inject the chrome right after <body>, and the studio driver right before
    // </body> (after the inlined viewer, so its global rebuild()/parseScene()
    // already exist when studio.js spawns the worker).
    let html = base
        .replacen("<body>", &format!("<body>\n{chrome}"), 1)
        .replacen("</body>", "<script src=\"studio.js\"></script>\n</body>", 1);

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("studio/dist");
    std::fs::create_dir_all(&dir)?;
    let out = dir.join("index.html");
    std::fs::write(&out, html)?;
    eprintln!("wrote {}", out.display());

    // A plain viewer (no studio chrome) for the side-by-side compare iframes.
    // Loaded as `embed.html#embed`, where viewer.js's compare-embed block
    // accepts an injected scene + synced camera over postMessage.
    std::fs::write(dir.join("embed.html"), &base)?;
    eprintln!("wrote {}", dir.join("embed.html").display());

    // The raw demo corpus — fetched by the studio / compare worker so the demo
    // can be re-projected, compared, morphed and queried (the textarea stays
    // empty for the user's own paste).
    let corpus = sphereql_examples::demo_corpus_json(sphereql_corpus::CorpusId::HandCrafted);
    std::fs::write(dir.join("demo-corpus.json"), &corpus)?;
    eprintln!(
        "wrote {} ({} bytes)",
        dir.join("demo-corpus.json").display(),
        corpus.len()
    );
    Ok(())
}

#[cfg(all(not(feature = "scene"), not(target_arch = "wasm32")))]
fn main() {
    eprintln!("build_studio requires the `scene` feature (on by default)");
    std::process::exit(1);
}

// Native-only tool — never run on wasm; trivial main keeps `wasm-pack test`
// (which compiles examples for wasm32) from referencing the host-only deps.
#[cfg(target_arch = "wasm32")]
fn main() {}
