//! Emit the self-contained Studio shell to `sphereql-wasm/studio/dist/index.html`.
//!
//! The page is the ordinary sphereql-vis viewer (three.js + the SHARED
//! `viewer.js` inlined, empty initial scene) with the Studio chrome injected
//! after `<body>` and `studio.js` referenced before `</body>`. Reusing
//! `Scene::to_html()` means the studio and the baked viewer can never drift.
//!
//! Run: `cargo run -p sphereql-wasm --example build_studio`
//! Then build the wasm + copy the worker (see `studio/build.sh`).

#[cfg(feature = "scene")]
fn main() -> std::io::Result<()> {
    use sphereql_vis::Scene;

    let chrome = include_str!("../studio/chrome.html");
    let base = Scene::builder().title("SphereQL Studio").build().to_html();

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
    Ok(())
}

#[cfg(not(feature = "scene"))]
fn main() {
    eprintln!("build_studio requires the `scene` feature (on by default)");
    std::process::exit(1);
}
