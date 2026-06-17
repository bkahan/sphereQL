// SphereQL Compare — side-by-side projections with synced cameras.
// One Web Worker builds two Scenes from the same corpus (projection A / B) and
// injects each into its own viewer iframe (embed.html#embed). The iframes
// broadcast their camera moves to this parent, which relays them to the other
// pane, so orbiting one orbits both. The viewer's own epsilon gate stops the
// relay from echoing into a feedback loop.
//
// The corpus textarea is optional: leave it empty to compare the built-in
// 775-point demo corpus (fetched), or paste your own {categories,embeddings}.
(function () {
  "use strict";

  const worker = new Worker("worker.js");
  const paneA = document.getElementById("paneA");
  const paneB = document.getElementById("paneB");
  const corpus = document.getElementById("corpus");
  const projA = document.getElementById("projA");
  const projB = document.getElementById("projB");
  const go = document.getElementById("go");
  const statusEl = document.getElementById("status");
  const tagA = document.getElementById("tagA");
  const tagB = document.getElementById("tagB");

  let ready = false; // wasm worker ready
  let nextId = 0;
  const pend = {}; // id → the iframe whose pane the response should fill
  const paneReady = new Map(); // iframe → its embed viewer registered its listener
  const pendingScene = new Map(); // iframe → scene awaiting a not-yet-ready pane

  let demoCorpus = "";
  fetch("demo-corpus.json")
    .then((r) => (r.ok ? r.text() : ""))
    .then((t) => { demoCorpus = t; refreshStatus(); })
    .catch(() => {});

  const projName = (v) =>
    ({ "": "PCA", UmapSphere: "UMAP sphere", LaplacianEigenmap: "Laplacian", KernelPca: "Kernel PCA" }[v] || v);

  function refreshStatus() {
    if (!ready) return;
    statusEl.textContent = demoCorpus
      ? "ready — press compare (uses the demo corpus, or paste your own)"
      : "ready — paste a corpus, pick two projections";
  }

  // Inject a scene into a pane once that pane's viewer is ready (else queue it).
  function inject(pane, sceneJson) {
    if (paneReady.get(pane)) {
      try { pane.contentWindow.postMessage({ type: "sphereql-scene", scene: JSON.parse(sceneJson) }, "*"); }
      catch (err) { statusEl.textContent = "✗ " + err.message; }
    } else {
      pendingScene.set(pane, sceneJson);
    }
  }

  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") { ready = true; refreshStatus(); return; }
    if (m.type === "fatal") { statusEl.textContent = "✗ wasm failed: " + m.error; return; }
    const pane = pend[m.id];
    if (m.id !== undefined) delete pend[m.id];
    if (!m.ok) { statusEl.textContent = "✗ " + m.error; return; }
    if (pane && m.json !== undefined) { inject(pane, m.json); statusEl.textContent = "✓ compared"; }
  };

  function build() {
    if (!ready) return;
    const c = corpus.value.trim() || demoCorpus;
    if (!c) { statusEl.textContent = "paste corpus JSON (demo not loaded yet)"; return; }
    statusEl.textContent = "building…";
    tagA.textContent = projName(projA.value);
    tagB.textContent = projName(projB.value);
    const ia = ++nextId, ib = ++nextId;
    pend[ia] = paneA;
    pend[ib] = paneB;
    worker.postMessage({ id: ia, kind: "corpus", corpus: c, config: projA.value ? JSON.stringify({ projection_kind: projA.value }) : null, title: projName(projA.value) });
    worker.postMessage({ id: ib, kind: "corpus", corpus: c, config: projB.value ? JSON.stringify({ projection_kind: projB.value }) : null, title: projName(projB.value) });
  }

  go.addEventListener("click", build);

  window.addEventListener("message", (e) => {
    const m = e.data;
    if (!m || typeof m !== "object") return;
    const fromA = e.source === paneA.contentWindow;
    const fromB = e.source === paneB.contentWindow;
    if (m.type === "sphereql-embed-ready") {
      // A pane's viewer is live; flush any scene that arrived before it.
      const pane = fromA ? paneA : fromB ? paneB : null;
      if (pane) {
        paneReady.set(pane, true);
        const queued = pendingScene.get(pane);
        if (queued) { pendingScene.delete(pane); inject(pane, queued); }
      }
      return;
    }
    if (m.type === "sphereql-cam") {
      // Relay a pane's camera to the other pane.
      const to = fromA ? paneB : fromB ? paneA : null;
      if (to) { try { to.contentWindow.postMessage(m, "*"); } catch (err) { /* not loaded yet */ } }
    }
  });
})();
