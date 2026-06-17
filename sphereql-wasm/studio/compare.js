// SphereQL Compare — side-by-side projections with synced cameras.
// One Web Worker builds two Scenes from the same corpus (projection A / B) and
// injects each into its own viewer iframe (embed.html#embed). The iframes
// broadcast their camera moves to this parent, which relays them to the other
// pane, so orbiting one orbits both. The viewer's own epsilon gate stops the
// relay from echoing into a feedback loop.
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

  let ready = false;
  let nextId = 0;
  const pend = {}; // id → the iframe whose pane the response should fill

  const projName = (v) =>
    ({ "": "PCA", UmapSphere: "UMAP sphere", LaplacianEigenmap: "Laplacian", KernelPca: "Kernel PCA" }[v] || v);

  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") { ready = true; statusEl.textContent = "ready — paste a corpus, pick two projections"; return; }
    if (m.type === "fatal") { statusEl.textContent = "✗ wasm failed: " + m.error; return; }
    const pane = pend[m.id];
    if (m.id !== undefined) delete pend[m.id];
    if (!m.ok) { statusEl.textContent = "✗ " + m.error; return; }
    if (pane && m.json !== undefined) {
      try {
        pane.contentWindow.postMessage({ type: "sphereql-scene", scene: JSON.parse(m.json) }, "*");
        statusEl.textContent = "✓ compared";
      } catch (err) {
        statusEl.textContent = "✗ " + err.message;
      }
    }
  };

  function build() {
    if (!ready) return;
    const c = corpus.value.trim();
    if (!c) { statusEl.textContent = "paste corpus JSON first"; return; }
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

  // Relay a pane's camera to the other pane.
  window.addEventListener("message", (e) => {
    const m = e.data;
    if (!m || m.type !== "sphereql-cam") return;
    const to = e.source === paneA.contentWindow ? paneB : e.source === paneB.contentWindow ? paneA : null;
    if (to) {
      try { to.contentWindow.postMessage(m, "*"); } catch (err) { /* not loaded yet */ }
    }
  });
})();
