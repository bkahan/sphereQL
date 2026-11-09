// SphereQL Studio driver (main thread). Wires the studio chrome to a Web
// Worker that runs the wasm pipeline, and feeds the resulting Scene to the
// viewer's GLOBAL rebuild()/parseScene() (defined by the inlined viewer.js
// that ran just before this script). Debounced so typing re-projects live
// without flooding the worker; stale worker responses are dropped by id.

/* global rebuild, parseScene, highlightByIds, setMorphTarget, applyMorph, clearMorph, D */
(function () {
  "use strict";

  const LINGUA_EXAMPLE =
    "A neural network is a model that learns from data. Gradient descent " +
    "optimizes the model's parameters. Backpropagation computes the gradients. " +
    "A transformer is a neural network for language. Cooking transforms raw " +
    "ingredients with heat. A recipe is a set of instructions for cooking.";

  // The built-in demo corpus (raw {categories, embeddings}), fetched lazily so
  // corpus operations (run / morph / query) can act on the 775-point demo
  // without dumping ~2 MB of vectors into the textarea. The textarea is for
  // pasting your OWN corpus; an empty box falls back to this demo.
  let demoCorpus = "";
  let demoFailed = false; // demo-corpus.json fetch failed (missing / served wrong)?
  let demoPrimed = false; // demo pipeline built in the worker yet?
  let corpusRan = false; // has the user run/morphed a corpus of their own?
  fetch("demo-corpus.json")
    .then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.text(); })
    .then((t) => { demoCorpus = t; primeDemo(); })
    .catch(() => { demoFailed = true; });
  const corpusText = () => input.value.trim() || demoCorpus;

  // Build the demo corpus pipeline in the worker (no scene, so the displayed
  // baked scene is untouched) so a query works immediately against the demo —
  // even with the textarea empty. Skipped once the user runs their own corpus.
  function primeDemo() {
    if (!wasmReady || !demoCorpus || demoPrimed || corpusRan) return;
    demoPrimed = true;
    worker.postMessage({ id: ++latestId, kind: "load", corpus: demoCorpus, config: null });
  }

  const worker = new Worker("worker.js");
  worker.onerror = (e) => {
    wasmFailed = true; pendingRun = false;
    setStatus("✗ wasm worker crashed — reload the page", "err");
  };
  const input = document.getElementById("studio-input");
  const statusEl = document.getElementById("studio-status");
  const proj = document.getElementById("studio-proj");
  const corpusOpts = document.getElementById("st-corpus-opts");
  const runBtn = document.getElementById("studio-run");
  const exBtn = document.getElementById("studio-example");
  const collapse = document.getElementById("studio-collapse");
  const panel = document.getElementById("studio");
  const queryRow = document.getElementById("st-query-row");
  const queryInput = document.getElementById("studio-query");
  const kInput = document.getElementById("studio-k");
  const findBtn = document.getElementById("studio-find");
  const morphRow = document.getElementById("st-morph-row");
  const morphProj = document.getElementById("studio-morph-proj");
  const morphSlider = document.getElementById("studio-morph");

  let mode = "corpus"; // open on Corpus mode (showing the baked 775 demo scene)
  let wasmReady = false;
  let wasmFailed = false; // wasm module failed to load (unrecoverable without reload)
  let pendingRun = false; // a run requested before wasm finished loading
  let latestId = 0;
  let debounce = null;
  const morphPending = {}; // ids whose corpus response is a morph target, not a rebuild
  // Per-mode input buffers: Corpus starts EMPTY (paste your own; the default
  // *scene* is the baked 775 demo, restored via rebuild(D)). Lingua seeds the
  // prose example.
  const buffers = { corpus: "", lingua: LINGUA_EXAMPLE };

  function resetMorphUI() {
    morphProj.value = "";
    morphSlider.value = "0";
    morphSlider.disabled = true;
  }

  function setStatus(text, cls) {
    statusEl.textContent = text;
    statusEl.className = "st-status" + (cls ? " " + cls : "");
  }

  function run() {
    if (wasmFailed) { setStatus("wasm unavailable — reload the page", "err"); return; }
    if (!wasmReady) { pendingRun = true; return; }
    const id = ++latestId;
    if (mode === "lingua") {
      const text = input.value;
      if (!text.trim()) return;
      setStatus("computing…", "busy");
      worker.postMessage({ id, kind: "lingua", text });
    } else {
      // When connected to a server, delegate projection to the live server instead
      // of blocking the WASM worker (UMAP in WASM on the demo corpus is O(n²) slow).
      if (typeof window.__sqServerReproject === "function") { window.__sqServerReproject(proj.value); return; }
      const corpus = corpusText(); // your paste, else the demo corpus
      if (!corpus) { setStatus(demoFailed ? "demo corpus unavailable — paste your own" : "paste a corpus to run", "err"); return; }
      const cfg = proj.value ? JSON.stringify({ projection_kind: proj.value }) : null;
      corpusRan = true;
      setStatus("computing…", "busy");
      worker.postMessage({ id, kind: "corpus", corpus, config: cfg, title: "Corpus" });
    }
  }

  function schedule() {
    // Live re-projection only in lingua mode; corpus JSON is run explicitly
    // (it's large, and the default is the baked demo scene).
    if (mode !== "lingua") return;
    clearTimeout(debounce);
    debounce = setTimeout(run, 320);
  }

  // Restore the baked 775-point demo scene instantly (no wasm) — the Corpus
  // mode default and what we show on returning to Corpus mode.
  function restoreDemoScene() {
    try {
      rebuild(D);
      resetMorphUI();
      setStatus("✓ demo corpus · " + (D.points ? D.points.length : 0) + " points");
    } catch (err) {
      setStatus("✗ " + err.message, "err");
    }
  }

  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") {
      // The baked default scene (the 775-point demo corpus) is already showing;
      // don't auto-run over it. The user drives it from here (type / example /
      // corpus mode / morph).
      wasmReady = true;
      setStatus(demoFailed ? "ready — demo corpus unavailable · paste a corpus or use Lingua" : "ready — demo corpus · paste your own, or try Lingua text");
      if (pendingRun) { pendingRun = false; run(); }
      primeDemo(); // make the demo queryable in the background
      return;
    }
    if (m.type === "fatal") {
      wasmFailed = true;
      pendingRun = false; // never executes — don't leave it queued
      setStatus("✗ wasm failed to load — reload the page", "err");
      return;
    }
    if (m.id < latestId) return; // a newer request superseded this one
    if (!m.ok) {
      setStatus("✗ " + m.error, "err");
      return;
    }
    if (m.loaded) return; // demo pipeline primed in the worker — query is now ready
    if (m.neighbors !== undefined) {
      // Query result: highlight the nearest points by id on the current scene.
      try {
        const n = highlightByIds(m.neighbors);
        setStatus("◎ " + n + " / " + m.neighbors.length + " neighbors");
      } catch (err) {
        setStatus("✗ " + err.message, "err");
      }
      return;
    }
    try {
      const scene = parseScene(JSON.parse(m.json));
      if (morphPending[m.id]) {
        delete morphPending[m.id];
        const matched = setMorphTarget(scene); // A stays; B becomes the morph target
        morphSlider.value = "0";
        morphSlider.disabled = false;
        applyMorph(0);
        setStatus("morph ready · " + matched + " points aligned");
      } else {
        rebuild(scene); // a fresh scene clears any morph (teardown) — reset its UI
        resetMorphUI();
        setStatus("✓ " + scene.points.length + " points · " + scene.overlays.length + " overlays");
      }
    } catch (err) {
      setStatus("✗ " + err.message, "err");
    }
  };

  function buildMorphTarget() {
    if (wasmFailed) { setStatus("wasm unavailable — reload the page", "err"); resetMorphUI(); return; }
    if (!wasmReady || mode !== "corpus") return;
    if (!morphProj.value) {
      clearMorph();
      applyMorph(0);
      morphSlider.value = "0";
      morphSlider.disabled = true;
      return;
    }
    const corpus = corpusText(); // your paste, else the demo corpus
    if (!corpus) { setStatus(demoFailed ? "demo corpus unavailable — paste your own to morph" : "paste a corpus first to morph", "err"); resetMorphUI(); return; }
    corpusRan = true;
    const id = ++latestId;
    morphPending[id] = true;
    setStatus("building morph target…", "busy");
    worker.postMessage({ id, kind: "corpus", corpus, config: JSON.stringify({ projection_kind: morphProj.value }), title: "B" });
  }

  function find() {
    if (wasmFailed) { setStatus("wasm unavailable — reload the page", "err"); return; }
    if (!wasmReady || mode !== "corpus") return;
    const query = queryInput.value.trim();
    if (!query) {
      setStatus("enter a query vector", "err");
      return;
    }
    // Query needs a pipeline in the worker. If the user hasn't run their own and
    // the demo couldn't be primed, say so clearly rather than waiting on the
    // worker's generic "run a corpus first".
    if (!corpusRan && !demoPrimed) {
      setStatus(demoFailed ? "demo corpus unavailable — run a corpus to query" : "demo still loading — try again in a moment", "err");
      return;
    }
    const id = ++latestId;
    setStatus("querying…", "busy");
    worker.postMessage({ id, kind: "query", query, k: parseInt(kInput.value, 10) || 8 });
  }

  // ── chrome wiring ───────────────────────────────────────────────────────
  input.addEventListener("input", schedule);
  runBtn.addEventListener("click", run);
  proj.addEventListener("change", run);
  findBtn.addEventListener("click", find);
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") find();
  });
  morphProj.addEventListener("change", buildMorphTarget);
  morphSlider.addEventListener("input", () => applyMorph(parseFloat(morphSlider.value)));
  exBtn.addEventListener("click", () => {
    // Lingua-only: load the prose example and run it. (Hidden in corpus mode —
    // the corpus default is the baked demo scene.)
    input.value = LINGUA_EXAMPLE;
    run();
  });
  collapse.addEventListener("click", () => panel.classList.toggle("collapsed"));

  // Reflect the active mode in the chrome (button + which rows / placeholder).
  function showModeUI() {
    document.querySelectorAll(".st-mode").forEach((x) => x.classList.toggle("active", x.dataset.mode === mode));
    const corpus = mode === "corpus";
    corpusOpts.style.display = corpus ? "flex" : "none";
    queryRow.style.display = corpus ? "flex" : "none";
    morphRow.style.display = corpus ? "flex" : "none";
    exBtn.style.display = corpus ? "none" : "inline-flex"; // example is lingua-only
    input.placeholder = corpus
      ? 'Paste corpus JSON {"categories":[…],"embeddings":[[…],…]} and Run — or stay on the demo scene'
      : "Paste prose — concepts are placed on the sphere as you type…";
  }

  function switchMode(next) {
    if (next === mode) return;
    buffers[mode] = input.value; // remember this workspace
    mode = next;
    input.value = buffers[mode]; // restore the target workspace
    if (mode !== "corpus") resetMorphUI();
    showModeUI();
    if (mode === "corpus") restoreDemoScene(); // Corpus default = the baked 775 scene
    else run(); // Lingua: build from the (example) prose
  }

  document.querySelectorAll(".st-mode").forEach((b) =>
    b.addEventListener("click", () => switchMode(b.dataset.mode))
  );

  // ── Initial state: Corpus mode, demo scene already baked + showing ───────
  input.value = buffers[mode];
  showModeUI();
})();
