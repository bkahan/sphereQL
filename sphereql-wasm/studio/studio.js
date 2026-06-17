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

  const CORPUS_EXAMPLE = JSON.stringify(
    {
      categories: ["science", "cooking", "science", "cooking", "science", "cooking"],
      embeddings: [
        [0.9, 0.1, 0.0, 0.1],
        [0.1, 0.9, 0.1, 0.0],
        [0.85, 0.15, 0.05, 0.1],
        [0.05, 0.95, 0.0, 0.1],
        [0.8, 0.2, 0.1, 0.0],
        [0.1, 0.85, 0.2, 0.05],
      ],
    },
    null,
    0
  );

  const worker = new Worker("worker.js");
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
    if (!wasmReady) { pendingRun = true; return; }
    const text = input.value;
    if (!text.trim()) {
      // Nothing to build — keep whatever scene is showing (e.g. the demo).
      if (mode === "corpus") setStatus("paste a corpus, or stay on the demo scene");
      return;
    }
    const id = ++latestId;
    setStatus("computing…", "busy");
    if (mode === "lingua") {
      worker.postMessage({ id, kind: "lingua", text });
    } else {
      const cfg = proj.value ? JSON.stringify({ projection_kind: proj.value }) : null;
      worker.postMessage({ id, kind: "corpus", corpus: text, config: cfg, title: "Corpus" });
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
      setStatus("ready — demo corpus · paste your own, or try Lingua text");
      if (pendingRun) { pendingRun = false; run(); }
      return;
    }
    if (m.type === "fatal") {
      setStatus("✗ wasm failed: " + m.error, "err");
      return;
    }
    if (m.id < latestId) return; // a newer request superseded this one
    if (!m.ok) {
      setStatus("✗ " + m.error, "err");
      return;
    }
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
    if (!wasmReady || mode !== "corpus") return;
    if (!morphProj.value) {
      clearMorph();
      applyMorph(0);
      morphSlider.value = "0";
      morphSlider.disabled = true;
      return;
    }
    if (!input.value.trim()) { setStatus("paste & run a corpus first to morph", "err"); resetMorphUI(); return; }
    const id = ++latestId;
    morphPending[id] = true;
    setStatus("building morph target…", "busy");
    worker.postMessage({ id, kind: "corpus", corpus: input.value, config: JSON.stringify({ projection_kind: morphProj.value }), title: "B" });
  }

  function find() {
    if (!wasmReady || mode !== "corpus") return;
    const query = queryInput.value.trim();
    if (!query) {
      setStatus("enter a query vector", "err");
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
    // Load the matching example template and run it (explicit user action).
    input.value = mode === "lingua" ? LINGUA_EXAMPLE : CORPUS_EXAMPLE;
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
