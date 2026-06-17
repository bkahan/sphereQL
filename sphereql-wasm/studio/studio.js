// SphereQL Studio driver (main thread). Wires the studio chrome to a Web
// Worker that runs the wasm pipeline, and feeds the resulting Scene to the
// viewer's GLOBAL rebuild()/parseScene() (defined by the inlined viewer.js
// that ran just before this script). Debounced so typing re-projects live
// without flooding the worker; stale worker responses are dropped by id.

/* global rebuild, parseScene, highlightByIds, setMorphTarget, applyMorph, clearMorph */
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

  let mode = "lingua";
  let wasmReady = false;
  let latestId = 0;
  let debounce = null;
  const morphPending = {}; // ids whose corpus response is a morph target, not a rebuild

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
    if (!wasmReady) return;
    const text = input.value;
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
    clearTimeout(debounce);
    debounce = setTimeout(run, 320);
  }

  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") {
      // The baked default scene (the 775-point demo corpus) is already showing;
      // don't auto-run over it. The user drives it from here (type / example /
      // corpus mode / morph).
      wasmReady = true;
      setStatus("ready — showing the demo corpus · type or load an example");
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
    input.value = mode === "lingua" ? LINGUA_EXAMPLE : CORPUS_EXAMPLE;
    run();
  });
  collapse.addEventListener("click", () => panel.classList.toggle("collapsed"));

  document.querySelectorAll(".st-mode").forEach((b) =>
    b.addEventListener("click", () => {
      if (b.dataset.mode === mode) return;
      mode = b.dataset.mode;
      document.querySelectorAll(".st-mode").forEach((x) => x.classList.toggle("active", x === b));
      corpusOpts.style.display = mode === "corpus" ? "flex" : "none";
      queryRow.style.display = mode === "corpus" ? "flex" : "none";
      morphRow.style.display = mode === "corpus" ? "flex" : "none";
      if (mode !== "corpus") resetMorphUI();
      input.placeholder =
        mode === "lingua"
          ? "Paste prose — concepts are placed on the sphere as you type…"
          : 'Paste corpus JSON: {"categories":[…],"embeddings":[[…],…]}';
      // Swap to the matching example if the box still holds the other one.
      if (input.value === LINGUA_EXAMPLE || input.value === CORPUS_EXAMPLE || !input.value) {
        input.value = mode === "lingua" ? LINGUA_EXAMPLE : CORPUS_EXAMPLE;
      }
      run();
    })
  );
})();
