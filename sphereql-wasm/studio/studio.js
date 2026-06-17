// SphereQL Studio driver (main thread). Wires the studio chrome to a Web
// Worker that runs the wasm pipeline, and feeds the resulting Scene to the
// viewer's GLOBAL rebuild()/parseScene() (defined by the inlined viewer.js
// that ran just before this script). Debounced so typing re-projects live
// without flooding the worker; stale worker responses are dropped by id.

/* global rebuild, parseScene, highlightByIds */
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

  let mode = "lingua";
  let wasmReady = false;
  let latestId = 0;
  let debounce = null;

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
      wasmReady = true;
      if (!input.value) input.value = LINGUA_EXAMPLE;
      run();
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
      rebuild(scene);
      setStatus("✓ " + scene.points.length + " points · " + scene.overlays.length + " overlays");
    } catch (err) {
      setStatus("✗ " + err.message, "err");
    }
  };

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
