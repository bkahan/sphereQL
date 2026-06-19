// Phase D streaming-debugger flow, headless: drive connectToServer against a
// mock fetch and verify the wiring — manifest + base tile + diagnostics are
// fetched, the streamer is live with the base tile rendered, and inspecting a
// row issues the /points + /nearest fetches. (The DOM stub returns ephemeral
// elements, so this asserts on fetch traffic + streamer state, not on rendered
// markup — the visuals are browser-validated.)
const path = require("path");
const { run } = require("./harness.cjs");

const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
const ok = (c, m) => { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); };

// The cross-language golden tile (2 points, rows 7 & 42) — reused as tile bytes.
const GOLDEN = "535154310100000002000000000000000000c03f000000c00000003f030000000700000000000000" +
  "0000803e000040bf0401000070110100";
const golden = Buffer.from(GOLDEN, "hex");
const goldenAB = () => golden.buffer.slice(golden.byteOffset, golden.byteOffset + golden.byteLength);

const calls = [];
const fetch = (url, init) => {
  calls.push({ url, init });
  const j = (o) => ({ ok: true, json: () => Promise.resolve(o) });
  if (url.endsWith("/manifest")) return Promise.resolve(j({
    title: "demo", total_points: 50, surface_radius: 1,
    bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
    stats: { projection_kind: "pca", evr: 0.33, evr_label: "PCA variance" },
    overlays: [], palette: [{ name: "A", color: "#5cc8ff", count: 30 }, { name: "B", color: "#ff8a65", count: 20 }],
    lod: { levels: 4, base_budget: 20000 },
  }));
  if (url.indexOf("/tiles") >= 0) return Promise.resolve({ ok: true, arrayBuffer: () => Promise.resolve(goldenAB()) });
  if (url.endsWith("/diagnostics")) return Promise.resolve(j({
    projection_kind: "pca", evr: 0.33, evr_label: "PCA variance", total_points: 50,
    warnings: [{ message: "EVR low", severity: "info", evr: 0.33 }],
    certainty: { bins: [3, 1, 0, 5], min: 0.1, max: 0.9 }, intensity: { bins: [2, 2], min: 0, max: 2 },
    outliers: [{ row: 7, label: "p7", category: "A", certainty: 0.12 }],
  }));
  if (url.endsWith("/points")) return Promise.resolve(j({ points: [{ row: 7, label: "p7", cat: 0, category: "A", certainty: 0.5, intensity: 1, vector: [0.1, -0.2, 0.3], x: 1, y: 0, z: 0, r: 1, theta: 0, phi: 1.5 }] }));
  if (url.endsWith("/nearest")) return Promise.resolve(j({ neighbors: [{ row: 42, similarity: 0.91 }] }));
  return Promise.resolve({ ok: false, status: 404 });
};

const vt = run(VIEWER, { points: [{ x: 1, y: 0, z: 0, cat: "x" }] }, { globals: { fetch } });

(async () => {
  ok(typeof vt.connectToServer === "function", "connectToServer exported");
  const streamer = await vt.connectToServer("http://srv");
  ok(streamer && vt.streamStreamer === streamer, "connectToServer installs a live streamer");
  ok(streamer.manifest && streamer.manifest.total_points === 50, "manifest fetched + applied");
  ok(calls.some((c) => c.url === "http://srv/manifest"), "fetched /manifest");
  ok(calls.some((c) => c.url.indexOf("/tiles") >= 0), "fetched a tile (base)");
  ok(calls.some((c) => c.url.endsWith("/diagnostics")), "loaded /diagnostics on connect");
  ok(streamer.loadedKeys().includes("base"), "base tile is loaded into the working set");

  // Inspect a streamed row → /points + /nearest.
  const before = calls.length;
  await vt.selectStreamRow(7);
  ok(calls.filter((c) => c.url.endsWith("/points")).length >= 1, "selectStreamRow fetches /points");
  ok(calls.some((c) => c.url.endsWith("/nearest") && JSON.parse(c.init.body).row === 7), "selectStreamRow fetches /nearest for the row");
  ok(calls.length > before, "inspect issued network calls");

  // renderDiagnostics tolerates a payload without throwing (DOM stub absorbs it).
  let threw = false;
  try { vt.renderDiagnostics({ projection_kind: "pca", evr: 0.5, total_points: 9, warnings: [], certainty: { bins: [1], min: 0, max: 1 }, intensity: { bins: [1], min: 0, max: 1 }, outliers: [] }); }
  catch (e) { threw = true; }
  ok(!threw, "renderDiagnostics runs without throwing");

  console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
  process.exit(fails === 0 ? 0 : 1);
})();
