// Phase D streaming-debugger flow, headless: drive connectToServer against a
// mock fetch and verify the wiring — manifest + base tile + diagnostics are
// fetched, the streamer is live with the base tile rendered, inspecting a row
// issues the /points + /nearest fetches, renderDiagnostics survives a full
// payload and null, applyStreamFilter pushes a server-side filter, and
// disconnect tears the stream down. (The DOM stub returns ephemeral elements, so
// this asserts on fetch traffic + streamer state, not rendered markup — the
// visuals + the escHtml/safeColor XSS hardening are browser/review-validated.)
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
  if (url.endsWith("/points")) return Promise.resolve(j({ points: [{ row: 7, label: "p7", cat: 0, category: "A", certainty: 0.5, intensity: 1, x: 1, y: 0, z: 0, vector: [0.1, -0.2, 0.3] }] }));
  if (url.endsWith("/nearest")) return Promise.resolve(j({ neighbors: [{ row: 42, similarity: 0.91 }] }));
  return Promise.resolve({ ok: false, status: 404 });
};

const vt = run(VIEWER, { points: [{ x: 1, y: 0, z: 0, cat: "x" }] }, { globals: { fetch } });

(async () => {
  ok(typeof vt.connectToServer === "function", "connectToServer on the instance");
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

  // renderDiagnostics tolerates a full payload AND null without throwing.
  let threw = false;
  try {
    vt.renderDiagnostics({ projection_kind: "pca", evr: 0.5, total_points: 9, warnings: [{ message: "<img src=x onerror=alert(1)>", severity: "critical" }], certainty: { bins: [1], min: 0, max: 1 }, intensity: { bins: [1], min: 0, max: 1 }, outliers: [{ row: 7, label: "p7", certainty: 0.1 }] });
    vt.renderDiagnostics(null);
  } catch (e) { threw = true; }
  ok(!threw, "renderDiagnostics runs (full payload incl. hostile warning + null) without throwing");

  // applyStreamFilter pushes a server-side filter through the streamer.
  if (typeof vt.applyStreamFilter === "function" && typeof vt.streamStreamer.setFilter === "function") {
    let filterThrew = false;
    try { vt.applyStreamFilter(); } catch (e) { filterThrew = true; }
    ok(!filterThrew, "applyStreamFilter runs without throwing");
  }

  // Disconnect tears the stream down.
  vt.disconnectServer();
  ok(vt.streamStreamer == null, "disconnectServer clears the streamer");

  // Zombie-stream regression: rebuild() (e.g. drag-drop / Open Scene while
  // streaming) must tear down the active stream via teardown→disconnectServer,
  // else getHovered/updateHover stay on the streaming branch and the new offline
  // scene is unpickable.
  await vt.connectToServer("http://srv");
  ok(!!vt.streamStreamer, "reconnected for the rebuild-teardown check");
  vt.rebuild(vt.parseScene([{ x: 1, y: 0, z: 0, cat: "q" }, { x: 0, y: 1, z: 0, cat: "q" }]));
  ok(vt.streamStreamer == null, "rebuild() disconnects the active stream (no zombie)");
  ok(vt.dataSource && vt.dataSource.scene && vt.dataSource.scene.points.length === 2, "rebuild() reinstalls an InlineSource for the offline scene");

  // A /nearest row is server data: it must reach the neighbor list as a number,
  // never as markup (a shared #v= link can point the viewer at any server).
  {
    const hostile = (url, init) => url.endsWith("/nearest")
      ? Promise.resolve({ ok: true, json: () => Promise.resolve({ neighbors: [{ row: '1" onmouseover="alert(1)', similarity: 0.5 }] }) })
      : fetch(url, init);
    const hv = run(VIEWER, { points: [] }, { globals: { fetch: hostile } });
    const list = hv.body.querySelector();
    const lookup = hv.body.querySelector;
    hv.body.querySelector = (sel) => (sel === "[data-info-neighbors]" ? list : lookup(sel));
    await hv.connectToServer("http://srv");
    await hv.selectStreamRow(7);
    hv.body.querySelector = lookup;
    ok(list.innerHTML.includes('data-row="NaN"') && !list.innerHTML.includes('onmouseover="'), "a hostile neighbor row can't inject markup");
  }

  // Re-projection, latest wins: a slower, older response is ignored.
  {
    const held = [];
    const deferred = (url, init) => url.endsWith("/reproject")
      ? new Promise((res) => held.push({ kind: JSON.parse(init.body).projection, res }))
      : fetch(url, init);
    const rv = run(VIEWER, { points: [] }, { globals: { fetch: deferred } });
    await rv.connectToServer("http://srv");
    const pick = rv.window.__sqServerReproject;
    pick("UmapSphere"); pick("KernelPca");
    const answer = (i) => held[i].res({ ok: true, json: () => Promise.resolve({
      title: "t", total_points: 50, palette: [], stats: { projection_kind: held[i].kind },
      surface_radius: 2 + i, overlays: [{ kind: "centroid", pos: [0, 0, 2 + i], label: "A", color: "#ff8a65" }],
    }) });
    answer(0); await new Promise((r) => setImmediate(r));
    ok(rv.streamStreamer.manifest.stats.projection_kind === "pca", "the superseded re-projection isn't applied");
    answer(1); await new Promise((r) => setImmediate(r));
    ok(rv.streamStreamer.manifest.stats.projection_kind === "kernel_pca", "the newest re-projection is applied");
    ok(rv.SR === 3 && rv.overlayKinds.includes("centroid"), "the globe and overlays are redrawn for the new projection");
    ok(rv.dataSource && rv.dataSource.base === "http://srv", "the stream survives the redraw");
  }

  // Disconnecting stops the tile-decode worker and frees its blob URL.
  {
    const workers = [], revoked = [];
    let decodeFn = null;
    function FakeWorker(url) {
      this.url = url; this.terminated = false; workers.push(this);
      this.postMessage = (m) => setImmediate(() => this.onmessage({ data: { id: m.id, d: decodeFn(m.buf) } }));
      this.terminate = () => { this.terminated = true; };
    }
    const wv = run(VIEWER, { points: [] }, { globals: { fetch, Worker: FakeWorker, Blob: function () {},
      URL: { createObjectURL: () => "blob:decoder", revokeObjectURL: (u) => revoked.push(u) } } });
    decodeFn = wv.decodeTile;
    await wv.connectToServer("http://srv");
    ok(workers.length === 1 && wv.streamStreamer.loadedKeys().includes("base"), "tiles decode through the worker");
    wv.disconnectServer();
    ok(workers[0].terminated && revoked.includes("blob:decoder"), "disconnect terminates the worker and revokes its URL");
  }

  console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
  process.exit(fails === 0 ? 0 : 1);
})();
