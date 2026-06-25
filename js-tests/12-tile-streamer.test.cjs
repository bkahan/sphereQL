// Phase C streaming orchestration: TileStreamer turns camera motion into tile
// requests and manages a bounded working set of per-tile meshes via an injected
// sink. This locks the headless-testable core — camera→request mapping, the
// persistent base + detail tiles, dedup, LRU eviction, and load cancellation —
// against mock source/sink. The actual THREE per-tile rendering (the sink impl)
// is browser-validated.
const path = require("path");
const { run } = require("./harness.cjs");

const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
const ok = (c, m) => { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); };

const vt = run(VIEWER, { points: [{ x: 1, y: 0, z: 0, cat: "A" }] });
ok(typeof vt.TileStreamer === "function", "TileStreamer exported");

function tile(rows) { const n = rows.length; return { count: n, positions: new Float32Array(n * 3), cats: new Uint16Array(n), rows: Uint32Array.from(rows) }; }
function mockSink() {
  const tiles = new Map(), adds = [], removes = [];
  return {
    tiles, adds, removes,
    addTile(k, d) { tiles.set(k, d); adds.push(k); },
    removeTile(k) { if (tiles.delete(k)) removes.push(k); },
    clear() { tiles.clear(); },
  };
}
function mockSource() {
  const calls = [];
  return {
    calls,
    manifest: async () => ({ total_points: 1000, surface_radius: 1, lod: { levels: 4, base_budget: 100 }, palette: [], stats: {}, overlays: [] }),
    tiles: async (req) => { calls.push(req); return tile(req.half_angle >= Math.PI ? [0, 1, 2] : [10, 11]); },
  };
}

(async () => {
  // start() loads the manifest + the persistent coarse base tile.
  {
    const src = mockSource(), sink = mockSink(), s = new vt.TileStreamer(src, sink, { maxDetail: 2 });
    const m = await s.start();
    ok(m.total_points === 1000, "start() returns the manifest");
    ok(sink.tiles.has("base"), "start() loads the base tile (whole sphere, LOD 0)");
    ok(src.calls.length === 1 && src.calls[0].half_angle >= Math.PI && src.calls[0].lod === 0, "base request is a full-sphere LOD-0 cone");
    ok(s.loadedKeys().includes("base"), "base is in the loaded set");
  }

  // lodFor: bounded and non-increasing as the camera retreats.
  {
    const s = new vt.TileStreamer(mockSource(), mockSink(), {});
    s.near = 1.05; s.far = 8; s.lodLevels = 4;
    ok(s.lodFor(1.0) === 3, "at/inside the shell → finest LOD");
    ok(s.lodFor(100) === 0, "very far → coarsest LOD");
    ok(s.lodFor(2) >= s.lodFor(6), "LOD is non-increasing with distance");
    const all = [s.lodFor(1), s.lodFor(3), s.lodFor(8), s.lodFor(50)];
    ok(all.every((l) => l >= 0 && l <= 3), "LOD stays within [0, levels-1]");
  }

  // requestFor + keyFor: sensible cone, stable quantized key.
  {
    const s = new vt.TileStreamer(mockSource(), mockSink(), {});
    s.near = 1.05; s.far = 8; s.lodLevels = 4;
    const r = s.requestFor({ theta: 1.0, phi: 1.4, dist: 3 });
    ok(r.half_angle > 0 && r.half_angle <= Math.PI, "request half_angle in (0, π]");
    ok(r.theta === 1.0 && Math.abs(r.phi - 1.4) < 1e-9, "request aims where the camera looks");
    ok(s.keyFor(s.requestFor({ theta: 1.0, phi: 1.4, dist: 3 })) === s.keyFor(s.requestFor({ theta: 1.001, phi: 1.4, dist: 3 })),
      "tiny camera jitter quantizes to the same tile key");
  }

  // update(): loads a detail tile, dedups an identical viewport, evicts by LRU.
  {
    const src = mockSource(), sink = mockSink(), s = new vt.TileStreamer(src, sink, { maxDetail: 2 });
    await s.start();
    const baseCalls = src.calls.length; // 1
    const k1 = await s.update({ theta: 0, phi: 1.5, dist: 1.2 });
    ok(k1 && sink.tiles.has(k1), "update() loads a detail tile and adds it to the sink");
    ok(src.calls.length === baseCalls + 1, "detail tile fetched once");
    await s.update({ theta: 0, phi: 1.5, dist: 1.2 });
    ok(src.calls.length === baseCalls + 1, "identical viewport dedups — no refetch");
    // Three more distinct viewports; maxDetail=2 → detail count capped, base kept.
    await s.update({ theta: 0.5, phi: 1.5, dist: 1.2 });
    await s.update({ theta: 1.0, phi: 1.5, dist: 1.2 });
    await s.update({ theta: 1.5, phi: 1.5, dist: 1.2 });
    const detail = s.loadedKeys().filter((k) => k !== "base");
    ok(detail.length <= 2, "detail working set capped at maxDetail (" + detail.length + ")");
    ok(sink.tiles.has("base"), "base tile is never evicted");
    ok(sink.removes.length >= 1, "evicted detail tiles are removed from the sink");
  }

  // Cancellation: a load that resolves after its entry is cleared is dropped.
  {
    let release;
    const sink = mockSink();
    const src = {
      manifest: async () => ({ surface_radius: 1, lod: { levels: 4, base_budget: 100 } }),
      tiles: (req) => req.half_angle >= Math.PI
        ? Promise.resolve(tile([0]))
        : new Promise((res) => { release = () => res(tile([5])); }),
    };
    const s = new vt.TileStreamer(src, sink, {});
    await s.start();
    const addsBefore = sink.adds.length;
    const p = s.update({ theta: 0, phi: 1.5, dist: 1.2 }); // starts loading (deferred)
    s.clear();      // entry removed while the fetch is in flight
    release();      // fetch now resolves
    const k = await p;
    ok(k === null, "a load that resolves after clear() returns null");
    ok(sink.adds.length === addsBefore, "the dropped tile is never added to the sink");
  }

  // setFilter merges cats/min_certainty into every tile request and reloads the
  // base so the whole streamed view reflects the filter.
  {
    const src = mockSource(), sink = mockSink(), s = new vt.TileStreamer(src, sink, {});
    await s.start();
    const before = src.calls.length;
    await s.setFilter({ cats: [0, 2], minCertainty: 0.5 });
    ok(src.calls.length > before, "setFilter reloads the base tile");
    const baseReq = src.calls[src.calls.length - 1];
    ok(baseReq.cats === "0,2" && baseReq.min_certainty === 0.5, "base request carries the filter params");
    const r = s.requestFor({ theta: 0.2, phi: 1.5, dist: 1.2 });
    ok(r.cats === "0,2" && r.min_certainty === 0.5, "detail requests carry the filter params");
    await s.setFilter({});
    ok(s.requestFor({ theta: 0, phi: 1.5, dist: 1.2 }).cats === undefined, "cleared filter → no cats param");
  }

  // tileMeshSink: builds a THREE.Points per tile with palette colors, sizes, and
  // a pick id baked from the GLOBAL row (so picking resolves a row across tiles).
  {
    const group = { children: [], add(c) { this.children.push(c); }, remove(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); } };
    const palette = [{ name: "A", color: "#ff0000", count: 1 }, { name: "B", color: "#00ff00", count: 1 }];
    const sink = vt.tileMeshSink(group, palette, { dispose() {} });
    sink.addTile("t1", { count: 2, positions: Float32Array.from([1, 0, 0, 0, 1, 0]), cats: Uint16Array.from([0, 1]), rows: Uint32Array.from([7, 42]) });
    ok(group.children.length === 1 && sink.count() === 1, "addTile adds one mesh to the group");
    const geo = group.children[0].geometry;
    ok(geo.attrs.position.array.length === 6, "position attribute holds the 2 streamed points");
    ok(geo.attrs.size.array.length === 2 && geo.attrs.color.array.length === 6, "size + color attributes sized to the tile");
    const pa = geo.attrs.aPickColor.array;
    ok(vt.pickDecode(Math.round(pa[0] * 255), Math.round(pa[1] * 255), Math.round(pa[2] * 255)) === 7, "point 0 pick id = its global row (7)");
    ok(vt.pickDecode(Math.round(pa[3] * 255), Math.round(pa[4] * 255), Math.round(pa[5] * 255)) === 42, "point 1 pick id = its global row (42)");
    sink.addTile("t1", { count: 1, positions: Float32Array.from([0, 0, 1]), cats: Uint16Array.from([0]), rows: Uint32Array.from([9]) });
    ok(group.children.length === 1, "re-adding the same key replaces (not duplicates) the mesh");
    sink.removeTile("t1");
    ok(group.children.length === 0 && sink.count() === 0, "removeTile removes the mesh from the group");
  }

  // connectToServer / disconnectServer are exported (browser-validated wiring).
  ok(typeof vt.connectToServer === "function" && typeof vt.disconnectServer === "function", "connectToServer/disconnectServer exported");

  // safeColor: accept real color literals from a server palette, reject CSS-
  // attribute injection (the stream legend interpolates color into style="").
  ok(vt.safeColor("#5cc8ff") === "#5cc8ff" && vt.safeColor("#fff") === "#fff", "hex colors pass");
  ok(vt.safeColor("rgb(10, 20, 30)") === "rgb(10, 20, 30)" && vt.safeColor("teal") === "teal", "rgb()/named colors pass");
  ok(vt.safeColor("red;background:url(http://evil/x)") === "#90a4ae", "CSS injection falls back to a safe default");
  ok(vt.safeColor(undefined) === "#90a4ae" && vt.safeColor("</span>") === "#90a4ae", "non-color input falls back");

  console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
  process.exit(fails === 0 ? 0 : 1);
})();
