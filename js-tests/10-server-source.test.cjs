// DataSource seam — ServerSource + TileCache + the worker-decoder fallback,
// driven against a mock fetch (no socket). Verifies URL/method/body shaping,
// SQT1 tile decode over the wire, caching, and error propagation.
const path = require("path");
const { run } = require("./harness.cjs");

const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
const ok = (c, m) => { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); };

const GOLDEN =
  "535154310100000002000000000000000000c03f000000c00000003f030000000700000000000000" +
  "0000803e000040bf0401000070110100";
const golden = Buffer.from(GOLDEN, "hex");
const goldenAB = () => golden.buffer.slice(golden.byteOffset, golden.byteOffset + golden.byteLength);

// A recording fetch stub returning Response-likes from a router fn.
function makeFetch(router) {
  const calls = [];
  const fetch = (url, init) => { calls.push({ url, init }); return Promise.resolve(router(url, init)); };
  fetch.calls = calls;
  return fetch;
}

// Just enough IndexedDB for TileCache: one object store, async request events.
function fakeIndexedDB() {
  const data = new Map();
  const req = (fn) => { const r = {}; setImmediate(() => { r.result = fn(); r.onsuccess && r.onsuccess(); }); return r; };
  const store = { get: (k) => req(() => data.get(k)), put: (v, k) => req(() => { data.set(k, v); }), clear: () => req(() => data.clear()) };
  const db = { transaction: () => ({ objectStore: () => store }), createObjectStore() {} };
  return { data, open: () => req(() => db) };
}

const vt = run(VIEWER, { points: [{ x: 1, y: 0, z: 0, cat: "A", label: "a" }] });

(async () => {
  const fetch = makeFetch((url) => {
    if (url.endsWith("/manifest")) return { ok: true, json: () => Promise.resolve({ total_points: 3, palette: [] }) };
    if (url.indexOf("/tiles") >= 0) return { ok: true, arrayBuffer: () => Promise.resolve(goldenAB()) };
    if (url.endsWith("/points")) return { ok: true, json: () => Promise.resolve({ points: [{ row: 0, label: "p0" }] }) };
    if (url.endsWith("/nearest")) return { ok: true, json: () => Promise.resolve({ neighbors: [{ row: 5, similarity: 0.9 }] }) };
    if (url.endsWith("/diagnostics")) return { ok: true, json: () => Promise.resolve({ projection_kind: "pca", evr: 0.5, total_points: 3, warnings: [], certainty: { bins: [1, 2], min: 0, max: 1 }, outliers: [] }) };
    if (url.endsWith("/reproject")) return { ok: true, json: () => Promise.resolve({ title: "t", total_points: 3, stats: { projection_kind: "umap_sphere", evr: 0.4 } }) };
    return { ok: false, status: 404 };
  });

  const src = new vt.ServerSource("http://srv/", { fetch });
  ok(src.base === "http://srv", "trailing slash trimmed from base URL");

  const m = await src.manifest();
  ok(m.total_points === 3, "GET /manifest parsed");
  ok(fetch.calls.some((c) => c.url === "http://srv/manifest"), "manifest hit /manifest");

  const tile = await src.tiles({ theta: 1, half_angle: 0.5, budget: 100 });
  ok(tile.count === 2, "GET /tiles decoded from SQT1 (count=2)");
  const tcall = fetch.calls.find((c) => c.url.indexOf("/tiles") >= 0);
  ok(/theta=1/.test(tcall.url) && /half_angle=0.5/.test(tcall.url) && /budget=100/.test(tcall.url),
    "tile request carries cone + budget query params");

  const meta = await src.pointMeta([0]);
  ok(meta.length === 1 && meta[0].label === "p0", "POST /points → points[]");
  const pc = fetch.calls.find((c) => c.url.endsWith("/points"));
  ok(pc.init.method === "POST" && JSON.parse(pc.init.body).rows[0] === 0, "POST /points body carries rows");

  const nn = await src.nearest({ row: 2 }, 4);
  ok(nn.length === 1 && nn[0].row === 5, "POST /nearest → neighbors[]");
  const nc = fetch.calls.find((c) => c.url.endsWith("/nearest"));
  ok(JSON.parse(nc.init.body).row === 2 && JSON.parse(nc.init.body).k === 4, "POST /nearest body carries row + k");

  const diag = await src.diagnostics();
  ok(diag.projection_kind === "pca" && diag.certainty.bins.length === 2, "GET /diagnostics parsed");
  ok(fetch.calls.some((c) => c.url.endsWith("/diagnostics")), "diagnostics() hits /diagnostics");

  const rm = await src.reproject("umap_sphere");
  ok(rm.stats.projection_kind === "umap_sphere", "reproject() returns the re-projected manifest");
  const rc = fetch.calls.find((c) => c.url.endsWith("/reproject"));
  ok(rc.init.method === "POST" && JSON.parse(rc.init.body).projection === "umap_sphere", "POST /reproject body carries the projection kind");

  // non-ok response → throws.
  let threw = false;
  try { await new vt.ServerSource("http://srv", { fetch: makeFetch(() => ({ ok: false, status: 500 })) }).manifest(); }
  catch (e) { threw = true; }
  ok(threw, "a non-ok response rejects");

  // TileCache: identical tile request served from cache (one fetch); a
  // different key misses.
  const cache = new vt.TileCache({ indexedDB: null });
  const cf = makeFetch((url) => url.indexOf("/tiles") >= 0
    ? { ok: true, arrayBuffer: () => Promise.resolve(goldenAB()) }
    : { ok: false, status: 404 });
  const csrc = new vt.ServerSource("http://srv", { fetch: cf, cache });
  await csrc.tiles({ budget: 5 });
  await csrc.tiles({ budget: 5 });
  ok(cf.calls.length === 1, "TileCache serves an identical tiles() from cache (1 fetch)");
  await csrc.tiles({ budget: 6 });
  ok(cf.calls.length === 2, "a different tile key misses the cache (2 fetches)");

  // TileCache LRU eviction (memory tier).
  const lru = new vt.TileCache({ indexedDB: null, max: 2 });
  await lru.put("a", 1); await lru.put("b", 2); await lru.put("c", 3); // evicts "a"
  ok((await lru.get("a")) === null, "LRU evicts the oldest past max");
  ok((await lru.get("c")) === 3 && (await lru.get("b")) === 2, "LRU keeps the recent entries");

  // Regression: a worker-backed decode TRANSFERS (detaches) the buffer it is
  // given. Combined with a cache, the retained blob must NOT be the one
  // transferred, or the next cache hit decodes a detached 0-byte buffer.
  const tcache = new vt.TileCache({ indexedDB: null });
  let decodes = 0;
  const transferringDecode = (b) => {
    decodes++;
    const out = vt.decodeTile(b);        // read it first…
    structuredClone(0, { transfer: [b] }); // …then detach `b`, like postMessage transfer
    return out;
  };
  const tf = makeFetch((url) => url.indexOf("/tiles") >= 0
    ? { ok: true, arrayBuffer: () => Promise.resolve(goldenAB()) }
    : { ok: false, status: 404 });
  const tsrc = new vt.ServerSource("http://srv", { fetch: tf, cache: tcache, decode: transferringDecode });
  const t1 = await tsrc.tiles({ budget: 5 });
  const t2 = await tsrc.tiles({ budget: 5 }); // cache hit, after the buffer was transferred
  ok(t1.count === 2 && t2.count === 2, "worker-transfer decode + cache hit both decode (cached blob survives)");
  ok(decodes === 2 && tf.calls.length === 1, "second tiles() served from cache despite the transfer");

  // makeWorkerDecoder falls back to inline decode when Worker is unavailable.
  const dec = vt.makeWorkerDecoder();
  ok(typeof dec === "function", "makeWorkerDecoder returns a decode fn");
  ok((await dec(golden)).count === 2, "worker-decoder (inline fallback) decodes the golden tile");

  // Tile-cache keys carry the server and the manifest: another server, or the
  // same server after a re-projection, never gets a tile cached for another scene.
  {
    const shared = new vt.TileCache({ indexedDB: null });
    const manifestFetch = (projection) => makeFetch((url) => {
      if (url.endsWith("/manifest")) return { ok: true, json: () => Promise.resolve({ stats: { projection_kind: projection } }) };
      if (url.endsWith("/reproject")) return { ok: true, json: () => Promise.resolve({ stats: { projection_kind: "umap_sphere" } }) };
      return { ok: true, arrayBuffer: () => Promise.resolve(goldenAB()) };
    });
    const f1 = manifestFetch("pca"), s1 = new vt.ServerSource("http://one", { fetch: f1, cache: shared });
    const f2 = manifestFetch("pca"), s2 = new vt.ServerSource("http://two", { fetch: f2, cache: shared });
    await s1.manifest(); await s1.tiles({ budget: 5 });
    await s2.manifest(); await s2.tiles({ budget: 5 });
    ok(f2.calls.some((c) => c.url.indexOf("/tiles") >= 0), "another server misses the first one's tile");
    await s1.manifest(); await s1.tiles({ budget: 5 });
    const tilesFetched = () => f1.calls.filter((c) => c.url.indexOf("/tiles") >= 0).length;
    ok(tilesFetched() === 2, "re-scoping drops the old scene from memory, so it refetches");
    await s1.tiles({ budget: 5 });
    ok(tilesFetched() === 2, "same server + same manifest hits the cache");
    await s1.reproject("umap_sphere"); await s1.tiles({ budget: 5 });
    ok(tilesFetched() === 3, "after a re-projection the same tile request misses");
  }

  // IndexedDB keeps one scene: switching scope clears the other scene's tiles.
  {
    const idb = fakeIndexedDB();
    const c1 = new vt.TileCache({ indexedDB: idb });
    await c1.scope("A"); await c1.put("A|t", 1);
    const c2 = new vt.TileCache({ indexedDB: idb }); // a reload, same scene
    await c2.scope("A");
    ok((await c2.get("A|t")) === 1, "a reload under the same scene keeps its tiles");
    await c2.scope("B");
    ok(!idb.data.has("A|t"), "switching scene clears the old scene from IndexedDB");
    await c2.put("B|t", 2);
    ok((await c2.get("B|t")) === 2, "the new scene caches normally");
  }

  console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
  process.exit(fails === 0 ? 0 : 1);
})();
