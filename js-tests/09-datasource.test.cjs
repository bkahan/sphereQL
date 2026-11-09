// DataSource seam — InlineSource: the offline blob served through the same
// async interface a streaming server would. Also asserts the boot now routes
// through InlineSource without changing what gets rendered.
const path = require("path");
const { run } = require("./harness.cjs");

const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
const ok = (c, m) => { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); };

const boot = {
  title: "demo", surface_radius: 2,
  stats: { projection_kind: "pca", evr: 0.5, evr_label: "PCA variance" },
  overlays: [{ kind: "centroid", pos: [0, 0, 1], label: "c", color: "#fff" }],
  points: [
    { id: "a", x: 2, y: 0, z: 0, cat: "sci", label: "alpha", certainty: 0.9, intensity: 1.2 },
    { id: "b", x: 0, y: 2, z: 0, cat: "sci", label: "beta" },
    { id: "c", x: 0, y: 0, z: 2, cat: "art", label: "gamma" },
  ],
};
const vt = run(VIEWER, boot);

(async () => {
  // Boot routes through InlineSource, rendering the same scene object.
  ok(vt.dataSource && typeof vt.dataSource.manifest === "function", "boot installed a dataSource");
  ok(vt.dataSource.scene === boot, "InlineSource wraps the inline scene (render path unchanged)");
  ok(vt.N === 3, "rebuild(dataSource.scene) rendered all 3 points");

  const src = new vt.InlineSource(boot);

  // manifest()
  const m = await src.manifest();
  ok(m.total_points === 3, "manifest.total_points = 3");
  ok(m.surface_radius === 2, "manifest.surface_radius honoured");
  ok(m.palette.length === 2 && m.palette[0].name === "art" && m.palette[1].name === "sci",
    "palette is sorted-unique categories (art, sci)");
  ok(m.palette.reduce((s, c) => s + c.count, 0) === 3, "palette counts sum to total");
  ok(m.overlays.length === 1 && m.overlays[0].kind === "centroid", "manifest carries overlays");
  ok(Array.isArray(m.bounds.min) && m.bounds.max[0] === 2, "manifest carries xyz bounds");

  // tiles()
  const tile = await src.tiles({});
  ok(tile.count === 3, "tiles() (no budget) returns all points");
  const catOfRow = {}; for (let i = 0; i < tile.count; i++) catOfRow[tile.rows[i]] = tile.cats[i];
  ok(catOfRow[2] === 0, "row 2 (art) → cat id 0 (palette order)");
  ok(catOfRow[0] === 1 && catOfRow[1] === 1, "rows 0,1 (sci) → cat id 1");
  ok((await src.tiles({ budget: 2 })).count <= 2, "budget thins the tile (≤2)");

  // pointMeta()
  const meta = await src.pointMeta([0, 2, 99]);
  ok(meta.length === 2, "pointMeta returns 2 (out-of-range row 99 skipped)");
  ok(meta[0].label === "alpha" && meta[0].category === "sci", "meta carries label + category");
  ok(meta[0].certainty === 0.9 && meta[0].intensity === 1.2, "meta carries quality signals");
  ok(Math.abs(meta[0].x - 2) < 1e-9 && isFinite(meta[0].r), "meta carries coordinates");

  // nearest()
  const nn = await src.nearest({ row: 0 }, 5);
  ok(nn.length === 2 && nn.every((h) => h.row !== 0), "nearest by row returns the others, excludes self");
  ok(nn[0].similarity >= nn[1].similarity, "nearest sorted descending by similarity");
  const nv = await src.nearest({ vector: [1, 0, 0] }, 1);
  ok(nv.length === 1 && nv[0].row === 0, "nearest by +x vector → the +x point (row 0)");
  ok((await src.nearest({}, 3)).length === 0, "nearest with neither row nor vector → empty");

  console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
  process.exit(fails === 0 ? 0 : 1);
})();
