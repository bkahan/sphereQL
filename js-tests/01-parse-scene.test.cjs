const path = require("path");
const fs = require("fs");
const { run } = require("./harness.cjs");

const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
function ok(cond, msg) { if (!cond) { console.log("FAIL:", msg); fails++; } else console.log("ok  :", msg); }
function throws(fn, msg) { try { fn(); console.log("FAIL (no throw):", msg); fails++; } catch (e) { console.log("ok  :", msg, "→", e.message); } }

// Boot the module once with a small real scene (exercises full rebuild path).
const boot = {
  title: "boot", surface_radius: 1,
  stats: { projection_kind: "pca", evr: 0.5, evr_label: "x" },
  overlays: [{ kind: "centroid", pos: [0, 0, 1], label: "c", color: "#fff" }],
  points: [
    { id: "a", x: 1, y: 0, z: 0, r: 1, theta: 0, phi: 1.5708, cat: "A", label: "a" },
    { id: "b", x: 0, y: 1, z: 0, r: 1, theta: 1.5708, phi: 1.5708, cat: "B", label: "b" },
  ],
};
const vt = run(VIEWER, boot);
ok(vt && typeof vt.parseScene === "function", "module booted; parseScene exported");
ok(vt.N === 2, "boot rebuild loaded 2 points (N=" + vt.N + ")");
ok(vt.overlayKinds.indexOf("centroid") >= 0, "boot overlay kind present");

// parseScene: full Scene round-trips.
const full = vt.parseScene(boot);
ok(full.points.length === 2, "full scene → 2 points");
ok(full.points[0].id === "a", "id preserved");
ok(full.stats.projection_kind === "pca", "stats preserved");

// parseScene: bare points array accepted.
const bare = vt.parseScene([{ x: 1, y: 2, z: 2 }, { x: -1, y: 0, z: 0 }]);
ok(bare.points.length === 2, "bare array → 2 points");
ok(Math.abs(bare.points[0].r - 3) < 1e-9, "r derived from xyz (3)");
ok(bare.points[0].theta >= 0 && bare.points[0].theta < 2 * Math.PI, "theta wrapped to [0,2π)");
ok(bare.points[0].phi >= 0 && bare.points[0].phi <= Math.PI, "phi in [0,π]");
ok(bare.stats.projection_kind === "imported", "default stats label");
ok(bare.surface_radius > 0 && isFinite(bare.surface_radius), "surface_radius derived finite");

// surface_radius must exclude origin/zero-norm points (parity with Rust
// surface_radius_for) — else a dropped scene with origin points lands on a
// different shell than baked. norms=[0,2,8] → filtered [2,8] → median idx1 = 8.
const withOrigin = vt.parseScene({ points: [{ x: 0, y: 0, z: 0, cat: "o" }, { x: 2, y: 0, z: 0, cat: "a" }, { x: 8, y: 0, z: 0, cat: "b" }] });
ok(Math.abs(withOrigin.surface_radius - 8) < 1e-9, "surface_radius ignores origin points (8, matches Rust) got " + withOrigin.surface_radius);

// parseScene: spherical-only points → xyz derived (convention check).
const sph = vt.parseScene({ points: [{ r: 2, theta: 0, phi: Math.PI / 2, cat: "S", label: "s" }] });
ok(Math.abs(sph.points[0].x - 2) < 1e-9 && Math.abs(sph.points[0].y) < 1e-9 && Math.abs(sph.points[0].z) < 1e-9,
  "rθφ→xyz: (2,0,π/2)→(2,0,0)  got (" + sph.points[0].x.toFixed(3) + "," + sph.points[0].y.toFixed(3) + "," + sph.points[0].z.toFixed(3) + ")");

// parseScene: drops non-placeable points, keeps placeable ones.
const mixed = vt.parseScene({ points: [{ x: NaN, y: 1, z: 1 }, { foo: 1 }, { x: 1, y: 1, z: 1 }] });
ok(mixed.points.length === 1, "non-finite / empty points dropped (kept " + mixed.points.length + ")");

// parseScene: hard errors.
throws(() => vt.parseScene(null), "null rejected");
throws(() => vt.parseScene({ nope: 1 }), "missing points[] rejected");
throws(() => vt.parseScene({ points: [{ foo: 1 }] }), "all-invalid points rejected");

// rebuild with an imported (parsed) scene should swap cleanly to new N.
vt.rebuild(vt.parseScene([{ x: 1, y: 0, z: 0, cat: "Z" }, { x: 0, y: 1, z: 0, cat: "Z" }, { x: 0, y: 0, z: 1, cat: "Q" }]));
ok(vt.N === 3, "rebuild swapped to 3 points (N=" + vt.N + ")");
ok(vt.catSet.length === 2 && vt.catSet.indexOf("Q") >= 0, "rebuild recomputed categories [" + vt.catSet + "]");

// ── stats count fields coerced to numbers — no HTML smuggling (finding #5) ─
// (moved here from the deleted 02-tools suite — render-path invariants that only
//  need parseScene/rebuild.)
const evil = vt.parseScene({ points: [{ x: 1, y: 0, z: 0 }], stats: { sampled_from: "<img src=x onerror=alert(1)>", dropped_nonfinite: "5" } });
ok(evil.stats.sampled_from === undefined, "malicious sampled_from string coerced away");
ok(evil.stats.dropped_nonfinite === 5, "numeric-string dropped_nonfinite coerced to 5");

// ── malformed overlays filtered out (finding #6) ─────────────────────────
const ov = vt.parseScene({ points: [{ x: 1, y: 0, z: 0 }], overlays: [{ kind: "centroid", pos: [0, 0, 1], label: "c" }, { nope: 1 }, null, { kind: 42 }] });
ok(ov.overlays.length === 1, "only well-formed overlays survive parseScene (" + ov.overlays.length + ")");
// ...and a malformed overlay reaching rebuild() doesn't abort the whole scene
vt.rebuild({ title: "t", surface_radius: 1, stats: {}, points: [{ x: 1, y: 0, z: 0, r: 1, theta: 0, phi: 1.5, cat: "A", label: "a" }], overlays: [{ kind: "bridge" /* missing from/to */ }, { kind: "centroid", pos: [0, 0, 1], label: "c" }] });
ok(vt.N === 1, "rebuild survives a malformed overlay (scene still loads)");

// Optional integration check: round-trip a REAL emitted scene through
// parseScene. Point SPHEREQL_EMIT_HTML at an emitted page to enable it, e.g.
//   cargo run -p sphereql-examples --example visualize_corpus -- --out /tmp/v.html
//   SPHEREQL_EMIT_HTML=/tmp/v.html node js-tests/01-parse-scene.test.cjs
// Skipped (not failed) when unset, so CI stays self-contained.
const emitHtml = process.env.SPHEREQL_EMIT_HTML;
if (emitHtml && fs.existsSync(emitHtml)) {
  const html = fs.readFileSync(emitHtml, "utf8");
  const m = "<script>\nconst D=";
  const di = html.indexOf(m) + m.length;
  const line = html.slice(di, html.indexOf("\n", di));
  const realScene = JSON.parse(line.replace(/;$/, ""));
  const reparsed = vt.parseScene(realScene);
  ok(reparsed.points.length === realScene.points.length, "real emitted scene round-trips (" + reparsed.points.length + " pts)");
} else {
  console.log("skip : real-emitted-scene round-trip (set SPHEREQL_EMIT_HTML to enable)");
}

console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
process.exit(fails === 0 ? 0 : 1);
