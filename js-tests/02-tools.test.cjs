// Offline tools (re-ported onto the createViewer factory): the great-circle
// ruler (0/90/180° + on-shell antipodal arc), PNG snapshot export, and the
// shareable #v= hash round-trip including view SETTINGS (scale/radial/spread)
// and the ruler tool flag. The XSS/overlay-coercion invariants moved to suite 01.
const path = require("path");
const { run } = require("./harness.cjs");
const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
function ok(c, m) { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); }
function near(a, b, e, m) { ok(Math.abs(a - b) < (e || 1e-6), m + " (" + a + " vs " + b + ")"); }

const boot = {
  title: "t", surface_radius: 1, stats: { projection_kind: "pca", evr: 0.5, evr_label: "x" }, overlays: [],
  points: [{ x: 1, y: 0, z: 0, r: 1, theta: 0, phi: 1.5708, cat: "A", label: "a" },
           { x: 0, y: 1, z: 0, r: 1, theta: 1.5708, phi: 1.5708, cat: "B", label: "b" }],
};
const vt = run(VIEWER, boot);
ok(vt && typeof vt.setRuler === "function", "module booted; tools exported");
ok(vt.rulerOn === false, "ruler starts off");

// ── Ruler great-circle math ──────────────────────────────────────────────
vt.setRuler(true); ok(vt.rulerOn === true, "setRuler(true) enables ruler");
vt.rulerAddPick([3, 0, 0]); ok(vt.rulerPicks.length === 1, "first pick registered");
vt.rulerAddPick([0, 5, 0]); ok(vt.rulerPicks.length === 2, "second pick registered");
near(vt.rulerLast.deg, 90, 1e-4, "orthogonal dirs → 90°");
near(vt.rulerLast.chord, Math.SQRT2, 1e-4, "chord of 90° = √2");
vt.rulerAddPick([1, 0, 0]); ok(vt.rulerPicks.length === 1, "third pick resets to a new measurement");
vt.rulerAddPick([-1, 0, 0]); near(vt.rulerLast.deg, 180, 1e-4, "antipodal dirs → 180°");
near(vt.rulerLast.chord, 2, 1e-4, "chord of 180° = 2");
vt.rulerAddPick([1, 0, 0]); vt.rulerAddPick([1, 0, 0]); near(vt.rulerLast.deg, 0, 1e-4, "same dir → 0°");
vt.setRuler(false); ok(vt.rulerOn === false && vt.rulerPicks.length === 0, "setRuler(false) clears picks");

// ── Antipodal arc must stay on the shell ─────────────────────────────────
vt.setRuler(true);
vt.rulerAddPick([1, 0, 0]); vt.rulerAddPick([-1, 0, 0]);
near(vt.rulerLast.deg, 180, 1e-4, "antipodal picks → 180°");
const arc = vt.rulerGroup.children.find((c) => c.geometry && c.geometry._pts && c.geometry._pts.length > 2);
ok(arc, "antipodal arc line created");
const onShell = arc && arc.geometry._pts.every((p) => Math.abs(Math.hypot(p.x, p.y, p.z) - vt.SR) < 1e-6);
ok(onShell, "antipodal arc stays on the shell (no collapse through the interior)");
const mids = arc ? arc.geometry._pts[Math.floor(arc.geometry._pts.length / 2)] : { x: 0, y: 0, z: 0 };
ok(arc && Math.hypot(mids.x, mids.y, mids.z) > 0.99 * vt.SR, "antipodal arc midpoint is on the shell, not at the origin");
vt.setRuler(false);
// coincident picks → a visible (>=2 vertex) line
vt.setRuler(true); vt.rulerAddPick([1, 0, 0]); vt.rulerAddPick([1, 0, 0]);
const seg = vt.rulerGroup.children.find((c) => c.geometry && c.geometry._pts);
ok(seg && seg.geometry._pts.length >= 2, "coincident picks → >=2-vertex visible line");
vt.setRuler(false);

// ── PNG export ───────────────────────────────────────────────────────────
vt.exportPNG();
ok(vt.downloads.length >= 1 && vt.downloads[vt.downloads.length - 1].download === "sphereql-view.png",
  "exportPNG triggers a sphereql-view.png download");
ok(/^data:image\/png/.test(vt.downloads[vt.downloads.length - 1].href), "PNG download is a data:image/png URL");

// ── Shareable view hash round-trip (settings + camera) ───────────────────
vt.applyScale(33); vt.radialG = 2.5; vt.spreadF = 1.8;
vt.camera.position.set(5, 6, 7); vt.controls.target.set(1, 2, 3);
vt.shareLink();
// perturb, then restore from the hash
vt.applyScale(7); vt.radialG = 1; vt.spreadF = 1;
vt.camera.position.set(0, 0, 0); vt.controls.target.set(0, 0, 0);
const ret = vt.applyViewHash();
ok(ret == null, "offline applyViewHash returns undefined (not a promise)");
near(vt.curScale, 33, 1e-9, "scale restored from hash");
near(vt.radialG, 2.5, 1e-9, "radial restored from hash");
near(vt.spreadF, 1.8, 1e-9, "domain spread restored from hash");
near(vt.camera.position.x, 5, 1e-9, "camera x restored");
near(vt.camera.position.z, 7, 1e-9, "camera z restored");
near(vt.controls.target.y, 2, 1e-9, "target y restored");

// ── rulerOn resets on scene swap ─────────────────────────────────────────
vt.setRuler(true); ok(vt.rulerOn === true, "ruler armed before swap");
vt.rebuild(vt.parseScene([{ x: 1, y: 0, z: 0, cat: "Z" }, { x: 0, y: 1, z: 0, cat: "Z" }]));
ok(vt.rulerOn === false, "rebuild disarms the ruler");

console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
process.exit(fails === 0 ? 0 : 1);
