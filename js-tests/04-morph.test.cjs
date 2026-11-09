const path = require("path");
const { run } = require("./harness.cjs");
const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
const ok = (c, m) => { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); };
const near = (a, b, e, m) => ok(Math.abs(a - b) < (e || 1e-6), m + " (" + (+a).toFixed(4) + " vs " + (+b).toFixed(4) + ")");

// Scene A: s-0 at [1,0,0], s-1 at [0,0,1] (s-1 has no B match).
const A = {
  title: "A", surface_radius: 1, stats: { projection_kind: "pca", evr: 0.5, evr_label: "x" }, overlays: [],
  points: [
    { id: "s-0", x: 1, y: 0, z: 0, r: 1, theta: 0, phi: 1.5708, cat: "C", label: "p0" },
    { id: "s-1", x: 0, y: 0, z: 1, r: 1, theta: 0, phi: 0, cat: "C", label: "p1" },
  ],
};
// Scene B: s-0 moved to [0,2,0] (dir [0,1,0], r=2). No s-1.
const B = { points: [{ id: "s-0", x: 0, y: 2, z: 0 }] };

const vt = run(VIEWER, A);
ok(typeof vt.setMorphTarget === "function", "morph API exported");

const matched = vt.setMorphTarget(B);
ok(matched === 1, "alignment: 1 of A's points has a B match (" + matched + ")");

// t = 0 → A.
vt.applyMorph(0);
let p = vt.curPos(0);
near(Math.hypot(p[0], p[1], p[2]), 1, 1e-6, "t=0 keeps A radius (s-0)");
near(p[0], 1, 1e-6, "t=0 s-0 at A.x");

// t = 1 → B.
vt.applyMorph(1);
p = vt.curPos(0);
near(p[0], 0, 1e-6, "t=1 s-0.x → B.x (0)");
near(p[1], 2, 1e-6, "t=1 s-0.y → B.y (2)");
near(Math.hypot(p[0], p[1], p[2]), 2, 1e-6, "t=1 s-0 radius → B radius (2)");

// t = 0.5 → on the geodesic: dir slerp of orthogonal pair = 45°, radius lerped.
vt.applyMorph(0.5);
p = vt.curPos(0);
const mag = Math.hypot(p[0], p[1], p[2]);
near(mag, 1.5, 1e-6, "t=0.5 radius is lerped (1.5)");
near(p[0] / mag, Math.SQRT1_2, 1e-6, "t=0.5 dir.x = cos45 (slerp midpoint)");
near(p[1] / mag, Math.SQRT1_2, 1e-6, "t=0.5 dir.y = sin45 (slerp midpoint)");

// s-1 (no B match) stays at A regardless of t.
const q = vt.curPos(1);
near(q[2], 1, 1e-6, "unmatched s-1 stays at A (z=1)");
near(Math.hypot(q[0], q[1], q[2]), 1, 1e-6, "unmatched s-1 keeps A radius");

// clearMorph → applyMorph(1) becomes a no-op morph (normal A view).
vt.clearMorph();
vt.applyMorph(1);
p = vt.curPos(0);
near(p[0], 1, 1e-6, "after clearMorph, applyMorph(1) restores A (no target)");

// A rebuild clears the morph target.
vt.setMorphTarget(B);
vt.rebuild(vt.parseScene([{ id: "z", x: 1, y: 0, z: 0, cat: "Q" }]));
ok(vt.morphT === 0, "rebuild reset morphT");
ok(vt.setMorphTarget({ points: [] }) === 0, "morph target empty after rebuild context");

console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
process.exit(fails === 0 ? 0 : 1);
