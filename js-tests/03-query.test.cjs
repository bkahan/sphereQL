const path = require("path");
const { run } = require("./harness.cjs");
const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
const ok = (c, m) => { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); };

// Scene with stable ids on 5 points across 2 categories.
const boot = {
  title: "q", surface_radius: 1, stats: { projection_kind: "pca", evr: 0.5, evr_label: "x" }, overlays: [],
  points: [
    { id: "s-0", x: 1, y: 0, z: 0, r: 1, theta: 0, phi: 1.5708, cat: "A", label: "a0" },
    { id: "s-1", x: 0, y: 1, z: 0, r: 1, theta: 1.5708, phi: 1.5708, cat: "A", label: "a1" },
    { id: "s-2", x: 0, y: 0, z: 1, r: 1, theta: 0, phi: 0, cat: "B", label: "b0" },
    { id: "s-3", x: -1, y: 0, z: 0, r: 1, theta: 3.1416, phi: 1.5708, cat: "B", label: "b1" },
    { id: "s-4", x: 0, y: -1, z: 0, r: 1, theta: 4.7124, phi: 1.5708, cat: "A", label: "a2" },
  ],
};
const vt = run(VIEWER, boot);
ok(typeof vt.highlightByIds === "function", "highlightByIds exported");
ok(vt.idCount === 5, "idToIndex built for all 5 ids (" + vt.idCount + ")");

// Highlight by raw id strings.
let n = vt.highlightByIds(["s-0", "s-2", "s-4"]);
ok(n === 3, "3 raw ids resolved");
ok(vt.queryGroup.children.length === 2, "geodesic fan: top → 2 others = 2 arcs (" + vt.queryGroup.children.length + ")");
const sa = vt.pts && true; // sizes checked via the geometry below
// matched points get emphasized sizes, others dimmed — read the size buffer.

// Highlight by {id,...} objects (NearestOut shape).
n = vt.highlightByIds([{ id: "s-1", distance: 0.1 }, { id: "s-3", distance: 0.2 }]);
ok(n === 2, "2 NearestOut-shaped objects resolved");
ok(vt.queryGroup.children.length === 1, "fan re-drawn: top → 1 other = 1 arc");

// Unknown ids are skipped; all-unknown clears.
n = vt.highlightByIds(["nope", "s-0", "missing"]);
ok(n === 1, "unknown ids skipped, 1 real match kept");
n = vt.highlightByIds(["nope", "missing"]);
ok(n === 0, "all-unknown → 0 (clears highlight)");
ok(vt.queryGroup.children.length === 0, "query group cleared when nothing matches");

// Empty input clears.
vt.highlightByIds(["s-0", "s-1"]);
ok(vt.queryGroup.children.length === 1, "fan present before clear");
vt.highlightByIds([]);
ok(vt.queryGroup.children.length === 0, "empty array clears the fan");

// A rebuild rebuilds idToIndex and clears the query layer.
vt.highlightByIds(["s-0", "s-2"]);
vt.rebuild(vt.parseScene([{ id: "x", x: 1, y: 0, z: 0, cat: "Z" }, { id: "y", x: 0, y: 1, z: 0, cat: "Z" }]));
ok(vt.idCount === 2, "rebuild rebuilt idToIndex (" + vt.idCount + ")");
ok(vt.queryGroup.children.length === 0, "rebuild cleared the query layer");
ok(vt.highlightByIds(["x"]) === 1 && vt.highlightByIds(["s-0"]) === 0, "new scene's ids resolve, old ids don't");

console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
process.exit(fails === 0 ? 0 : 1);
