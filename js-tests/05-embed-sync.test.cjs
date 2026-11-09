const path = require("path");
const { run } = require("./harness.cjs");
const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
const ok = (c, m) => { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); };

const boot = {
  title: "e", surface_radius: 1, stats: { projection_kind: "pca", evr: 0.5, evr_label: "x" }, overlays: [],
  points: [{ x: 1, y: 0, z: 0, r: 1, theta: 0, phi: 1.5708, cat: "A", label: "a" }],
};

// ── With #embed: sync is active ──────────────────────────────────────────
const vt = run(VIEWER, boot, { hash: "#embed" });
ok(vt.N === 1, "embed viewer booted (N=1)");

// Scene injection via postMessage → rebuild.
vt.fireWindow("message", { data: { type: "sphereql-scene", scene: { points: [{ x: 1, y: 0, z: 0, cat: "Z" }, { x: 0, y: 1, z: 0, cat: "Z" }, { x: 0, y: 0, z: 1, cat: "Q" }] } } });
ok(vt.N === 3, "injected scene rebuilt the viewer (N=3)");

// Camera broadcast: move camera + fire 'change' → one parent post.
vt.parentPosts.length = 0;
vt.camera.position.set(20, 20, 20);
vt.fireCtrlChange();
ok(vt.parentPosts.length === 1 && vt.parentPosts[0].type === "sphereql-cam", "camera move broadcasts {sphereql-cam}");
ok(Array.isArray(vt.parentPosts[0].s) && vt.parentPosts[0].s.length === 6, "broadcast carries the 6-tuple pose");

// Epsilon gate: firing 'change' again WITHOUT moving must NOT re-post.
vt.fireCtrlChange();
ok(vt.parentPosts.length === 1, "no move → no duplicate broadcast (epsilon gate)");

// Move past epsilon → posts again.
vt.camera.position.set(40, 20, 20);
vt.fireCtrlChange();
ok(vt.parentPosts.length === 2, "moving past epsilon broadcasts again");

// Echo guard: applying a remote camera must NOT bounce a broadcast back.
vt.parentPosts.length = 0;
vt.fireWindow("message", { data: { type: "sphereql-cam", s: [5, 6, 7, 1, 2, 3] } });
ok(vt.camera.position.x === 5 && vt.controls.target.y === 2, "remote camera applied");
vt.fireCtrlChange(); // the change that controls.update() would emit
ok(vt.parentPosts.length === 0, "applied remote camera does not echo back (no feedback storm)");

// Malformed messages are ignored.
vt.fireWindow("message", { data: { type: "sphereql-cam", s: [1, 2, 3] } }); // wrong length
vt.fireWindow("message", { data: "garbage" });
ok(true, "malformed sync messages ignored without throwing");

// Messages from a NON-parent window are rejected (origin/source hardening).
const beforeN = vt.N;
vt.fireWindow("message", { source: {}, data: { type: "sphereql-scene", scene: { points: [{ x: 1, y: 0, z: 0, cat: "X" }] } } });
ok(vt.N === beforeN, "scene-inject from a non-parent source is rejected");

// Independent orbit / zoom locks from the compare host.
vt.fireWindow("message", { data: { type: "sphereql-lock", lockRotate: true, lockZoom: false } });
ok(vt.controls.enableRotate === false, "lockRotate → controls.enableRotate=false");
ok(vt.controls.enableZoom === true && vt.zoomLocked === false, "zoom stays unlocked (independent)");
vt.fireWindow("message", { data: { type: "sphereql-lock", lockRotate: false, lockZoom: true } });
ok(vt.controls.enableRotate === true, "lockRotate cleared → rotate re-enabled");
ok(vt.controls.enableZoom === false && vt.zoomLocked === true, "lockZoom → pinch off + wheel zoomLocked");

// ── Without #embed: completely inert ─────────────────────────────────────
const plain = run(VIEWER, boot, { hash: "" });
plain.camera.position.set(99, 99, 99);
plain.fireCtrlChange();
ok(plain.parentPosts.length === 0, "no #embed → viewer never posts to parent");

console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
process.exit(fails === 0 ? 0 : 1);
