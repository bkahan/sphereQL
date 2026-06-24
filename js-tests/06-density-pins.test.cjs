// Density-shading heatmap + pins + TOML config round-trip, on the createViewer
// factory. rebuild() bins every point into a θ×φ grid → a per-point normalized
// density attribute; the Settings density toggle flips the shader uniform.
// Pins drop (θ,φ) annotation markers; currentSettings()/applySettings() round-
// trips the whole Settings pane (incl. base64 pins) through a .toml-style blob.
const path = require("path");
const { run } = require("./harness.cjs");
const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
const ok = (c, m) => { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); };
const near = (a, b, e, m) => ok(Math.abs(a - b) < (e || 1e-6), m + " (" + (+a).toFixed(4) + " vs " + (+b).toFixed(4) + ")");

// 4 points clustered near [1,0,0] (one θ×φ bin) + 1 lone point at [0,0,1].
const boot = {
  title: "d", surface_radius: 1, stats: { projection_kind: "pca", evr: 0.5, evr_label: "x" }, overlays: [],
  points: [
    { x: 1, y: 0, z: 0, r: 1, theta: 0, phi: 1.5708, cat: "A", label: "0" },
    { x: 1, y: 0, z: 0, r: 1, theta: 0, phi: 1.5708, cat: "A", label: "1" },
    { x: 1, y: 0, z: 0, r: 1, theta: 0, phi: 1.5708, cat: "A", label: "2" },
    { x: 1, y: 0, z: 0, r: 1, theta: 0, phi: 1.5708, cat: "A", label: "3" },
    { x: 0, y: 0, z: 1, r: 1, theta: 0, phi: 0, cat: "B", label: "lone" },
  ],
};
const vt = run(VIEWER, boot);

// ── Density attribute ────────────────────────────────────────────────────
const dens = vt.densityArr;
ok(dens && dens.length === 5, "density attribute built per point");
near(Math.max(dens[0], dens[1], dens[2], dens[3]), 1.0, 1e-6, "densest bin normalizes to 1.0");
near(dens[4], 0.25, 1e-6, "lone point's bin = 1/4 of the max");
ok(dens[0] > dens[4], "clustered points are denser than the lone point");

// Density toggle via settings (uniform flips).
ok(vt.densityOn === 0, "density off by default");
vt.applySettings({ density: true });
ok(vt.densityOn === 1, "applySettings density:true → heatmap uniform on");
vt.applySettings({ density: false });
ok(vt.densityOn === 0, "applySettings density:false → off");

// ── Pins ─────────────────────────────────────────────────────────────────
ok(vt.pins.length === 0 && vt.pinGroup.children.length === 0, "no pins initially");
vt.addPin(1.0, 1.2);
vt.addPin(2.5, 0.8, "north");
ok(vt.pins.length === 2, "two pins added");
ok(vt.pinGroup.children.length === 2, "two pin markers in the scene");
ok(vt.pins[1].label === "north", "explicit pin label kept");
ok(/^pin /.test(vt.pins[0].label), "auto-labeled pin is ASCII");

// Pin mode toggle.
vt.setPinMode(true); ok(vt.pinOn === true, "pin mode on");

// ── TOML round-trip of pins + density ────────────────────────────────────
vt.applySettings({ density: true });
const s = vt.currentSettings();
ok(typeof s.pins === "string" && s.density === true, "settings carry pins(base64)+density");
vt.clearPins();
ok(vt.pins.length === 0, "pins cleared");
vt.applySettings(s);
ok(vt.pins.length === 2, "pins restored from settings");
near(vt.pins[0].theta, 1.0, 1e-9, "restored pin theta");
near(vt.pins[1].phi, 0.8, 1e-9, "restored pin phi");
ok(vt.pins[1].label === "north", "restored pin label");

// ── Rebuild disarms pin mode + clears pins/density ───────────────────────
vt.setPinMode(true); vt.addPin(0.5, 0.5);
vt.rebuild(vt.parseScene([{ x: 1, y: 0, z: 0, cat: "Z" }, { x: 0, y: 1, z: 0, cat: "Z" }]));
ok(vt.pinOn === false, "rebuild disarms pin mode");
ok(vt.pins.length === 0 && vt.pinGroup.children.length === 0, "rebuild cleared pins");
ok(vt.densityOn === 0, "rebuild reset density to default");

console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
process.exit(fails === 0 ? 0 : 1);
