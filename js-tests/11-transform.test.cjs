// GPU transform parity: the per-point spread/radial/morph transform now runs in
// the vertex shader (`sphTransform`), with `curPos(i)` as the CPU mirror that
// every CPU feature (selection, minimap, ruler, geodesics) reads. This suite
// locks two things headlessly: (1) curPos matches the documented formula, and
// (2) applyTransform pushes the uniforms the GLSL consumes — so the GPU and the
// CPU mirror are driven by identical inputs. The GLSL itself is a line-for-line
// transcription of curPos, verified visually in the browser smoke-test.
const path = require("path");
const { run } = require("./harness.cjs");

const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
const ok = (c, m) => { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); };
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
// curPos computes from `origPos` (a Float32Array — matching the shader's f32
// `position` input), so it carries f32 rounding (~1e-7) vs an f64 reference.
// 1e-5 tolerates that while still catching any real formula error (≥1e-2).
const near = (a, b) => Math.abs(a - b) < 1e-5;

const boot = {
  surface_radius: 1, stats: {}, overlays: [],
  points: [
    { x: 1, y: 0, z: 0, cat: "A" }, { x: 0.9, y: 0.35, z: 0.1, cat: "A" },
    { x: 0, y: 1, z: 0, cat: "B" }, { x: 0.1, y: 0.8, z: 0.5, cat: "B" },
    { x: 0, y: 0, z: 1, cat: "B" },
  ],
};
const vt = run(VIEWER, boot);

// Reference spread+radial transform — the formula curPos() and the GLSL both
// implement. Kept independent so a drift in either implementation is caught.
function ref(p, cat, spreadF, radialG, SR, catDir) {
  const mag = Math.hypot(p.x, p.y, p.z);
  if (mag < 1e-9) return [p.x, p.y, p.z];
  let dx = p.x / mag, dy = p.y / mag, dz = p.z / mag;
  if (spreadF !== 1) {
    const c = catDir[cat], dot = clamp(c[0] * dx + c[1] * dy + c[2] * dz, -1, 1), om = Math.acos(dot);
    if (om >= 1e-4) {
      const s = Math.sin(om), w1 = Math.sin((1 - spreadF) * om) / s, w2 = Math.sin(spreadF * om) / s;
      const nx = c[0] * w1 + dx * w2, ny = c[1] * w1 + dy * w2, nz = c[2] * w1 + dz * w2, nm = Math.hypot(nx, ny, nz) || 1;
      dx = nx / nm; dy = ny / nm; dz = nz / nm;
    }
  }
  const nmag = Math.max(0.02, SR + (mag - SR) * radialG);
  return [dx * nmag, dy * nmag, dz * nmag];
}

// At defaults (spread=1, radial=1) the transform is the identity: curPos === origPos.
let identity = true;
for (let i = 0; i < vt.N; i++) {
  const c = vt.curPos(i), p = vt.pts[i];
  if (!(near(c[0], p.x) && near(c[1], p.y) && near(c[2], p.z))) identity = false;
}
ok(identity, "default spread/radial → curPos is the identity (origPos)");

// Drive spread + radial, then check curPos parity AND that the uniforms the
// shader reads were pushed to the same values.
vt.spreadF = 0.4; vt.radialG = 1.8;
vt.applyTransform();
const u = vt.uniforms;
ok(u && u.uSpread.value === 0.4 && u.uRadial.value === 1.8 && u.uSR.value === vt.SR,
  "applyTransform pushed uSpread/uRadial/uSR — GPU gets curPos's inputs");
ok(u.uMorphT.value === 0 && u.uHasMorph.value === 0, "no morph → morph uniforms zeroed");

let parity = true;
for (let i = 0; i < vt.N; i++) {
  const c = vt.curPos(i), r = ref(vt.pts[i], vt.pts[i].cat, 0.4, 1.8, vt.SR, vt.catDir);
  if (!(near(c[0], r[0]) && near(c[1], r[1]) && near(c[2], r[2]))) { parity = false; console.log("  mismatch @", i, c, r); }
}
ok(parity, "curPos matches the spread+radial reference for every point");

// Radial-only (spread stays 1) is a clean check independent of catDir.
vt.spreadF = 1; vt.radialG = 0.5; vt.applyTransform();
let radialOk = true;
for (let i = 0; i < vt.N; i++) {
  const p = vt.pts[i], mag = Math.hypot(p.x, p.y, p.z), nmag = Math.max(0.02, vt.SR + (mag - vt.SR) * 0.5);
  const c = vt.curPos(i), e = [p.x / mag * nmag, p.y / mag * nmag, p.z / mag * nmag];
  if (!(near(c[0], e[0]) && near(c[1], e[1]) && near(c[2], e[2]))) radialOk = false;
}
ok(radialOk, "radial-only scaling matches max(0.02, SR+(mag-SR)*g)·dir");

// Morph drives the uniforms + the morph attributes, and curPos follows the
// morph path. (04-morph covers the endpoint math; here we lock the GPU wiring.)
// Re-boot with ids so morph can match A↔B by id.
vt.rebuild({ surface_radius: 1, stats: {}, overlays: [], points: boot.points.map((p, i) => ({ ...p, id: "p" + i })) });
const B = { points: vt.pts.map((p) => ({ id: p.id, x: p.x, y: p.y, z: p.z })) }; // B === A by id
const matched = vt.setMorphTarget(B);
ok(matched === vt.N, "setMorphTarget matched all points by id (" + matched + ")");
vt.applyMorph(1);
ok(vt.uniforms.uHasMorph.value === 1 && vt.uniforms.uMorphT.value === 1, "applyMorph(1) pushes uHasMorph=1, uMorphT=1");
// B == A, so morph to t=1 returns each point to itself.
let morphOk = true;
for (let i = 0; i < vt.N; i++) { const c = vt.curPos(i), p = vt.pts[i]; if (!(near(c[0], p.x) && near(c[1], p.y) && near(c[2], p.z))) morphOk = false; }
ok(morphOk, "morph t=1 to an identical B returns each point to itself");
vt.clearMorph();
ok(vt.uniforms.uHasMorph.value === 0, "clearMorph drops the morph uniforms");

// GPU pick id codec: index baked into an RGB color (id=i+1) survives the 8-bit
// channel round-trip; 0,0,0 = background.
let codecOk = true;
for (const i of [0, 1, 42, 255, 256, 65535, 65536, 1000000, 16777214]) {
  const f = vt.pickEncode(i), b = f.map((x) => Math.round(x * 255));
  if (vt.pickDecode(b[0], b[1], b[2]) !== i) { codecOk = false; console.log("  pick codec fail @", i); }
}
ok(codecOk, "pick id codec round-trips i↔rgb across the 24-bit range");
ok(vt.pickDecode(0, 0, 0) === -1, "rgb (0,0,0) decodes to -1 (background)");
// In the headless harness there's no render target, so getHovered degrades to
// the CPU fallback; an off-screen cursor resolves to no point.
ok(vt.getHovered({ clientX: -9999, clientY: -9999 }) === -1, "getHovered → -1 for an off-screen cursor (CPU fallback)");

console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
process.exit(fails === 0 ? 0 : 1);
