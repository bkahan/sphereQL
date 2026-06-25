// SQT1 binary tile decoder (viewer.js `decodeTile`). The headline check is a
// cross-language golden: the exact bytes that sphereql-vis tile.rs'
// `golden_bytes_match` asserts `encode_tile` produces. Decoding them here
// proves the JS reader and the Rust writer agree on the wire format — including
// u16/u32 little-endian layout (cat=260, row=70000 exceed a byte).
const path = require("path");
const { run } = require("./harness.cjs");

const VIEWER = path.join(__dirname, "..", "sphereql-vis", "src", "viewer.js");
let fails = 0;
const ok = (c, m) => { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); };
const throws = (fn, m) => { try { fn(); console.log("FAIL (no throw):", m); fails++; } catch (e) { console.log("ok  :", m, "→", e.message); } };

const vt = run(VIEWER, { points: [{ x: 1, y: 0, z: 0, cat: "A", label: "a" }] });
ok(typeof vt.decodeTile === "function", "decodeTile exported");

// Must stay byte-identical to tile.rs golden_bytes_match.
const GOLDEN =
  "535154310100000002000000000000000000c03f000000c00000003f030000000700000000000000" +
  "0000803e000040bf0401000070110100";
const bytes = Buffer.from(GOLDEN, "hex");

const t = vt.decodeTile(bytes);
ok(t.count === 2, "golden tile decodes to 2 points (count=" + t.count + ")");
ok(Math.abs(t.positions[0] - 1.5) < 1e-6 && Math.abs(t.positions[1] + 2.0) < 1e-6 && Math.abs(t.positions[2] - 0.5) < 1e-6,
  "point 0 xyz = (1.5, -2, 0.5)");
ok(t.cats[0] === 3 && t.rows[0] === 7, "point 0 cat=3 row=7");
ok(Math.abs(t.positions[3]) < 1e-6 && Math.abs(t.positions[4] - 0.25) < 1e-6 && Math.abs(t.positions[5] + 0.75) < 1e-6,
  "point 1 xyz = (0, 0.25, -0.75)");
ok(t.cats[1] === 260, "point 1 cat=260 (u16 LE, > one byte)");
ok(t.rows[1] === 70000, "point 1 row=70000 (u32 LE, > u16)");

// Typed-array shapes.
ok(t.positions.length === 6 && t.cats.length === 2 && t.rows.length === 2, "buffers sized to count");

// Accepts a raw ArrayBuffer too.
const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
ok(vt.decodeTile(ab).count === 2, "decodes from an ArrayBuffer (not just Uint8Array)");

// Empty tile (header only, count 0).
ok(vt.decodeTile(Buffer.from("53515431010000000000000000000000", "hex")).count === 0, "empty tile → 0 points");

// Error paths mirror Rust TileError.
throws(() => vt.decodeTile(Buffer.from([1, 2, 3])), "short buffer rejected");
const badMagic = Buffer.from(GOLDEN, "hex"); badMagic[0] = 0x58;
throws(() => vt.decodeTile(badMagic), "bad magic rejected");
const futureVer = Buffer.from(GOLDEN, "hex"); futureVer[4] = 99;
throws(() => vt.decodeTile(futureVer), "future version rejected");
throws(() => vt.decodeTile(Buffer.from(GOLDEN, "hex").subarray(0, 52)), "truncated body rejected (count mismatch)");

console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
process.exit(fails === 0 ? 0 : 1);
