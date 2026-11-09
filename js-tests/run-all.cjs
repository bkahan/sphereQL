#!/usr/bin/env node
// Run every js-tests/*.test.cjs in its own Node process; exit non-zero if any
// suite fails. Used by CI (`node js-tests/run-all.cjs`) and locally.
const fs = require("fs");
const path = require("path");
const cp = require("child_process");

const dir = __dirname;
const tests = fs.readdirSync(dir).filter((f) => f.endsWith(".test.cjs")).sort();
let failed = 0;

for (const t of tests) {
  process.stdout.write("\n▶ " + t + "\n");
  const r = cp.spawnSync(process.execPath, [path.join(dir, t)], { stdio: "inherit" });
  if (r.status !== 0) {
    failed++;
    console.log("✗ FAILED: " + t);
  }
}

console.log(
  "\n" +
    (failed === 0
      ? "✓ all " + tests.length + " JS suites passed"
      : "✗ " + failed + " of " + tests.length + " suites FAILED")
);
process.exit(failed === 0 ? 0 : 1);
