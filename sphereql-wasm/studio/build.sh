#!/usr/bin/env bash
# Build the SphereQL Studio into ./dist (separate-file packaging:
# self-contained index.html + worker.js + studio.js + pkg/*.wasm).
#
#   ./build.sh
#   (cd dist && python -m http.server 8080)   # then open http://localhost:8080/
#
# A worker + wasm need to be served over http(s) — file:// will not work.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DIST="$HERE/dist"
mkdir -p "$DIST/pkg"

echo "1/4  emit studio shell (viewer + chrome)…"
( cd "$ROOT" && cargo run -q -p sphereql-wasm --example build_studio )

echo "2/4  build wasm (release)…"
( cd "$ROOT" && cargo build -q -p sphereql-wasm --target wasm32-unknown-unknown --release )

echo "3/4  generate no-modules glue (wasm-bindgen)…"
# Prefer wasm-pack if present (applies wasm-opt itself); else wasm-bindgen + opt.
if command -v wasm-pack >/dev/null 2>&1; then
  ( cd "$ROOT" && wasm-pack build sphereql-wasm --release --target no-modules --out-dir "$DIST/pkg" )
elif command -v wasm-bindgen >/dev/null 2>&1; then
  wasm-bindgen --target no-modules --no-typescript \
    --out-dir "$DIST/pkg" \
    "$ROOT/target/wasm32-unknown-unknown/release/sphereql_wasm.wasm"
  if command -v wasm-opt >/dev/null 2>&1; then
    wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
      "$DIST/pkg/sphereql_wasm_bg.wasm" -o "$DIST/pkg/sphereql_wasm_bg.wasm"
  fi
else
  echo "error: need wasm-pack or wasm-bindgen-cli." >&2
  echo "  install one:  cargo install wasm-pack   (or)   cargo install wasm-bindgen-cli" >&2
  exit 1
fi

echo "4/4  copy worker + drivers…"
cp "$HERE/worker.js" "$HERE/studio.js" "$HERE/compare.js" "$HERE/compare.html" "$DIST/"

echo "✓ Studio built → $DIST/index.html  (compare: $DIST/compare.html)"
echo "  serve:  (cd '$DIST' && python -m http.server 8080)  →  http://localhost:8080/"
