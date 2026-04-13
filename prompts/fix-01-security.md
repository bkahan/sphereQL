# Fix: Security Issues Across Codebase

## Priority: HIGH — address before any public release

## Context

A full codebase review identified security vulnerabilities at FFI
boundaries, in generated HTML output, and in storage connectors. These
are exploitable by callers passing crafted input through the public APIs.

---

## 1. XSS in viz template (`sphereql-python`)

**File:** `sphereql-python/src/viz_template.html:182,246,272`

`innerHTML` interpolates unescaped user-supplied labels and categories.
A category string like `<img src=x onerror=alert(1)>` executes JS in
any browser that opens the generated HTML.

**Fix:** Escape all user-derived strings before `innerHTML` interpolation:

```javascript
const escHtml = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
```

Apply to every `innerHTML` assignment that includes `p.label`, `p.cat`,
`pts[d.i].label`, or `cat`. Three locations: tooltip (line 182), neighbor
list (line 246), legend rows (line 272).

---

## 2. WASM `cache_write` path traversal

**File:** `sphereql-wasm/src/lib.rs:292-320`

The `#[cfg(not(target_arch = "wasm32"))]` branches pass raw caller `&str`
paths to `std::fs::read_to_string(path)` and `std::fs::write(path, json)`.
In Node.js/WASI targets, `cache_write("../../.env", "...")` overwrites
arbitrary files.

**Fix:** Either:
- Canonicalize the path and verify it stays within an expected directory, or
- Remove the `std::fs` branches entirely (these are for WASM, not native), or
- Accept only a filename (no path separators) and resolve against a fixed cache dir.

---

## 3. Qdrant FNV-1a hash collision — silent data loss

**File:** `sphereql-vectordb/src/qdrant.rs:354-366`

String IDs are hashed to `u64` PointIds via FNV-1a. Two IDs that collide
silently overwrite each other on `upsert`. The `get` method returns the
wrong record. Risk grows with collection size.

**Fix:** Use Qdrant's native UUID PointId variant (Qdrant accepts UUID
strings directly). For non-UUID IDs, either:
- Store a mapping table and detect collisions on write, or
- Use a collision-resistant hash (SHA-256 truncated to 128 bits as UUID).

---

## 4. Pinecone client — no request timeout, allows `http://`

**File:** `sphereql-vectordb/src/pinecone.rs:72-78`

`reqwest::Client::builder().build()` has no timeout. A hung Pinecone
endpoint blocks the caller indefinitely. All five operations are affected.

The URL normalization also allows `http://` scheme silently.

**Fix:**
```rust
let client = Client::builder()
    .timeout(Duration::from_secs(30))
    .build()
    .map_err(|e| VectorStoreError::Connection(e.to_string()))?;
```

Consider rejecting `http://` hosts or requiring an explicit `allow_insecure`
flag in `PineconeConfig`.

---

## 5. Subscription `Lagged` silently kills stream

**File:** `sphereql-graphql/src/subscription.rs:56-68` (and lines 82-98, 104-110)

`while let Ok(event) = rx.recv().await` exits the loop on
`Err(RecvError::Lagged(n))`, silently disconnecting slow subscribers.

**Fix:** Handle both error variants explicitly:

```rust
loop {
    match rx.recv().await {
        Ok(event) => { /* yield / filter */ }
        Err(broadcast::error::RecvError::Lagged(n)) => {
            // log warn, continue
        }
        Err(broadcast::error::RecvError::Closed) => break,
    }
}
```

Apply to all three subscription resolvers.

---

## Verification

- For XSS: create a test embedding with category `<script>alert(1)</script>`,
  generate HTML, open in browser, confirm no script execution.
- For path traversal: unit test that rejects `../` in cache paths.
- For Qdrant: unit test that two colliding IDs are stored and retrieved correctly.
- For Pinecone: integration test with a mock server that never responds; confirm
  the client times out within 30s.
- For subscriptions: test with a slow receiver; confirm stream continues after Lagged.
