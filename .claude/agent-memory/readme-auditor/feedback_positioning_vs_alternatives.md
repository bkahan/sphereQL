---
name: Missing comparison-to-alternatives is the #1 README adoption gap
description: For libraries in a crowded category, omitting a "how does this compare to X" block is the most common reason a strong project flops on HN/r/rust
type: feedback
---

Every library that lives in a crowded ecosystem (vector DBs, ORMs, web frameworks, ANN indexes) needs an explicit "How this compares to alternatives" block in the top README. Skipping it is the most common adoption-blocker found in audits.

**Why:** Observed pattern across multiple audits. The reader's first question is always "why not just use $POPULAR_TOOL?" — if the README doesn't answer it, the reader either bounces or fills it in unfavorably from their own assumptions. Then on HN, the top comment is "How is this different from FAISS / Qdrant / pgvector / hnswlib?" and the maintainer plays defense for the entire thread.

**How to apply:** In any README audit:
1. Identify the project's category (vector search, embedding store, ANN index, etc.).
2. Search for explicit comparison text: "compared to", "unlike", "vs", names of competing tools.
3. If absent, flag as **BLOCKER** for v1 release readiness regardless of how strong the rest of the README is.
4. Recommend: 4-6 line block, framed as positioning ("complement to", "replacement for", "layer on top of") not feature-checklists.
5. Honest framing earns more credibility than aggressive comparison: "Use sphereQL alongside Qdrant, not instead of it" beats "10x faster than Qdrant".
