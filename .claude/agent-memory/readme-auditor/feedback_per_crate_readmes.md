---
name: Per-crate README anti-patterns in Rust workspaces
description: Common gaps in per-crate READMEs for multi-crate Rust workspaces — what to flag during audit
type: feedback
---

In Rust workspaces, per-crate READMEs are the docs.rs landing page for each crate. They are the *primary* discovery surface for that crate, not an afterthought.

**Why:** Found during sphereQL audit (2026-05-05) — most crates had 7-line READMEs with no code example. Pattern observed across many Rust workspaces: maintainers think "the workspace README is enough" and underinvest in per-crate ones.

**How to apply:** When auditing per-crate READMEs in a workspace, check for:
- A minimal usage example (not just a description). docs.rs visitors arrive without the workspace context.
- A link back to the workspace repo.
- Version disclosure ("Current version: x.y.z, API may change") if pre-1.0.
- Badges (docs.rs, crates.io, license) — even one is better than none.
- Consistent voice/capitalization across all per-crate READMEs in the same workspace.

Acceptable exceptions: dev/internal crates clearly labeled as such (e.g. sphereql-corpus's "This is a dev/examples support crate — sphereQL users do not need to depend on it" is exemplary).
