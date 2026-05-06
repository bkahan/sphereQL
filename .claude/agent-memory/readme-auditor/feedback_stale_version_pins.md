---
name: Stale version pins in install snippets after workspace bump
description: Workspaces that bump `[workspace.package].version` often forget to update README install snippets — recurring audit finding
type: feedback
---

When a Rust workspace bumps `[workspace.package].version`, install snippets in README.md, docs/quickstart-*.md, and per-crate READMEs frequently get left behind.

**Why:** sphereQL audit (2026-05-05) — workspace was at `0.2.0-alpha` but `README.md:41`, `docs/quickstart-rust.md:8`, `docs/architecture.md:63` all still showed `version = "0.1"`. Same anti-pattern observed across other Rust projects.

**How to apply:** During any README audit on a Rust project:
1. Read `Cargo.toml` `[workspace.package].version` first.
2. `grep -rn 'version = "[0-9]'` across `README.md` + `docs/*.md` + per-crate READMEs.
3. Cross-check every match against the workspace version.
4. Flag as **BLOCKER** for any release audit — copy-pasting users get either nothing on crates.io or the wrong version.

Recommended fix: prefer `cargo add <crate> --features <X>` snippets in README — they auto-resolve the latest published version and never go stale. Show the explicit `Cargo.toml` snippet as a secondary form for IDE-less readers.
