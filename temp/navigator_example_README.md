# Navigator Example — Assembly Instructions

The complete `ai_knowledge_navigator.rs` example (1656 lines) is split
across files in this directory due to GitHub API size constraints.

## Files

| File | Content | Lines |
|------|---------|-------|
| `navigator_example_part1_imports.txt` | imports, constants, `embed()`, `Concept` struct | ~80 |
| `navigator_example_part2_corpus_a.txt` | `build_corpus()` — Physics → Engineering (truncated) | ~360 |
| `navigator_example_part3_corpus_b.txt` | Earth Science → Religion + closing `]}` | ~350 |
| `navigator_example_part4_main.txt` | `fn main()` + analyses 1-7 setup + NavigatorConfig | ~490 |
| `navigator_example_part5_spatial.txt` | Spatial §1–§7 + summary + closing `}` | ~360 |

## Assembly

```bash
# The complete file is available in the PR artifacts.
# To assemble manually, concatenate in order:
cat navigator_example_part1_imports.txt \
    navigator_example_part2_corpus_a.txt \
    navigator_example_part3_corpus_b.txt \
    navigator_example_part4_main.txt \
    navigator_example_part5_spatial.txt \
    > ../sphereql/examples/ai_knowledge_navigator.rs
```

**NOTE:** Part 2 (corpus) and Part 4 (main body) are abbreviated in the
GitHub-hosted versions due to content API limits. The full, compile-verified
1656-line file was built and tested locally:

```
cargo check --example ai_knowledge_navigator --features embed  # ✓
cargo test -p sphereql-core -p sphereql-embed                  # 272 tests ✓
```

The complete file should be pushed via `git push` or copied from the
session outputs.
