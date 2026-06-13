# sphereql-examples

Runnable Rust examples for the [sphereQL](https://github.com/bkahan/sphereQL)
workspace. Dev-only crate (`publish = false`) — it exists so every
example can run without per-crate `--features` juggling:

```bash
cargo run -p sphereql-examples --example <name>
```

The full annotated catalog with invocations lives in
[docs/examples.md](https://github.com/bkahan/sphereQL/blob/main/docs/examples.md).
A quick map of what's in [`examples/`](./examples/):

- **Basics** — `basic_positioning`, `geospatial`, `graphql_server`,
  `word_embeddings`, `semantic_search`, `auto_categorize`
- **Category & spatial analysis** — `category_enrichment`,
  `ai_knowledge_navigator`, `spatial_analysis`, `e2e_transformer`,
  `benchmark`
- **Auto-tuning & metalearning** — `auto_tune`, `meta_learn`,
  `meta_warm_start`, `meta_feedback`, `cq_e2e`
- **Corpus tooling** — `corpus_self_tune`, `bulk_ingest` (some sources
  need `--features bulk-dump` / `bulk-dbpedia`), `load_500k_smoke`,
  `lap_diag`
- **Lingua** — `lingua_e2e` (text → `ConceptGraph` → sphereQL positions)
- **The whole story** — `full_e2e`: an interactive 8-phase demo that
  runs the complete metalearning loop end to end (auto-tune →
  meta-learn → projection → spatial analysis → category analysis →
  queries → divergence cartography → self-tune controller). Run with
  `--release`.

Part of the sphereQL workspace, currently `0.3.0`.
