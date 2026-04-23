# sphereql-corpus

Shared test corpora for examples in the [sphereQL](https://github.com/bkahan/sphereQL) project.

`build_corpus()` provides 775 concepts across 31 academic domains (physics, mathematics, biology, medicine, neuroscience, CS, AI, chemistry, engineering, astronomy, earth science, environmental science, psychology, philosophy, religion, linguistics, literature, history, sociology, anthropology, political science, law, economics, education, visual arts, music, film, performing arts, culinary arts, nanotechnology) with hand-crafted 128-dimensional sparse embeddings. Every semantic axis receives mass and bridge concepts deliberately straddle category boundaries.

`build_stress_corpus()` provides a second 300-concept synthetic corpus: 10 categories, 30 concepts each, exactly 2 authored signal axes per concept, `0.2` noise amplitude (5× the built-in default of `0.04`). A controlled A/B probe for projection families — variance-maximizing projections (PCA) degrade in this regime while connectivity-preserving ones (Laplacian eigenmap) recover the authored signature.

Both corpora use the same embedding format. Use `embed(features, seed)` for the default noise amplitude or `embed_with_noise(features, seed, amplitude)` for explicit control; `DEFAULT_NOISE_AMPLITUDE` and `STRESS_NOISE_AMPLITUDE` are exposed as constants.

This is a dev/examples support crate — it is not part of the core sphereQL library and sphereQL users do not need to depend on it. It exists so examples like `ai_knowledge_navigator`, `spatial_analysis`, `auto_tune`, `meta_learn`, and `meta_warm_start` can share meaningful corpora without inlining thousands of lines of concept data.

See the [main repository](https://github.com/bkahan/sphereQL) for full documentation, examples, and architecture overview.
