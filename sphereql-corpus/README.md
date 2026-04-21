# sphereql-corpus

Shared test corpus for examples in the [sphereQL](https://github.com/bkahan/sphereQL) project.

Provides 775 concepts across 31 academic domains (physics, mathematics, biology, medicine, neuroscience, CS, AI, chemistry, engineering, astronomy, earth science, environmental science, psychology, philosophy, religion, linguistics, literature, history, sociology, anthropology, political science, law, economics, education, visual arts, music, film, performing arts, culinary arts, nanotechnology) with hand-crafted 128-dimensional sparse embeddings. Every semantic axis receives mass and bridge concepts deliberately straddle category boundaries, making it a stress-test corpus for spherical projection and category enrichment.

This is a dev/examples support crate -- it is not part of the core sphereQL library and sphereQL users do not need to depend on it. It exists so examples like `ai_knowledge_navigator` and `spatial_analysis` can share a meaningful corpus without inlining thousands of lines of concept data.

See the [main repository](https://github.com/bkahan/sphereQL) for full documentation, examples, and architecture overview.
