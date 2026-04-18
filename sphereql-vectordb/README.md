# sphereql-vectordb

Vector database connectors for the [sphereQL](https://github.com/bkahan/sphereQL) project.

Bridges external vector stores (InMemory, Qdrant via gRPC, Pinecone) with the sphereQL embedding pipeline. Handles sync, PCA fitting, projection, and hybrid search with cosine similarity re-ranking in the original embedding space.

See the [main repository](https://github.com/bkahan/sphereQL) for full documentation, examples, and architecture overview.
