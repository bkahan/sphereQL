# sphereql-lingua

Language → native sphereQL coordinates for the [sphereQL](https://github.com/bkahan/sphereQL) project.

A six-stage pipeline that turns free-form text into a `ConceptGraph`
with every node resolved to a sphereQL `(r, θ, φ)` position. Built on
`sphereql-core` so the coordinate convention, distance math (Vincenty),
and slerp are shared with the rest of the workspace.

## Pipeline stages

1. **Concept extraction** — pluggable `ConceptExtractor` trait, with
   `RegexExtractor` as the default heuristic. Swap in an LLM-backed
   extractor for production.
2. **θ (domain angle)** — `DomainTaxonomy` maps each concept onto the
   azimuthal axis. Anchored on a small set of canonical domains;
   ambiguous concepts are interpolated.
3. **φ (abstraction level)** — `AbstractionResolver` reads
   abstraction-marker patterns from the source text to place each
   concept on the polar axis (north pole = maximally abstract, south
   pole = maximally concrete).
4. **r (epistemic weight)** — `SalienceScorer` derives a salience
   scalar (frequency, position, marker proximity) and maps it to
   radius.
5. **Relation encoding** — `RelationEncoder` lifts surface patterns
   ("X causes Y", "X is a kind of Y", …) into typed `Relation`s
   between concepts, including geodesic-arc directionality where
   applicable.
6. **Graph assembly** — emit a `ConceptGraph { concepts, relations }`
   with `to_sphereql()` for printing in the canonical sphereQL textual
   form.

## Example

```rust
use sphereql_lingua::LinguaPipeline;

let pipeline = LinguaPipeline::new();
let graph = pipeline.process("Photosynthesis converts light into chemical energy.");
println!("{}", graph.to_sphereql(pipeline.taxonomy()));
```

The default `LinguaPipeline::new()` uses heuristic extractors so it
runs without external dependencies. For higher-fidelity extraction,
pass a custom `ConceptExtractor` via `.with_extractor(...)`.

## Relationship to `lingua-spherica` (Python)

The Python package `lingua-spherica` is a thin skeleton — coordinate
types and basic spherical math only. The full pipeline lives here in
Rust and is exposed to Python via the `sphereql-python` bindings
behind the `lingua` Cargo feature (`maturin develop --features
lingua`; the default PyPI wheel does not include it yet).
**Rust is the source of truth.**

## Status

Part of the sphereQL workspace, currently `0.3.0`. Pre-1.0:
expect breaking changes between minor versions.

## Documentation

See the workspace
[architecture.md](https://github.com/bkahan/sphereQL/blob/main/docs/architecture.md).
