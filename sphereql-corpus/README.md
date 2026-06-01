# sphereql-corpus

Shared test corpora for examples in the [sphereQL](https://github.com/bkahan/sphereQL) project.

## Corpora

### Hand-crafted (775 concepts)

`build_corpus()` provides 775 concepts across 31 academic domains with
hand-crafted 128-dimensional sparse embeddings. Every semantic axis receives
mass and bridge concepts deliberately straddle category boundaries.

### Extended (~5,000+ concepts)

`build_extended_corpus()` provides ~5,000+ concepts across the same 31
domains, sourced from:

- **OpenAlex Topics** — academic research topics with keyword-derived
  axis mappings. [OpenAlex](https://openalex.org) data is CC0 licensed.
- **OpenAlex Subfields** promoted to standalone concepts.
- **Gap-fill concepts** for categories underrepresented in OpenAlex
  (culinary arts, architecture, performing arts, film studies, visual arts,
  music, religion, law, linguistics, literature, astronomy, data science).

`build_full_corpus()` returns the union of both (~5,775+ concepts).

### Stress (300 concepts)

`build_stress_corpus()` provides a 300-concept synthetic corpus: 10
categories, 30 concepts each, exactly 2 authored signal axes per concept,
`0.2` noise amplitude (5× the built-in default of `0.04`). A controlled
A/B probe for projection families.

### Bulk / large-scale corpora

`CorpusId` is a registry enum that names every corpus the crate knows about,
including bulk-ingested parquets:

| Variant | Source file |
|---|---|
| `HandCrafted` | in-memory |
| `Extended` | `data/extended_corpus.parquet` |
| `Full` | HandCrafted + Extended |
| `Stress` | in-memory |
| `DBpedia50k` | `data/dbpedia_50k.parquet` |
| `DBpedia50kClustered` | `data/dbpedia_50k.clustered.parquet` |
| `DBpedia50kTuned` | `data/dbpedia_50k.clustered.tuned.parquet` |
| `DBpedia500k` | `data/dbpedia_500k.parquet` |
| `DBpedia500kClustered` | `data/dbpedia_500k.clustered.parquet` |
| `DBpedia500kTuned` | `data/dbpedia_500k.clustered.tuned.parquet` |
| `Wikidata50k` | `data/wikidata_50k.parquet` |
| `Parquet(path)` | any Parquet file |

```rust
use sphereql_corpus::CorpusId;

// Eagerly load one corpus.
let concepts = CorpusId::DBpedia500kClustered.load()?;

// Stream a large corpus without materializing the full Vec.
for concept in CorpusId::DBpedia500k.stream()? {
    let c = concept?;
    // ...
}

// Iterate all named corpora.
for id in CorpusId::all() {
    println!("{}: {:?}", id.name(), id.parquet_path());
}

// Any parquet file.
let custom = CorpusId::Parquet("/tmp/my_corpus.parquet".into()).load()?;
```

Bulk parquet files are produced by the `bulk_ingest` example — see
[`sphereql-examples/examples/bulk_ingest.rs`](../sphereql-examples/examples/bulk_ingest.rs)
for usage.

## Embedding format

All corpora use the same embedding format. Use `embed(features, seed)` for
the default noise amplitude or `embed_with_noise(features, seed, amplitude)`
for explicit control; `DEFAULT_NOISE_AMPLITUDE` and `STRESS_NOISE_AMPLITUDE`
are exposed as constants.

## Regenerating the extended corpus

The extended corpus data file (`data/extended_corpus.json`) is checked into
the repository. To regenerate from the OpenAlex API:

```bash
cd sphereql-corpus/tools
pip install -r requirements.txt
OPENALEX_API_KEY=your_email_or_api_key python3 generate_extended.py
python3 validate_corpus.py  # verify invariants
```

The `OPENALEX_API_KEY` may be either a Premium API key or your contact email
address (auto-detected) for the free polite pool. See
<https://openalex.org/settings/api>.

## Note

This is a dev/examples support crate — it is not part of the core sphereQL
library and sphereQL users do not need to depend on it.

See the [main repository](https://github.com/bkahan/sphereQL) for full
documentation, examples, and architecture overview.
