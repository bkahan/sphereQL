---
name: sphereQL benchmark accuracy crisis
description: README performance claims are 3 orders of magnitude wrong vs benchmark_results.json - blocking issue for any release
type: project
---

The README Performance section contains numbers from an earlier benchmark run that were never updated. Key discrepancies:
- SphereQL-only latency claimed as 2.1 us, actual is ~1,650 us (~1.65 ms) -- 785x error
- Speedup claimed as ~80,000x, actual is ~93x -- 860x error  
- KPCA query latency (~84 ms) is nearly as slow as brute force (~154 ms) but this isn't disclosed
- PCA/KPCA build times (72.5s / 85min) are not disclosed
- Precision@k (20.5% at k=5) is hidden by only showing nDCG

**Why:** Likely a microsecond-vs-millisecond unit confusion from an early prototype benchmark that predated the current spatial index implementation. The numbers were never re-validated after the benchmark suite was updated.

**How to apply:** When auditing performance claims in any README, always cross-reference against the project's own benchmark data files. Unit confusion (us vs ms vs ns) is a recurring pattern.
