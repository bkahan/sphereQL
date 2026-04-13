"""
Kernel PCA Projection — end-to-end example.

Compares linear PCA vs Gaussian kernel PCA for projecting embeddings
onto the sphere. Kernel PCA captures nonlinear structure that linear
PCA misses, which shows up as tighter per-category angular spread.

Usage:
    pip install sphereql
    python kernel_pca.py
"""

import math
import sphereql
from dataset import SENTENCES


def header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def angular_distance(a, b):
    """Great-circle distance between two SphericalPoints."""
    cos_d = (
        math.sin(a.theta) * math.sin(b.theta) * math.cos(a.phi - b.phi)
        + math.cos(a.theta) * math.cos(b.theta)
    )
    return math.acos(max(-1.0, min(1.0, cos_d)))


def main():
    categories = [s["category"] for s in SENTENCES]
    embeddings = [s["embedding"] for s in SENTENCES]
    texts = [s["text"] for s in SENTENCES]

    # ── 1. Fit both projections ──────────────────────────────────────
    header("1. Fit linear PCA vs Kernel PCA")

    pca = sphereql.PcaProjection.fit(embeddings)
    kpca = sphereql.KernelPcaProjection.fit(embeddings)

    print(f"  Linear PCA:  dim={pca.dimensionality}, "
          f"explained_variance={pca.explained_variance_ratio:.4f}")
    print(f"  Kernel PCA:  dim={kpca.dimensionality}, "
          f"sigma={kpca.sigma:.4f}, "
          f"explained_variance={kpca.explained_variance_ratio:.4f}, "
          f"training_points={kpca.num_training_points}")
    print(f"\n  {repr(kpca)}")

    # ── 2. Project a single embedding ────────────────────────────────
    header("2. Project a single embedding")

    emb = embeddings[0]
    print(f"  Text: \"{texts[0]}\"")
    print(f"  Category: {categories[0]}\n")

    pt_pca = pca.project(emb)
    pt_kpca = kpca.project(emb)

    print(f"  Linear PCA -> r={pt_pca.r:.4f}, "
          f"theta={pt_pca.theta:.4f}, phi={pt_pca.phi:.4f}")
    print(f"  Kernel PCA -> r={pt_kpca.r:.4f}, "
          f"theta={pt_kpca.theta:.4f}, phi={pt_kpca.phi:.4f}")

    # ── 3. Rich projection (certainty + intensity) ───────────────────
    header("3. Rich projection metadata")

    rich = kpca.project_rich(emb)
    pos = rich.position

    print(f"  position:   r={pos.r:.4f}, theta={pos.theta:.4f}, "
          f"phi={pos.phi:.4f}")
    print(f"  certainty:  {rich.certainty:.4f}")
    print(f"  intensity:  {rich.intensity:.4f}")
    print(f"  projection_magnitude: {rich.projection_magnitude:.4f}")

    # ── 4. Batch projection ──────────────────────────────────────────
    header("4. Batch projection (all 100 embeddings)")

    pts_pca = pca.project_batch(embeddings)
    pts_kpca = kpca.project_batch(embeddings)

    print(f"  Linear PCA: {len(pts_pca)} points projected")
    print(f"  Kernel PCA: {len(pts_kpca)} points projected")

    # ── 5. Compare category coherence ────────────────────────────────
    header("5. Category coherence: mean intra-category angular distance")
    print("  Lower = tighter clusters = better separation.\n")

    cats = sorted(set(categories))
    print(f"  {'Category':<20} {'Linear PCA':>12} {'Kernel PCA':>12}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 12}")

    for cat in cats:
        idxs = [i for i, c in enumerate(categories) if c == cat]
        if len(idxs) < 2:
            continue

        pca_dists, kpca_dists = [], []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                pca_dists.append(angular_distance(
                    pts_pca[idxs[i]], pts_pca[idxs[j]]))
                kpca_dists.append(angular_distance(
                    pts_kpca[idxs[i]], pts_kpca[idxs[j]]))

        mean_pca = sum(pca_dists) / len(pca_dists)
        mean_kpca = sum(kpca_dists) / len(kpca_dists)
        marker = " <-" if mean_kpca < mean_pca else ""
        print(f"  {cat:<20} {mean_pca:>12.4f} {mean_kpca:>12.4f}{marker}")

    # ── 6. Cartesian + geo coordinates ───────────────────────────────
    header("6. Coordinate conversions")

    pt = kpca.project(embeddings[0])
    cart = pt.to_cartesian()
    geo = pt.to_geo()

    print(f"  Spherical:  r={pt.r:.4f}, theta={pt.theta:.4f}, "
          f"phi={pt.phi:.4f}")
    print(f"  Cartesian:  x={cart.x:.4f}, y={cart.y:.4f}, z={cart.z:.4f}")
    print(f"  Geographic: lat={geo.lat:.4f}, lon={geo.lon:.4f}")

    # ── 7. Custom sigma ──────────────────────────────────────────────
    header("7. Tuning sigma")
    print("  sigma controls the Gaussian kernel bandwidth.")
    print("  Smaller = more local structure. Larger = smoother.\n")

    for sigma in [0.1, 0.5, 1.0, 5.0]:
        k = sphereql.KernelPcaProjection.fit(embeddings, sigma=sigma)
        pt = k.project(embeddings[0])
        print(f"  sigma={sigma:<5} -> explained_var={k.explained_variance_ratio:.4f}, "
              f"theta={pt.theta:.4f}, phi={pt.phi:.4f}")

    # ── 8. Volumetric mode ───────────────────────────────────────────
    header("8. Volumetric mode")
    print("  Volumetric mode lets radius vary with embedding magnitude,")
    print("  placing high-norm embeddings further from the origin.\n")

    kpca_vol = sphereql.KernelPcaProjection.fit(
        embeddings, volumetric=True)
    kpca_flat = sphereql.KernelPcaProjection.fit(embeddings)

    sample_idxs = [0, 25, 50, 75]
    print(f"  {'Index':<8} {'Text':<40} {'r (flat)':>10} {'r (vol)':>10}")
    print(f"  {'-' * 8} {'-' * 40} {'-' * 10} {'-' * 10}")
    for idx in sample_idxs:
        flat_pt = kpca_flat.project(embeddings[idx])
        vol_pt = kpca_vol.project(embeddings[idx])
        text = texts[idx][:38] + ".." if len(texts[idx]) > 40 else texts[idx]
        print(f"  {idx:<8} {text:<40} {flat_pt.r:>10.4f} {vol_pt.r:>10.4f}")

    # ── 9. Out-of-sample projection ─────────────────────────────────
    header("9. Out-of-sample projection")
    print("  Kernel PCA can project new embeddings not seen during fit.\n")

    novel = [x * 0.9 + 0.1 for x in embeddings[0]]
    novel_pt = kpca.project(novel)
    original_pt = kpca.project(embeddings[0])
    dist = angular_distance(novel_pt, original_pt)

    print(f"  Original:  theta={original_pt.theta:.4f}, "
          f"phi={original_pt.phi:.4f}")
    print(f"  Perturbed: theta={novel_pt.theta:.4f}, "
          f"phi={novel_pt.phi:.4f}")
    print(f"  Angular distance: {dist:.4f} rad "
          f"({math.degrees(dist):.2f} deg)")

    # ── Done ─────────────────────────────────────────────────────────
    header("Done!")
    print("Kernel PCA captures nonlinear relationships in embedding space")
    print("that linear PCA cannot. Use it when your data has curved or")
    print("clustered structure that a linear projection undersells.\n")
    print("Key API:")
    print("  KernelPcaProjection.fit(embs)              # auto sigma")
    print("  KernelPcaProjection.fit(embs, sigma=0.5)   # explicit")
    print("  kpca.project(emb)        -> SphericalPoint")
    print("  kpca.project_rich(emb)   -> ProjectedPoint (+ certainty)")
    print("  kpca.project_batch(embs) -> [SphericalPoint, ...]")
    print()


if __name__ == "__main__":
    main()
