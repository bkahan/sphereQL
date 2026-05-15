use sphereql_core::{SphericalPoint, angular_distance};
use sphereql_embed::{Embedding, LaplacianEigenmapProjection, Projection, RadialStrategy};

fn main() {
    let concepts = sphereql_corpus::build_corpus();
    let embeddings: Vec<Embedding> = concepts
        .iter()
        .enumerate()
        .map(|(i, c)| Embedding::new(sphereql_corpus::embed(&c.features, i as u64)))
        .collect();

    // active-set sizes
    let active_sizes: Vec<usize> = embeddings
        .iter()
        .map(|e| e.values.iter().filter(|v| v.abs() > 0.05).count())
        .collect();
    if active_sizes.is_empty() {
        eprintln!("corpus is empty — nothing to diagnose");
        return;
    }
    let min_a = *active_sizes.iter().min().unwrap();
    let max_a = *active_sizes.iter().max().unwrap();
    let mean_a: f64 =
        active_sizes.iter().map(|&x| x as f64).sum::<f64>() / active_sizes.len() as f64;
    println!("active set size: min={min_a} mean={mean_a:.1} max={max_a} (of 128)");

    let lap = match LaplacianEigenmapProjection::fit(&embeddings, RadialStrategy::Magnitude) {
        Ok(lap) => lap,
        Err(e) => {
            eprintln!("Laplacian fit failed: {e}");
            return;
        }
    };
    println!("eigenvalues: {:?}", lap.eigenvalues());
    println!("connectivity_ratio: {}", lap.connectivity_ratio());

    let coords: Vec<SphericalPoint> = embeddings.iter().map(|e| lap.project(e)).collect();
    let unique_thetas: std::collections::BTreeSet<i64> = coords
        .iter()
        .map(|p| (p.theta * 1e6).round() as i64)
        .collect();
    println!("unique theta buckets (×1e6): {}", unique_thetas.len());
    let max_d = coords
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            coords
                .iter()
                .skip(i + 1)
                .map(move |q| angular_distance(p, q))
        })
        .fold(0.0_f64, f64::max);
    println!("max pairwise angular distance among coords: {max_d:.6}");
}
