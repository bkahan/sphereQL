use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use sphereql_core::{Cone, Shell, SphericalPoint};
use sphereql_index::{SpatialIndex, SpatialIndexBuilder, SpatialItem};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

#[derive(Debug, Clone)]
struct BenchItem {
    id: usize,
    position: SphericalPoint,
}

impl SpatialItem for BenchItem {
    type Id = usize;
    fn id(&self) -> &usize {
        &self.id
    }
    fn position(&self) -> &SphericalPoint {
        &self.position
    }
}

fn make_items(n: usize) -> Vec<BenchItem> {
    (0..n)
        .map(|i| {
            let frac = i as f64 / n as f64;
            BenchItem {
                id: i,
                position: SphericalPoint::new_unchecked(
                    1.0 + (frac * 9.0),
                    (frac * TAU) % TAU,
                    frac * PI,
                ),
            }
        })
        .collect()
}

fn build_index(items: &[BenchItem]) -> SpatialIndex<BenchItem> {
    let mut index: SpatialIndex<BenchItem> = SpatialIndexBuilder::new()
        .uniform_shells(5, 10.0)
        .theta_divisions(12)
        .phi_divisions(6)
        .build();
    for item in items {
        index.insert(item.clone());
    }
    index
}

fn bench_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion");
    for count in [1_000, 10_000] {
        let items = make_items(count);
        group.bench_with_input(BenchmarkId::from_parameter(count), &items, |b, items| {
            b.iter(|| {
                let mut index: SpatialIndex<BenchItem> = SpatialIndexBuilder::new()
                    .uniform_shells(5, 10.0)
                    .theta_divisions(12)
                    .phi_divisions(6)
                    .build();
                for item in items {
                    index.insert(black_box(item.clone()));
                }
                index
            });
        });
    }
    group.finish();
}

fn bench_query_cone(c: &mut Criterion) {
    let items = make_items(10_000);
    let index = build_index(&items);
    let cone = Cone::new(
        SphericalPoint::new_unchecked(0.0, 0.0, 0.0),
        SphericalPoint::new_unchecked(1.0, 1.0, FRAC_PI_2),
        FRAC_PI_4,
    )
    .unwrap();

    c.bench_function("query_cone_10k", |b| {
        b.iter(|| index.query_cone(black_box(&cone)))
    });
}

fn bench_nearest(c: &mut Criterion) {
    let items = make_items(10_000);
    let index = build_index(&items);
    let query_point = SphericalPoint::new_unchecked(5.0, PI, FRAC_PI_2);

    c.bench_function("nearest_k5_10k", |b| {
        b.iter(|| index.nearest(black_box(&query_point), black_box(5)))
    });
}

fn bench_query_shell(c: &mut Criterion) {
    let items = make_items(10_000);
    let index = build_index(&items);
    let shell = Shell::new(2.0, 6.0).unwrap();

    c.bench_function("query_shell_10k", |b| {
        b.iter(|| index.query_shell(black_box(&shell)))
    });
}

criterion_group!(
    benches,
    bench_insertion,
    bench_query_cone,
    bench_nearest,
    bench_query_shell,
);
criterion_main!(benches);
