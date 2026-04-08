use criterion::{Criterion, black_box, criterion_group, criterion_main};
use sphereql_core::{
    SphericalPoint, angular_distance, great_circle_distance, slerp, spherical_to_cartesian,
};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

fn bench_spherical_point_new(c: &mut Criterion) {
    c.bench_function("SphericalPoint::new", |b| {
        b.iter(|| SphericalPoint::new(black_box(1.0), black_box(1.0), black_box(FRAC_PI_2)))
    });
}

fn bench_spherical_to_cartesian(c: &mut Criterion) {
    let point = SphericalPoint::new_unchecked(1.0, FRAC_PI_4, FRAC_PI_2);
    c.bench_function("spherical_to_cartesian", |b| {
        b.iter(|| spherical_to_cartesian(black_box(&point)))
    });
}

fn bench_angular_distance(c: &mut Criterion) {
    let a = SphericalPoint::new_unchecked(1.0, 0.3, 0.8);
    let b = SphericalPoint::new_unchecked(1.0, 2.1, 1.5);
    c.bench_function("angular_distance", |b_iter| {
        b_iter.iter(|| angular_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_great_circle_distance(c: &mut Criterion) {
    let a = SphericalPoint::new_unchecked(1.0, 0.3, 0.8);
    let b = SphericalPoint::new_unchecked(1.0, 2.1, 1.5);
    c.bench_function("great_circle_distance", |b_iter| {
        b_iter.iter(|| great_circle_distance(black_box(&a), black_box(&b), black_box(6371.0)))
    });
}

fn bench_slerp(c: &mut Criterion) {
    let a = SphericalPoint::new_unchecked(1.0, 0.0, FRAC_PI_2);
    let b = SphericalPoint::new_unchecked(1.0, PI, FRAC_PI_4);
    c.bench_function("slerp", |b_iter| {
        b_iter.iter(|| slerp(black_box(&a), black_box(&b), black_box(0.5)))
    });
}

criterion_group!(
    benches,
    bench_spherical_point_new,
    bench_spherical_to_cartesian,
    bench_angular_distance,
    bench_great_circle_distance,
    bench_slerp,
);
criterion_main!(benches);
