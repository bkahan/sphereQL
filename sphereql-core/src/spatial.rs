//! Spatial analysis primitives on S².
//!
//! Pure geometry operating on [`SphericalPoint`] and `[f64; 3]` unit vectors.
//! No embedding or category knowledge — those live in `sphereql-embed`.
//!
//! Covers: antipodal maps, spherical cap areas and intersections, great circle
//! arc geometry, spherical Voronoi tessellation, spherical excess, lune
//! containment, and Monte Carlo coverage estimation.

use std::f64::consts::PI;

use crate::distance::angular_distance;
use crate::types::SphericalPoint;

// ── §1  Antipodal geometry ─────────────────────────────────────────────

/// Returns the antipodal point of `p` on S² (angular distance π).
///
/// The antipode has θ′ = θ + π (mod 2π) and φ′ = π − φ.
/// Radial coordinate is preserved.
pub fn antipode(p: &SphericalPoint) -> SphericalPoint {
    let theta = (p.theta + PI) % std::f64::consts::TAU;
    let phi = PI - p.phi;
    SphericalPoint::new_unchecked(p.r, theta, phi)
}

/// Measures how coherently a set of points clusters near a target on S².
///
/// Returns the fraction of `points` within `radius` radians of `center`,
/// divided by the expected fraction for a uniform distribution on S²
/// (which equals the cap's solid angle / 4π).
///
/// Values > 1 mean the region is denser than chance; < 1 means sparser.
/// Returns 0.0 if `points` is empty.
#[must_use]
pub fn region_coherence(center: &SphericalPoint, radius: f64, points: &[SphericalPoint]) -> f64 {
    if points.is_empty() || radius <= 0.0 {
        return 0.0;
    }
    let hits = points
        .iter()
        .filter(|p| angular_distance(center, p) <= radius)
        .count();
    let observed = hits as f64 / points.len() as f64;
    let expected = cap_solid_angle(radius) / (4.0 * PI);
    if expected < f64::EPSILON {
        return 0.0;
    }
    observed / expected
}

// ── §2 / §5  Spherical cap areas and coverage ─────────────────────────

/// Solid angle of a spherical cap with half-angle `alpha` (radians).
///
/// Formula: Ω = 2π(1 − cos α). Returns a value in [0, 4π].
#[must_use]
#[inline]
pub fn cap_solid_angle(alpha: f64) -> f64 {
    2.0 * PI * (1.0 - alpha.cos())
}

/// Solid angle of the intersection of two spherical caps.
///
/// Given two caps (center_a, α_a) and (center_b, α_b), computes the
/// solid angle of their overlap region on S².
///
/// Uses the analytic formula for two-cap intersection via the lens
/// integral. Returns 0.0 if the caps don't overlap.
#[must_use]
pub fn cap_intersection_area(
    center_a: &SphericalPoint,
    alpha_a: f64,
    center_b: &SphericalPoint,
    alpha_b: f64,
) -> f64 {
    let d = angular_distance(center_a, center_b);

    // No overlap: caps are too far apart
    if d >= alpha_a + alpha_b {
        return 0.0;
    }

    // One cap contains the other entirely
    if d + alpha_b <= alpha_a {
        return cap_solid_angle(alpha_b);
    }
    if d + alpha_a <= alpha_b {
        return cap_solid_angle(alpha_a);
    }

    // The lens (intersection) area is the sum of two spherical cap sectors
    // minus the two copies of the spherical triangle formed by the cap
    // centers and an intersection point.
    //
    // Derivation (Mazonka 2012):
    //   Ω = 2·φ_a·(1 − cos α_a) + 2·φ_b·(1 − cos α_b) − 2·E
    // where φ_x is the dihedral half-angle of the lens at cap X's pole,
    // and E is the spherical excess of the triangle (center_a, center_b,
    // intersection_point) with side lengths (α_a, α_b, d), computed via
    // L'Huilier's theorem.

    let cos_d = d.cos();
    let sin_d = d.sin();
    let cos_a = alpha_a.cos();
    let cos_b = alpha_b.cos();

    if sin_d.abs() < 1e-15 {
        return cap_solid_angle(alpha_a.min(alpha_b));
    }

    let phi_a = ((cos_b - cos_a * cos_d) / (alpha_a.sin() * sin_d))
        .clamp(-1.0, 1.0)
        .acos();
    let phi_b = ((cos_a - cos_b * cos_d) / (alpha_b.sin() * sin_d))
        .clamp(-1.0, 1.0)
        .acos();

    // Spherical excess of the triangle with sides (α_a, α_b, d)
    let s = (alpha_a + alpha_b + d) / 2.0;
    let product = ((s / 2.0).tan()
        * ((s - alpha_a) / 2.0).tan()
        * ((s - alpha_b) / 2.0).tan()
        * ((s - d) / 2.0).tan())
    .max(0.0);
    let excess = 4.0 * product.sqrt().atan();

    2.0 * phi_a * (1.0 - cos_a) + 2.0 * phi_b * (1.0 - cos_b) - 2.0 * excess
}

/// Which side of an angular bisector a point falls on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LuneSide {
    CloserToA,
    CloserToB,
    OnBisector,
}

/// Determines which side of the angular bisector plane between `a` and `b`
/// a `point` lies on.
///
/// The bisector plane is defined by the great circle equidistant from `a`
/// and `b`. Points on A's side are angularly closer to A than to B.
#[must_use]
pub fn lune_classify(a: &SphericalPoint, b: &SphericalPoint, point: &SphericalPoint) -> LuneSide {
    let da = angular_distance(a, point);
    let db = angular_distance(b, point);
    let diff = da - db;
    if diff.abs() < 1e-12 {
        LuneSide::OnBisector
    } else if diff < 0.0 {
        LuneSide::CloserToA
    } else {
        LuneSide::CloserToB
    }
}

/// Returns the unit normal of the angular bisector plane between `a` and `b`.
///
/// The bisector plane passes through the origin and is perpendicular to the
/// great circle arc from `a` to `b`, positioned at the midpoint.
/// Points with dot(normal, p) > 0 are on A's side.
#[must_use]
pub fn angular_bisector_normal(a: &SphericalPoint, b: &SphericalPoint) -> [f64; 3] {
    let ac = a.unit_cartesian();
    let bc = b.unit_cartesian();
    let nx = ac[0] - bc[0];
    let ny = ac[1] - bc[1];
    let nz = ac[2] - bc[2];
    let mag = (nx * nx + ny * ny + nz * nz).sqrt();
    if mag < 1e-15 {
        return [0.0, 0.0, 1.0];
    }
    [nx / mag, ny / mag, nz / mag]
}

// ── §3  Great circle arc geometry ──────────────────────────────────────

/// Angular distance from `point` to the nearest point on the great circle
/// arc from `arc_start` to `arc_end`.
///
/// The arc is the shorter path (< π) between the two endpoints.
/// Returns a value in [0, π].
#[must_use]
pub fn distance_to_great_circle_arc(
    point: &SphericalPoint,
    arc_start: &SphericalPoint,
    arc_end: &SphericalPoint,
) -> f64 {
    let p = point.unit_cartesian();
    let a = arc_start.unit_cartesian();
    let b = arc_end.unit_cartesian();

    let nx = a[1] * b[2] - a[2] * b[1];
    let ny = a[2] * b[0] - a[0] * b[2];
    let nz = a[0] * b[1] - a[1] * b[0];
    let nmag = (nx * nx + ny * ny + nz * nz).sqrt();

    if nmag < 1e-15 {
        return angular_distance(point, arc_start);
    }

    let dot_pn = p[0] * nx / nmag + p[1] * ny / nmag + p[2] * nz / nmag;
    let gc_dist = (PI / 2.0 - dot_pn.abs().acos()).abs();

    let proj_x = p[0] - dot_pn * nx / nmag;
    let proj_y = p[1] - dot_pn * ny / nmag;
    let proj_z = p[2] - dot_pn * nz / nmag;
    let proj_mag = (proj_x * proj_x + proj_y * proj_y + proj_z * proj_z).sqrt();

    if proj_mag < 1e-15 {
        return angular_distance(point, arc_start).min(angular_distance(point, arc_end));
    }

    let cp = [proj_x / proj_mag, proj_y / proj_mag, proj_z / proj_mag];

    // Is `cp` on the short arc between `arc_start` and `arc_end`?
    // `cp`, `a`, and `b` are all unit vectors on the same great circle
    // (by construction), so both `a·cp` and `b·cp` are cos(arc-hop). The
    // arc-between-endpoints test reduces to: both dot products ≥ cos(
    // arc_len) ≡ dot(a, b). Cheaper than converting `cp` back to
    // spherical just to call `angular_distance` three times.
    let ab_cos = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let acp_cos = a[0] * cp[0] + a[1] * cp[1] + a[2] * cp[2];
    let bcp_cos = b[0] * cp[0] + b[1] * cp[1] + b[2] * cp[2];

    if acp_cos >= ab_cos - 1e-10 && bcp_cos >= ab_cos - 1e-10 {
        gc_dist
    } else {
        angular_distance(point, arc_start).min(angular_distance(point, arc_end))
    }
}

/// Finds all indices in `points` within `epsilon` radians of the great circle
/// arc from `arc_start` to `arc_end`.
///
/// Returns `(index, distance)` pairs sorted by distance.
pub fn geodesic_sweep(
    arc_start: &SphericalPoint,
    arc_end: &SphericalPoint,
    points: &[SphericalPoint],
    epsilon: f64,
) -> Vec<(usize, f64)> {
    let mut hits: Vec<(usize, f64)> = points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let d = distance_to_great_circle_arc(p, arc_start, arc_end);
            if d <= epsilon { Some((i, d)) } else { None }
        })
        .collect();
    hits.sort_by(|a, b| a.1.total_cmp(&b.1));
    hits
}

/// Samples the great circle arc at `num_samples` equally spaced points and
/// counts how many items in `points` lie within `radius` of each sample.
///
/// Returns a density profile: `density[i]` is the count at the i-th sample.
/// Empty stretches (density = 0) indicate interdisciplinary gaps.
pub fn geodesic_density_profile(
    start: &SphericalPoint,
    end: &SphericalPoint,
    points: &[SphericalPoint],
    radius: f64,
    num_samples: usize,
) -> Vec<usize> {
    let num_samples = num_samples.max(2);
    let mut profile = Vec::with_capacity(num_samples);
    let arc_len = angular_distance(start, end);
    if arc_len < 1e-15 {
        return vec![0; num_samples];
    }

    for i in 0..num_samples {
        let t = i as f64 / (num_samples - 1) as f64;
        let sample = crate::interpolation::slerp(start, end, t);
        let count = points
            .iter()
            .filter(|p| angular_distance(&sample, p) <= radius)
            .count();
        profile.push(count);
    }
    profile
}

// ── §4  Spherical Voronoi tessellation ─────────────────────────────────

/// A cell in the spherical Voronoi diagram.
///
/// NOTE: item-level Voronoi (beyond category centroids) may be explored
/// in a future version. For now, the input is expected to be O(10–100)
/// category centroids.
#[derive(Debug, Clone)]
pub struct VoronoiCell {
    pub generator_index: usize,
    pub neighbor_indices: Vec<usize>,
    /// Approximate cell area in steradians, estimated via Monte Carlo.
    pub area: f64,
}

/// Computes a spherical Voronoi tessellation from generator points.
///
/// Uses Monte Carlo sampling to estimate cell areas and determines Voronoi
/// neighbors from boundary-proximity detection.
///
/// `num_samples` controls precision: 100_000 gives ~1% relative error
/// for 31 cells; 1_000_000 gives ~0.3%.
///
/// NOTE: exact Delaunay-on-S² via convex hull may be added later.
pub fn spherical_voronoi(generators: &[SphericalPoint], num_samples: usize) -> Vec<VoronoiCell> {
    let n = generators.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![VoronoiCell {
            generator_index: 0,
            neighbor_indices: Vec::new(),
            area: 4.0 * PI,
        }];
    }

    let gen_carts: Vec<[f64; 3]> = generators.iter().map(|g| g.unit_cartesian()).collect();

    let mut cell_counts = vec![0usize; n];
    let mut neighbor_hits = vec![vec![false; n]; n];

    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1337;

    for _ in 0..num_samples {
        let (x, y, z) = uniform_sphere_sample(&mut rng_state);

        let mut best_idx = 0;
        let mut best_dot = f64::NEG_INFINITY;
        let mut second_dot = f64::NEG_INFINITY;
        let mut second_idx = 0;

        for (i, gc) in gen_carts.iter().enumerate() {
            let dot = x * gc[0] + y * gc[1] + z * gc[2];
            if dot > best_dot {
                second_dot = best_dot;
                second_idx = best_idx;
                best_dot = dot;
                best_idx = i;
            } else if dot > second_dot {
                second_dot = dot;
                second_idx = i;
            }
        }

        cell_counts[best_idx] += 1;

        let margin = best_dot - second_dot;
        if margin < 0.05 {
            neighbor_hits[best_idx][second_idx] = true;
            neighbor_hits[second_idx][best_idx] = true;
        }
    }

    let total_area = 4.0 * PI;
    let total_samples = num_samples as f64;

    (0..n)
        .map(|i| {
            let neighbor_indices: Vec<usize> =
                (0..n).filter(|&j| j != i && neighbor_hits[i][j]).collect();
            VoronoiCell {
                generator_index: i,
                neighbor_indices,
                area: total_area * cell_counts[i] as f64 / total_samples,
            }
        })
        .collect()
}

// ── §2  Coverage and void detection (Monte Carlo) ──────────────────────

/// Result of a sphere coverage analysis.
#[derive(Debug, Clone)]
pub struct CoverageReport {
    pub coverage_fraction: f64,
    pub covered_area: f64,
    /// NOTE: exact inclusion-exclusion may be added in a future version.
    pub overlap_area: f64,
    pub void_count: usize,
    pub total_samples: usize,
}

/// Estimates sphere coverage via Monte Carlo sampling.
///
/// Each "cap" is (center, half-angle). A sample is "covered" if it falls
/// within at least one cap. Accuracy ~ O(1/√num_samples).
pub fn estimate_coverage(
    centers: &[SphericalPoint],
    half_angles: &[f64],
    num_samples: usize,
) -> CoverageReport {
    assert_eq!(centers.len(), half_angles.len());
    let n = centers.len();

    let cap_carts: Vec<[f64; 3]> = centers.iter().map(|c| c.unit_cartesian()).collect();
    let cos_alphas: Vec<f64> = half_angles.iter().map(|a| a.cos()).collect();

    let mut covered = 0usize;
    let mut multi_covered = 0usize;
    let mut rng_state: u64 = 0x1234_5678_9ABC_DEF0;

    for _ in 0..num_samples {
        let (x, y, z) = uniform_sphere_sample(&mut rng_state);
        let mut hit_count = 0usize;

        for i in 0..n {
            let dot = x * cap_carts[i][0] + y * cap_carts[i][1] + z * cap_carts[i][2];
            if dot >= cos_alphas[i] {
                hit_count += 1;
            }
        }

        if hit_count > 0 {
            covered += 1;
        }
        if hit_count > 1 {
            multi_covered += 1;
        }
    }

    let total = num_samples as f64;
    let total_area = 4.0 * PI;
    let coverage_fraction = covered as f64 / total;

    CoverageReport {
        coverage_fraction,
        covered_area: coverage_fraction * total_area,
        overlap_area: multi_covered as f64 / total * total_area,
        void_count: num_samples - covered,
        total_samples: num_samples,
    }
}

/// Angular distance from `point` to the nearest cap boundary.
///
/// Positive = outside all caps (void). Negative = inside a cap.
#[must_use]
pub fn void_distance(
    point: &SphericalPoint,
    centers: &[SphericalPoint],
    half_angles: &[f64],
) -> f64 {
    assert_eq!(centers.len(), half_angles.len());
    let mut min_gap = f64::INFINITY;
    for (center, &alpha) in centers.iter().zip(half_angles.iter()) {
        let d = angular_distance(point, center);
        let gap = d - alpha;
        if gap < min_gap {
            min_gap = gap;
        }
    }
    min_gap
}

// ── §5  Pairwise overlap and exclusivity ───────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct PairwiseOverlap {
    pub category_a: usize,
    pub category_b: usize,
    pub intersection_area: f64,
}

/// Computes pairwise cap overlaps, sorted by descending intersection area.
pub fn pairwise_overlaps(centers: &[SphericalPoint], half_angles: &[f64]) -> Vec<PairwiseOverlap> {
    assert_eq!(centers.len(), half_angles.len());
    let n = centers.len();
    let mut overlaps = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            let area =
                cap_intersection_area(&centers[i], half_angles[i], &centers[j], half_angles[j]);
            if area > 1e-15 {
                overlaps.push(PairwiseOverlap {
                    category_a: i,
                    category_b: j,
                    intersection_area: area,
                });
            }
        }
    }

    overlaps.sort_by(|a, b| {
        b.intersection_area
            .partial_cmp(&a.intersection_area)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    overlaps
}

/// Exclusivity: fraction of a cap's area not overlapped by any other cap.
/// Returns [0, 1]. 1.0 = fully exclusive.
#[must_use]
pub fn cap_exclusivity(
    cap_index: usize,
    centers: &[SphericalPoint],
    half_angles: &[f64],
    num_samples: usize,
) -> f64 {
    let n = centers.len();
    assert!(cap_index < n);

    let cap_carts: Vec<[f64; 3]> = centers.iter().map(|c| c.unit_cartesian()).collect();
    let cos_alphas: Vec<f64> = half_angles.iter().map(|a| a.cos()).collect();

    let center = &cap_carts[cap_index];
    let cos_alpha = cos_alphas[cap_index];

    let mut in_cap = 0usize;
    let mut exclusive = 0usize;
    let mut rng_state: u64 = 0xFEED_FACE_0000_0001 + cap_index as u64;

    for _ in 0..num_samples {
        let (x, y, z) = uniform_sphere_sample(&mut rng_state);
        let dot = x * center[0] + y * center[1] + z * center[2];
        if dot < cos_alpha {
            continue;
        }
        in_cap += 1;

        let mut only_this = true;
        for (j, gc) in cap_carts.iter().enumerate() {
            if j == cap_index {
                continue;
            }
            let d = x * gc[0] + y * gc[1] + z * gc[2];
            if d >= cos_alphas[j] {
                only_this = false;
                break;
            }
        }
        if only_this {
            exclusive += 1;
        }
    }

    if in_cap == 0 {
        return 1.0;
    }
    exclusive as f64 / in_cap as f64
}

// ── §6  Spherical excess and curvature signatures ──────────────────────

/// Spherical excess (= area) of a triangle on S² with vertices a, b, c.
///
/// Uses L'Huilier's theorem:
///   E = 4·arctan(√[tan(s/2)·tan((s−a)/2)·tan((s−b)/2)·tan((s−c)/2)])
#[must_use]
pub fn spherical_excess(a: &SphericalPoint, b: &SphericalPoint, c: &SphericalPoint) -> f64 {
    let side_a = angular_distance(b, c);
    let side_b = angular_distance(a, c);
    let side_c = angular_distance(a, b);

    let s = (side_a + side_b + side_c) / 2.0;

    let t0 = (s / 2.0).tan();
    let t1 = ((s - side_a) / 2.0).tan();
    let t2 = ((s - side_b) / 2.0).tan();
    let t3 = ((s - side_c) / 2.0).tan();

    let product = t0 * t1 * t2 * t3;
    if product < 0.0 {
        return 0.0;
    }
    4.0 * product.sqrt().atan()
}

/// Curvature signature: distribution of spherical excesses across all
/// triples that include the point at `target`. Sorted ascending.
pub fn curvature_signature(target: usize, all_points: &[SphericalPoint]) -> Vec<f64> {
    let n = all_points.len();
    if n < 3 || target >= n {
        return Vec::new();
    }
    let mut excesses = Vec::new();
    for i in 0..n {
        if i == target {
            continue;
        }
        for j in (i + 1)..n {
            if j == target {
                continue;
            }
            let e = spherical_excess(&all_points[target], &all_points[i], &all_points[j]);
            excesses.push(e);
        }
    }
    excesses.sort_by(|a, b| a.total_cmp(b));
    excesses
}

// ── §3  Geodesic deviation ─────────────────────────────────────────────

/// Max angular distance from any interior path waypoint to the direct
/// great circle arc between first and last waypoints.
#[must_use]
pub fn geodesic_deviation(path: &[SphericalPoint]) -> f64 {
    if path.len() < 3 {
        return 0.0;
    }
    let start = path.first().unwrap();
    let end = path.last().unwrap();
    path[1..path.len() - 1]
        .iter()
        .map(|p| distance_to_great_circle_arc(p, start, end))
        .fold(0.0_f64, f64::max)
}

// ── Internal: uniform sphere sampling ──────────────────────────────────

#[inline]
fn next_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

#[inline]
fn uniform_sphere_sample(state: &mut u64) -> (f64, f64, f64) {
    let theta = std::f64::consts::TAU * next_f64(state);
    let cos_phi = 2.0 * next_f64(state) - 1.0;
    let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
    let (sin_t, cos_t) = theta.sin_cos();
    (sin_phi * cos_t, sin_phi * sin_t, cos_phi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn unit(theta: f64, phi: f64) -> SphericalPoint {
        SphericalPoint::new_unchecked(1.0, theta, phi)
    }

    #[test]
    fn antipode_of_north_pole() {
        let north = unit(0.0, 0.0);
        let south = antipode(&north);
        assert_relative_eq!(south.phi, PI, epsilon = 1e-12);
        assert_relative_eq!(angular_distance(&north, &south), PI, epsilon = 1e-12);
    }

    #[test]
    fn antipode_is_involution() {
        let p = unit(1.3, 0.7);
        let pp = antipode(&antipode(&p));
        assert_relative_eq!(angular_distance(&p, &pp), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn antipode_always_distance_pi() {
        for &(t, p) in &[(0.0, FRAC_PI_2), (1.0, 0.3), (3.0, 2.5), (5.5, 1.0)] {
            let pt = unit(t, p);
            let ap = antipode(&pt);
            assert_relative_eq!(angular_distance(&pt, &ap), PI, epsilon = 1e-10);
        }
    }

    #[test]
    fn region_coherence_all_at_center() {
        let c = unit(0.0, FRAC_PI_2);
        let points = vec![c; 100];
        let coh = region_coherence(&c, 0.1, &points);
        assert!(coh > 100.0);
    }

    #[test]
    fn region_coherence_empty() {
        let c = unit(0.0, FRAC_PI_2);
        assert_eq!(region_coherence(&c, 0.1, &[]), 0.0);
    }

    #[test]
    fn cap_solid_angle_hemisphere() {
        assert_relative_eq!(cap_solid_angle(FRAC_PI_2), 2.0 * PI, epsilon = 1e-12);
    }

    #[test]
    fn cap_solid_angle_full_sphere() {
        assert_relative_eq!(cap_solid_angle(PI), 4.0 * PI, epsilon = 1e-12);
    }

    #[test]
    fn cap_solid_angle_zero() {
        assert_relative_eq!(cap_solid_angle(0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn cap_solid_angle_small() {
        let alpha: f64 = 0.1;
        let expected = 2.0 * PI * (1.0 - alpha.cos());
        assert_relative_eq!(cap_solid_angle(alpha), expected, epsilon = 1e-12);
    }

    #[test]
    fn cap_intersection_no_overlap() {
        let a = unit(0.0, 0.1);
        let b = unit(PI, PI - 0.1);
        assert!(cap_intersection_area(&a, 0.2, &b, 0.2) < 1e-10);
    }

    #[test]
    fn cap_intersection_full_containment() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(0.0, FRAC_PI_2);
        let area = cap_intersection_area(&a, FRAC_PI_2, &b, FRAC_PI_4);
        assert_relative_eq!(area, cap_solid_angle(FRAC_PI_4), epsilon = 1e-10);
    }

    #[test]
    fn cap_intersection_identical_caps() {
        let a = unit(0.5, 1.0);
        let area = cap_intersection_area(&a, 0.3, &a, 0.3);
        assert_relative_eq!(area, cap_solid_angle(0.3), epsilon = 1e-10);
    }

    #[test]
    fn cap_intersection_symmetric() {
        let a = unit(0.0, FRAC_PI_4);
        let b = unit(0.5, FRAC_PI_2);
        let ab = cap_intersection_area(&a, 0.5, &b, 0.3);
        let ba = cap_intersection_area(&b, 0.3, &a, 0.5);
        assert_relative_eq!(ab, ba, epsilon = 1e-10);
    }

    #[test]
    fn cap_intersection_positive_for_overlapping() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(0.3, FRAC_PI_2);
        let area = cap_intersection_area(&a, 0.5, &b, 0.5);
        assert!(area > 0.0);
        assert!(area < cap_solid_angle(0.5));
    }

    #[test]
    fn distance_to_arc_endpoint() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(FRAC_PI_2, FRAC_PI_2);
        assert!(distance_to_great_circle_arc(&a, &a, &b) < 1e-10);
    }

    #[test]
    fn distance_to_arc_midpoint_on_arc() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(FRAC_PI_2, FRAC_PI_2);
        let mid = crate::interpolation::slerp(&a, &b, 0.5);
        assert!(distance_to_great_circle_arc(&mid, &a, &b) < 1e-10);
    }

    #[test]
    fn distance_to_arc_pole_is_pi_over_2() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(FRAC_PI_2, FRAC_PI_2);
        let pole = unit(0.0, 0.0);
        assert_relative_eq!(
            distance_to_great_circle_arc(&pole, &a, &b),
            FRAC_PI_2,
            epsilon = 1e-6
        );
    }

    #[test]
    fn geodesic_sweep_finds_nearby() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(FRAC_PI_2, FRAC_PI_2);
        let mid = crate::interpolation::slerp(&a, &b, 0.5);
        let far = unit(0.0, 0.1);
        let hits = geodesic_sweep(&a, &b, &[mid, far], 0.1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);
    }

    #[test]
    fn geodesic_density_profile_shape() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(FRAC_PI_2, FRAC_PI_2);
        let mid = crate::interpolation::slerp(&a, &b, 0.5);
        let profile = geodesic_density_profile(&a, &b, &[mid], 0.1, 10);
        assert_eq!(profile.len(), 10);
        assert!(profile.iter().sum::<usize>() > 0);
    }

    #[test]
    fn geodesic_deviation_straight_path() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(FRAC_PI_2, FRAC_PI_2);
        let mid = crate::interpolation::slerp(&a, &b, 0.5);
        assert!(geodesic_deviation(&[a, mid, b]) < 1e-8);
    }

    #[test]
    fn geodesic_deviation_detour() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(FRAC_PI_2, FRAC_PI_2);
        let detour = unit(0.0, 0.1);
        assert!(geodesic_deviation(&[a, detour, b]) > 1.0);
    }

    #[test]
    fn voronoi_single_point() {
        let cells = spherical_voronoi(&[unit(0.0, FRAC_PI_2)], 1000);
        assert_eq!(cells.len(), 1);
        assert_relative_eq!(cells[0].area, 4.0 * PI, epsilon = 1e-12);
    }

    #[test]
    fn voronoi_antipodal_pair_equal_area() {
        let a = unit(0.0, FRAC_PI_2);
        let b = antipode(&a);
        let cells = spherical_voronoi(&[a, b], 200_000);
        assert_eq!(cells.len(), 2);
        assert!((cells[0].area - cells[1].area).abs() < 0.5);
        assert!(cells[0].neighbor_indices.contains(&1));
    }

    #[test]
    fn voronoi_total_area_is_4pi() {
        let gens: Vec<SphericalPoint> = (0..6).map(|i| unit(i as f64 * 1.0, FRAC_PI_2)).collect();
        let cells = spherical_voronoi(&gens, 100_000);
        let total: f64 = cells.iter().map(|c| c.area).sum();
        assert_relative_eq!(total, 4.0 * PI, epsilon = 0.5);
    }

    #[test]
    fn coverage_single_hemisphere() {
        let c = unit(0.0, 0.0);
        let report = estimate_coverage(&[c], &[FRAC_PI_2], 100_000);
        assert!((report.coverage_fraction - 0.5).abs() < 0.02);
    }

    #[test]
    fn coverage_full_sphere() {
        let c = unit(0.0, 0.0);
        let report = estimate_coverage(&[c], &[PI], 10_000);
        assert!((report.coverage_fraction - 1.0).abs() < 0.01);
    }

    #[test]
    fn coverage_empty() {
        let report = estimate_coverage(&[], &[], 10_000);
        assert_eq!(report.coverage_fraction, 0.0);
        assert_eq!(report.void_count, 10_000);
    }

    #[test]
    fn void_distance_inside_cap() {
        let c = unit(0.0, FRAC_PI_2);
        let p = unit(0.05, FRAC_PI_2);
        assert!(void_distance(&p, &[c], &[0.5]) < 0.0);
    }

    #[test]
    fn void_distance_outside_all() {
        let c = unit(0.0, 0.1);
        let p = unit(PI, PI - 0.1);
        assert!(void_distance(&p, &[c], &[0.2]) > 0.0);
    }

    #[test]
    fn spherical_excess_right_triangle() {
        let a = unit(0.0, 0.0);
        let b = unit(0.0, FRAC_PI_2);
        let c = unit(FRAC_PI_2, FRAC_PI_2);
        assert_relative_eq!(spherical_excess(&a, &b, &c), FRAC_PI_2, epsilon = 1e-6);
    }

    #[test]
    fn spherical_excess_degenerate() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(0.5, FRAC_PI_2);
        let c = crate::interpolation::slerp(&a, &b, 0.5);
        assert!(spherical_excess(&a, &b, &c) < 1e-6);
    }

    #[test]
    fn spherical_excess_symmetric() {
        let a = unit(0.0, 0.5);
        let b = unit(1.0, 1.0);
        let c = unit(2.0, 0.8);
        let abc = spherical_excess(&a, &b, &c);
        let bca = spherical_excess(&b, &c, &a);
        assert_relative_eq!(abc, bca, epsilon = 1e-12);
    }

    #[test]
    fn curvature_signature_length() {
        let points: Vec<SphericalPoint> = (0..5).map(|i| unit(i as f64 * 1.0, FRAC_PI_2)).collect();
        assert_eq!(curvature_signature(0, &points).len(), 6);
    }

    #[test]
    fn lune_classify_closer_to_a() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(PI, FRAC_PI_2);
        assert_eq!(
            lune_classify(&a, &b, &unit(0.1, FRAC_PI_2)),
            LuneSide::CloserToA
        );
    }

    #[test]
    fn lune_classify_closer_to_b() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(PI, FRAC_PI_2);
        assert_eq!(
            lune_classify(&a, &b, &unit(PI - 0.1, FRAC_PI_2)),
            LuneSide::CloserToB
        );
    }

    #[test]
    fn lune_classify_on_bisector() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(PI, FRAC_PI_2);
        assert_eq!(
            lune_classify(&a, &b, &unit(FRAC_PI_2, FRAC_PI_2)),
            LuneSide::OnBisector
        );
    }

    #[test]
    fn angular_bisector_normal_sign() {
        let a = unit(0.0, FRAC_PI_2);
        let b = unit(FRAC_PI_2, FRAC_PI_2);
        let n = angular_bisector_normal(&a, &b);
        let ac = a.unit_cartesian();
        let bc = b.unit_cartesian();
        let dot_a = n[0] * ac[0] + n[1] * ac[1] + n[2] * ac[2];
        let dot_b = n[0] * bc[0] + n[1] * bc[1] + n[2] * bc[2];
        assert!(dot_a > dot_b);
    }

    #[test]
    fn cap_exclusivity_isolated() {
        let a = unit(0.0, 0.1);
        let b = unit(PI, PI - 0.1);
        let exc = cap_exclusivity(0, &[a, b], &[0.2, 0.2], 50_000);
        assert!(exc > 0.95, "got {exc}");
    }

    #[test]
    fn cap_exclusivity_identical_caps() {
        let a = unit(0.0, FRAC_PI_2);
        let exc = cap_exclusivity(0, &[a, a], &[0.5, 0.5], 50_000);
        assert!(exc < 0.05, "got {exc}");
    }
}
