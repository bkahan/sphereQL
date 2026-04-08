use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use sphereql::core::{Cone, Shell, SphericalPoint};
use sphereql::index::{SpatialIndexBuilder, SpatialItem};

#[derive(Debug, Clone)]
struct City {
    name: String,
    position: SphericalPoint,
}

impl SpatialItem for City {
    type Id = String;
    fn id(&self) -> &String {
        &self.name
    }
    fn position(&self) -> &SphericalPoint {
        &self.position
    }
}

fn city(name: &str, r: f64, theta: f64, phi: f64) -> City {
    City {
        name: name.to_string(),
        position: SphericalPoint::new_unchecked(r, theta, phi),
    }
}

fn main() {
    println!("=== SphereQL: Geospatial Indexing ===\n");

    let mut index = SpatialIndexBuilder::new()
        .uniform_shells(5, 10.0)
        .theta_divisions(12)
        .phi_divisions(6)
        .build::<City>();

    // Spread cities across a sphere at various radii, theta, phi
    let cities = vec![
        city("Alpha", 1.0, 0.2, 0.3),
        city("Bravo", 1.5, 0.5, FRAC_PI_4),
        city("Charlie", 2.0, 1.0, FRAC_PI_2),
        city("Delta", 2.5, 1.5, FRAC_PI_2),
        city("Echo", 3.0, 2.0, FRAC_PI_4),
        city("Foxtrot", 3.5, 2.5, 1.0),
        city("Golf", 4.0, 3.0, FRAC_PI_2),
        city("Hotel", 4.5, 3.5, 0.8),
        city("India", 5.0, 4.0, FRAC_PI_4),
        city("Juliet", 5.5, 4.5, 1.2),
        city("Kilo", 6.0, 5.0, FRAC_PI_2),
        city("Lima", 1.0, 5.5, PI - 0.3),
        city("Mike", 1.5, 0.0, PI - 0.5),
        city("November", 2.0, 0.8, 0.5),
        city("Oscar", 3.0, 1.2, 0.9),
        city("Papa", 7.0, 0.3, FRAC_PI_2),
        city("Quebec", 8.0, 1.0, 1.0),
        city("Romeo", 9.0, 2.0, FRAC_PI_2),
        city("Sierra", 0.5, 3.0, FRAC_PI_4),
        city("Tango", 1.2, 0.4, 0.6),
    ];

    for c in &cities {
        index.insert(c.clone());
    }
    println!("Inserted {} cities into the spatial index.\n", index.len());

    // Cone query: find cities near a particular direction
    println!("--- Cone Query ---");
    let axis = SphericalPoint::new_unchecked(1.0, 0.5, FRAC_PI_4);
    let cone = Cone::new(SphericalPoint::new_unchecked(0.0, 0.0, 0.0), axis, 0.6).unwrap();
    println!(
        "Looking for cities within 0.6 rad of direction (θ={:.2}, φ={:.2}):",
        axis.theta, axis.phi
    );

    let cone_result = index.query_cone(&cone);
    println!(
        "  Found {} items (scanned {})",
        cone_result.items.len(),
        cone_result.total_scanned
    );
    for c in &cone_result.items {
        let p = c.position();
        println!(
            "  - {} at r={:.1}, θ={:.2}, φ={:.2}",
            c.name, p.r, p.theta, p.phi
        );
    }

    // Shell query: find cities in a radius range
    println!("\n--- Shell Query ---");
    let shell = Shell::new(2.0, 4.0).unwrap();
    println!("Looking for cities with radius in [2.0, 4.0]:");

    let shell_result = index.query_shell(&shell);
    println!(
        "  Found {} items (scanned {})",
        shell_result.items.len(),
        shell_result.total_scanned
    );
    for c in &shell_result.items {
        let p = c.position();
        println!(
            "  - {} at r={:.1}, θ={:.2}, φ={:.2}",
            c.name, p.r, p.theta, p.phi
        );
    }

    // Nearest neighbors
    println!("\n--- Nearest Neighbors ---");
    let query_point = SphericalPoint::new_unchecked(1.0, 0.5, FRAC_PI_4);
    println!("5 nearest cities to (r=1.0, θ=0.50, φ={FRAC_PI_4:.2}):");

    let nearest = index.nearest(&query_point, 5);
    for (i, result) in nearest.iter().enumerate() {
        let p = result.item.position();
        println!(
            "  {}. {} (dist={:.4} rad) at r={:.1}, θ={:.2}, φ={:.2}",
            i + 1,
            result.item.name,
            result.distance,
            p.r,
            p.theta,
            p.phi
        );
    }

    // Within-distance query
    println!("\n--- Within Distance ---");
    let max_dist = 0.5;
    let within = index.within_distance(&query_point, max_dist);
    println!(
        "Cities within {max_dist} rad of query point ({} found, {} scanned):",
        within.items.len(),
        within.total_scanned
    );
    for c in &within.items {
        let p = c.position();
        println!(
            "  - {} at r={:.1}, θ={:.2}, φ={:.2}",
            c.name, p.r, p.theta, p.phi
        );
    }
}
