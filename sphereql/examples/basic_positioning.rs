use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use sphereql::core::{
    SphericalPoint, angular_distance, cartesian_to_spherical, chord_distance,
    great_circle_distance, slerp, spherical_to_cartesian,
};

fn main() {
    println!("=== SphereQL: Basic Positioning ===\n");

    // Create points on a unit sphere
    let north_pole = SphericalPoint::new(1.0, 0.0, 0.0).unwrap();
    let equator_x = SphericalPoint::new(1.0, 0.0, FRAC_PI_2).unwrap();
    let equator_y = SphericalPoint::new(1.0, FRAC_PI_2, FRAC_PI_2).unwrap();
    let midlat = SphericalPoint::new(1.0, FRAC_PI_4, FRAC_PI_4).unwrap();

    println!("Points created:");
    println!("  North pole:  {north_pole:?}");
    println!("  Equator +X:  {equator_x:?}");
    println!("  Equator +Y:  {equator_y:?}");
    println!("  Mid-latitude: {midlat:?}");

    // Convert to Cartesian and back
    println!("\n--- Coordinate Conversions ---");
    for (name, point) in [
        ("North pole", north_pole),
        ("Equator +X", equator_x),
        ("Equator +Y", equator_y),
    ] {
        let cart = spherical_to_cartesian(&point);
        let roundtrip = cartesian_to_spherical(&cart);
        println!("\n  {name}:");
        println!(
            "    Spherical:  (r={:.4}, θ={:.4}, φ={:.4})",
            point.r, point.theta, point.phi
        );
        println!(
            "    Cartesian:  (x={:.4}, y={:.4}, z={:.4})",
            cart.x, cart.y, cart.z
        );
        println!(
            "    Roundtrip:  (r={:.4}, θ={:.4}, φ={:.4})",
            roundtrip.r, roundtrip.theta, roundtrip.phi
        );
    }

    // Compute distances
    println!("\n--- Distance Calculations ---");
    let earth_radius_km = 6371.0;
    let pairs = [
        ("North pole <-> Equator +X", north_pole, equator_x),
        ("Equator +X <-> Equator +Y", equator_x, equator_y),
        ("North pole <-> Mid-latitude", north_pole, midlat),
    ];

    for (label, a, b) in &pairs {
        let ang = angular_distance(a, b);
        let gc = great_circle_distance(a, b, earth_radius_km);
        let ch = chord_distance(a, b);
        println!("\n  {label}:");
        println!(
            "    Angular:      {ang:.6} rad ({:.2} deg)",
            ang.to_degrees()
        );
        println!("    Great-circle: {gc:.2} km (Earth radius)");
        println!("    Chord:        {ch:.6}");
    }

    // Interpolation with slerp
    println!("\n--- Slerp Interpolation (Equator +X -> Equator +Y) ---");
    for i in 0..=4 {
        let t = i as f64 / 4.0;
        let p = slerp(&equator_x, &equator_y, t);
        let cart = spherical_to_cartesian(&p);
        println!(
            "  t={t:.2}  =>  θ={:.4} rad, φ={:.4} rad  (x={:.4}, y={:.4}, z={:.4})",
            p.theta, p.phi, cart.x, cart.y, cart.z
        );
    }

    // Verify the slerp midpoint is equidistant from both endpoints
    let mid = slerp(&equator_x, &equator_y, 0.5);
    let d_to_x = angular_distance(&mid, &equator_x);
    let d_to_y = angular_distance(&mid, &equator_y);
    println!("\n  Midpoint distance to +X: {d_to_x:.6} rad");
    println!("  Midpoint distance to +Y: {d_to_y:.6} rad");
    println!("  Equidistant: {}", (d_to_x - d_to_y).abs() < 1e-12);

    // Quick demo of validation
    println!("\n--- Validation ---");
    match SphericalPoint::new(-1.0, 0.0, 0.0) {
        Ok(_) => println!("  Unexpected success with negative radius"),
        Err(e) => println!("  Negative radius rejected: {e}"),
    }
    match SphericalPoint::new(1.0, 0.0, PI + 0.1) {
        Ok(_) => println!("  Unexpected success with phi > PI"),
        Err(e) => println!("  phi > PI rejected: {e}"),
    }
}
