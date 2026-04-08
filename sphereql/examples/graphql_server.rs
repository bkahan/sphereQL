use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use sphereql::core::SphericalPoint;
use sphereql::graphql::{PointItem, SpatialEventBus, build_schema, create_default_index};

#[tokio::main]
async fn main() {
    println!("=== SphereQL: GraphQL Schema Demo ===\n");

    let index = create_default_index();

    // Insert some sample points
    {
        let mut idx = index.write().await;
        let items = vec![
            PointItem {
                id: "alpha".into(),
                position: SphericalPoint::new_unchecked(1.0, 0.5, FRAC_PI_4),
            },
            PointItem {
                id: "bravo".into(),
                position: SphericalPoint::new_unchecked(1.0, 0.6, FRAC_PI_4 + 0.1),
            },
            PointItem {
                id: "charlie".into(),
                position: SphericalPoint::new_unchecked(3.0, 1.0, FRAC_PI_2),
            },
            PointItem {
                id: "delta".into(),
                position: SphericalPoint::new_unchecked(5.0, 2.0, FRAC_PI_2),
            },
            PointItem {
                id: "echo".into(),
                position: SphericalPoint::new_unchecked(7.0, 3.5, 1.2),
            },
        ];
        for item in items {
            idx.insert(item);
        }
        println!("Inserted {} items into the index.\n", idx.len());
    }

    let event_bus = SpatialEventBus::new(256);
    let schema = build_schema(index, event_bus);

    // Query 1: Find items within a cone
    println!("--- Query: withinCone ---");
    let cone_query = r#"{
        withinCone(cone: {
            apex: { r: 0.0, theta: 0.0, phi: 0.0 },
            axis: { r: 1.0, theta: 0.5, phi: 0.7854 },
            halfAngle: 0.5
        }) {
            items { r theta phi thetaDegrees phiDegrees }
            totalScanned
        }
    }"#;

    let res = schema.execute(cone_query).await;
    if !res.errors.is_empty() {
        eprintln!("  Errors: {:?}", res.errors);
    } else {
        let json = res.data.into_json().unwrap();
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    }

    // Query 2: Find items in a shell (radius range)
    println!("\n--- Query: withinShell ---");
    let shell_query = r#"{
        withinShell(shell: { inner: 2.0, outer: 6.0 }) {
            items { r theta phi }
            totalScanned
        }
    }"#;

    let res = schema.execute(shell_query).await;
    if !res.errors.is_empty() {
        eprintln!("  Errors: {:?}", res.errors);
    } else {
        let json = res.data.into_json().unwrap();
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    }

    // Query 3: Nearest neighbors
    println!("\n--- Query: nearestTo ---");
    let nearest_query = r#"{
        nearestTo(
            point: { r: 1.0, theta: 0.5, phi: 0.7854 },
            k: 3
        ) {
            point { r theta phi }
            distance
        }
    }"#;

    let res = schema.execute(nearest_query).await;
    if !res.errors.is_empty() {
        eprintln!("  Errors: {:?}", res.errors);
    } else {
        let json = res.data.into_json().unwrap();
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    }

    // Query 4: Distance between two points
    println!("\n--- Query: distanceBetween ---");
    let distance_query = r#"{
        distanceBetween(
            a: { r: 1.0, theta: 0.0, phi: 0.7854 },
            b: { r: 1.0, theta: 1.5708, phi: 1.5708 }
        )
    }"#;

    let res = schema.execute(distance_query).await;
    if !res.errors.is_empty() {
        eprintln!("  Errors: {:?}", res.errors);
    } else {
        let json = res.data.into_json().unwrap();
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    }
}
