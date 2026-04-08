use std::sync::Arc;

use sphereql_index::SpatialIndexBuilder;

use crate::query::{PointIndex, SphericalQueryRoot};
use crate::subscription::{SpatialEventBus, SphericalSubscriptionRoot};

pub type SphericalSchema = async_graphql::Schema<
    SphericalQueryRoot,
    async_graphql::EmptyMutation,
    SphericalSubscriptionRoot,
>;

pub fn build_schema(index: PointIndex, event_bus: SpatialEventBus) -> SphericalSchema {
    async_graphql::Schema::build(
        SphericalQueryRoot,
        async_graphql::EmptyMutation,
        SphericalSubscriptionRoot,
    )
    .data(index)
    .data(event_bus)
    .finish()
}

pub fn create_default_index() -> PointIndex {
    Arc::new(tokio::sync::RwLock::new(
        SpatialIndexBuilder::new()
            .uniform_shells(5, 10.0)
            .theta_divisions(12)
            .phi_divisions(6)
            .build(),
    ))
}

pub fn create_schema_with_defaults() -> SphericalSchema {
    build_schema(create_default_index(), SpatialEventBus::new(256))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    use sphereql_core::SphericalPoint;

    use crate::query::PointItem;

    fn point(r: f64, theta: f64, phi: f64) -> SphericalPoint {
        SphericalPoint::new_unchecked(r, theta, phi)
    }

    fn item(id: &str, r: f64, theta: f64, phi: f64) -> PointItem {
        PointItem {
            id: id.into(),
            position: point(r, theta, phi),
        }
    }

    async fn schema_with_items(items: Vec<PointItem>) -> SphericalSchema {
        let index = create_default_index();
        {
            let mut idx = index.write().await;
            for it in items {
                idx.insert(it);
            }
        }
        build_schema(index, SpatialEventBus::new(16))
    }

    #[tokio::test]
    async fn test_within_cone_query() {
        // Three items: two near (theta=0.5, phi=PI/4) and one far away
        let schema = schema_with_items(vec![
            item("a", 1.0, 0.5, FRAC_PI_4),
            item("b", 1.0, 0.6, FRAC_PI_4 + 0.1),
            item("c", 1.0, 2.5, FRAC_PI_2 + 1.0),
        ])
        .await;

        let res = schema
            .execute(
                r#"{ withinCone(
                    cone: {
                        apex: { r: 0.0, theta: 0.0, phi: 0.0 },
                        axis: { r: 1.0, theta: 0.5, phi: 0.7854 },
                        halfAngle: 0.5
                    }
                ) { items { r theta phi } totalScanned } }"#,
            )
            .await;

        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
        let data = res.data.into_json().unwrap();
        let items = data["withinCone"]["items"].as_array().unwrap();
        assert!(
            items.len() >= 2,
            "expected at least 2 items in cone, got {}",
            items.len()
        );
    }

    #[tokio::test]
    async fn test_within_shell_query() {
        // Items at different radii: r=1, r=3, r=7
        let schema = schema_with_items(vec![
            item("near", 1.0, 0.5, FRAC_PI_4),
            item("mid", 3.0, 0.5, FRAC_PI_4),
            item("far", 7.0, 0.5, FRAC_PI_4),
        ])
        .await;

        let res = schema
            .execute(r#"{ withinShell(shell: { inner: 2.0, outer: 5.0 }) { items { r } totalScanned } }"#)
            .await;

        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
        let data = res.data.into_json().unwrap();
        let items = data["withinShell"]["items"].as_array().unwrap();
        assert_eq!(items.len(), 1, "only the r=3 item should be in shell [2,5]");
        let r = items[0]["r"].as_f64().unwrap();
        assert!((r - 3.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_nearest_to_query() {
        let schema = schema_with_items(vec![
            item("a", 1.0, 0.5, FRAC_PI_4),
            item("b", 1.0, 0.6, FRAC_PI_4 + 0.1),
            item("c", 1.0, 2.0, FRAC_PI_2),
        ])
        .await;

        let res = schema
            .execute(
                r#"{ nearestTo(
                    point: { r: 1.0, theta: 0.5, phi: 0.7854 },
                    k: 2
                ) { point { r theta phi } distance } }"#,
            )
            .await;

        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
        let data = res.data.into_json().unwrap();
        let results = data["nearestTo"].as_array().unwrap();
        assert_eq!(results.len(), 2, "expected 2 nearest results");

        let d0 = results[0]["distance"].as_f64().unwrap();
        let d1 = results[1]["distance"].as_f64().unwrap();
        assert!(d0 <= d1, "results should be sorted by distance ascending");
    }

    #[tokio::test]
    async fn test_distance_between_query() {
        let schema = create_schema_with_defaults();

        let res = schema
            .execute(
                r#"{ distanceBetween(
                    a: { r: 1.0, theta: 0.0, phi: 0.7854 },
                    b: { r: 1.0, theta: 1.5, phi: 1.2 }
                ) }"#,
            )
            .await;

        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
        let data = res.data.into_json().unwrap();
        let distance = data["distanceBetween"].as_f64().unwrap();
        assert!(distance > 0.0, "distance between distinct points should be positive");
    }

    #[tokio::test]
    async fn test_within_region_query() {
        let schema = schema_with_items(vec![
            item("in_shell", 3.0, 0.5, FRAC_PI_4),
            item("out_shell", 8.0, 0.5, FRAC_PI_4),
        ])
        .await;

        let res = schema
            .execute(
                r#"{ withinRegion(
                    region: { shell: { inner: 2.0, outer: 5.0 } }
                ) { items { r } totalScanned } }"#,
            )
            .await;

        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
        let data = res.data.into_json().unwrap();
        let items = data["withinRegion"]["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        let r = items[0]["r"].as_f64().unwrap();
        assert!((r - 3.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_empty_index_queries() {
        let schema = create_schema_with_defaults();

        let cone_res = schema
            .execute(
                r#"{ withinCone(
                    cone: {
                        apex: { r: 0.0, theta: 0.0, phi: 0.0 },
                        axis: { r: 1.0, theta: 0.5, phi: 0.7854 },
                        halfAngle: 1.0
                    }
                ) { items { r } totalScanned } }"#,
            )
            .await;
        assert!(cone_res.errors.is_empty(), "cone errors: {:?}", cone_res.errors);
        let data = cone_res.data.into_json().unwrap();
        let items = data["withinCone"]["items"].as_array().unwrap();
        assert!(items.is_empty());

        let nearest_res = schema
            .execute(
                r#"{ nearestTo(
                    point: { r: 1.0, theta: 0.5, phi: 0.7854 },
                    k: 5
                ) { point { r } distance } }"#,
            )
            .await;
        assert!(nearest_res.errors.is_empty(), "nearest errors: {:?}", nearest_res.errors);
        let data = nearest_res.data.into_json().unwrap();
        let results = data["nearestTo"].as_array().unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_schema_sdl_contains_expected_types() {
        let schema = create_schema_with_defaults();
        let sdl = schema.sdl();

        let expected = [
            "SphericalPointOutput",
            "SpatialQueryResultOutput",
            "NearestResultOutput",
            "withinCone",
            "withinShell",
            "nearestTo",
            "distanceBetween",
            "SphericalPointInput",
        ];

        for token in &expected {
            assert!(
                sdl.contains(token),
                "SDL missing expected token '{}'. SDL:\n{}",
                token,
                &sdl[..sdl.len().min(2000)],
            );
        }
    }
}
