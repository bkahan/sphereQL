use std::sync::Arc;

use async_graphql::{Context, Object, Result};
use tokio::sync::RwLock;

use sphereql_core::{
    angular_distance, chord_distance, euclidean_distance, great_circle_distance,
    spherical_to_cartesian, SphericalPoint,
};
use sphereql_index::{SpatialIndex, SpatialItem};

use crate::types::{
    BandInput, ConeInput, DistanceMetric, NearestResultOutput, RegionInput, ShellInput,
    SpatialQueryResultOutput, SphericalPointInput, SphericalPointOutput,
};

#[derive(Debug, Clone)]
pub struct PointItem {
    pub id: String,
    pub position: SphericalPoint,
}

impl SpatialItem for PointItem {
    type Id = String;
    fn id(&self) -> &String {
        &self.id
    }
    fn position(&self) -> &SphericalPoint {
        &self.position
    }
}

pub type PointIndex = Arc<RwLock<SpatialIndex<PointItem>>>;

pub struct SphericalQueryRoot;

#[Object]
impl SphericalQueryRoot {
    async fn within_cone(
        &self,
        ctx: &Context<'_>,
        cone: ConeInput,
        limit: Option<i32>,
    ) -> Result<SpatialQueryResultOutput> {
        let core_cone = cone.to_core()?;
        let index = ctx
            .data::<PointIndex>()
            .map_err(|_| async_graphql::Error::new("SpatialIndex not found in context"))?;
        let idx = index.read().await;
        let result = idx.query_cone(&core_cone);

        let items: Vec<SphericalPointOutput> = result
            .items
            .iter()
            .map(|item| SphericalPointOutput::from(item.position()))
            .collect();

        let items = if let Some(n) = limit {
            items.into_iter().take(n.max(0) as usize).collect()
        } else {
            items
        };

        Ok(SpatialQueryResultOutput {
            items,
            total_scanned: result.total_scanned as i32,
        })
    }

    async fn within_shell(
        &self,
        ctx: &Context<'_>,
        shell: ShellInput,
        limit: Option<i32>,
    ) -> Result<SpatialQueryResultOutput> {
        let core_shell = shell.to_core()?;
        let index = ctx
            .data::<PointIndex>()
            .map_err(|_| async_graphql::Error::new("SpatialIndex not found in context"))?;
        let idx = index.read().await;
        let result = idx.query_shell(&core_shell);

        let items: Vec<SphericalPointOutput> = result
            .items
            .iter()
            .map(|item| SphericalPointOutput::from(item.position()))
            .collect();

        let items = if let Some(n) = limit {
            items.into_iter().take(n.max(0) as usize).collect()
        } else {
            items
        };

        Ok(SpatialQueryResultOutput {
            items,
            total_scanned: result.total_scanned as i32,
        })
    }

    async fn within_band(
        &self,
        ctx: &Context<'_>,
        band: BandInput,
        limit: Option<i32>,
    ) -> Result<SpatialQueryResultOutput> {
        let core_band = band.to_core()?;
        let index = ctx
            .data::<PointIndex>()
            .map_err(|_| async_graphql::Error::new("SpatialIndex not found in context"))?;
        let idx = index.read().await;
        let result = idx.query_band(&core_band);

        let items: Vec<SphericalPointOutput> = result
            .items
            .iter()
            .map(|item| SphericalPointOutput::from(item.position()))
            .collect();

        let items = if let Some(n) = limit {
            items.into_iter().take(n.max(0) as usize).collect()
        } else {
            items
        };

        Ok(SpatialQueryResultOutput {
            items,
            total_scanned: result.total_scanned as i32,
        })
    }

    async fn within_region(
        &self,
        ctx: &Context<'_>,
        region: RegionInput,
        limit: Option<i32>,
    ) -> Result<SpatialQueryResultOutput> {
        let core_region = region.to_core()?;
        let index = ctx
            .data::<PointIndex>()
            .map_err(|_| async_graphql::Error::new("SpatialIndex not found in context"))?;
        let idx = index.read().await;
        let result = idx.query_region(&core_region);

        let items: Vec<SphericalPointOutput> = result
            .items
            .iter()
            .map(|item| SphericalPointOutput::from(item.position()))
            .collect();

        let items = if let Some(n) = limit {
            items.into_iter().take(n.max(0) as usize).collect()
        } else {
            items
        };

        Ok(SpatialQueryResultOutput {
            items,
            total_scanned: result.total_scanned as i32,
        })
    }

    async fn nearest_to(
        &self,
        ctx: &Context<'_>,
        point: SphericalPointInput,
        k: i32,
        max_distance: Option<f64>,
    ) -> Result<Vec<NearestResultOutput>> {
        let core_point = point.to_core()?;
        let index = ctx
            .data::<PointIndex>()
            .map_err(|_| async_graphql::Error::new("SpatialIndex not found in context"))?;
        let idx = index.read().await;
        let results = idx.nearest(&core_point, k.max(0) as usize);

        let results: Vec<NearestResultOutput> = results
            .into_iter()
            .filter(|r| match max_distance {
                Some(max) => r.distance <= max,
                None => true,
            })
            .map(|r| NearestResultOutput {
                point: SphericalPointOutput::from(r.item.position()),
                distance: r.distance,
            })
            .collect();

        Ok(results)
    }

    async fn distance_between(
        &self,
        _ctx: &Context<'_>,
        a: SphericalPointInput,
        b: SphericalPointInput,
        metric: Option<DistanceMetric>,
        radius: Option<f64>,
    ) -> Result<f64> {
        let core_a = a.to_core()?;
        let core_b = b.to_core()?;
        let metric = metric.unwrap_or(DistanceMetric::Angular);

        let distance = match metric {
            DistanceMetric::Angular => angular_distance(&core_a, &core_b),
            DistanceMetric::GreatCircle => {
                great_circle_distance(&core_a, &core_b, radius.unwrap_or(1.0))
            }
            DistanceMetric::Chord => chord_distance(&core_a, &core_b),
            DistanceMetric::Euclidean => {
                let ca = spherical_to_cartesian(&core_a);
                let cb = spherical_to_cartesian(&core_b);
                euclidean_distance(&ca, &cb)
            }
        };

        Ok(distance)
    }
}
