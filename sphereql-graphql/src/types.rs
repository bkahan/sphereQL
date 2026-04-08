use sphereql_core::{
    Band, Cap, Cone, Region, Shell, SphericalPoint, Wedge,
};

// --- Output types ---

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct SphericalPointOutput {
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
    pub theta_degrees: f64,
    pub phi_degrees: f64,
}

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct CartesianPointOutput {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct GeoPointOutput {
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
}

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct SpatialQueryResultOutput {
    pub items: Vec<SphericalPointOutput>,
    pub total_scanned: i32,
}

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct NearestResultOutput {
    pub point: SphericalPointOutput,
    pub distance: f64,
}

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct DistanceResultOutput {
    pub angular: f64,
    pub great_circle: Option<f64>,
    pub chord: f64,
}

// --- Input types ---

#[derive(async_graphql::InputObject, Debug, Clone)]
pub struct SphericalPointInput {
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
}

#[derive(async_graphql::InputObject, Debug, Clone)]
pub struct ConeInput {
    pub apex: SphericalPointInput,
    pub axis: SphericalPointInput,
    pub half_angle: f64,
}

#[derive(async_graphql::InputObject, Debug, Clone)]
pub struct CapInput {
    pub center: SphericalPointInput,
    pub half_angle: f64,
}

#[derive(async_graphql::InputObject, Debug, Clone)]
pub struct ShellInput {
    pub inner: f64,
    pub outer: f64,
}

#[derive(async_graphql::InputObject, Debug, Clone)]
pub struct BandInput {
    pub phi_min: f64,
    pub phi_max: f64,
}

#[derive(async_graphql::InputObject, Debug, Clone)]
pub struct WedgeInput {
    pub theta_min: f64,
    pub theta_max: f64,
}

#[derive(async_graphql::InputObject, Debug, Clone)]
pub struct RegionInput {
    pub cone: Option<ConeInput>,
    pub cap: Option<CapInput>,
    pub shell: Option<ShellInput>,
    pub band: Option<BandInput>,
    pub wedge: Option<WedgeInput>,
    pub intersection: Option<Vec<RegionInput>>,
    pub union: Option<Vec<RegionInput>>,
}

// --- Enum ---

#[derive(async_graphql::Enum, Copy, Clone, Eq, PartialEq, Debug)]
pub enum DistanceMetric {
    Angular,
    GreatCircle,
    Chord,
    Euclidean,
}

// --- Conversions ---

impl From<&SphericalPoint> for SphericalPointOutput {
    fn from(p: &SphericalPoint) -> Self {
        Self {
            r: p.r,
            theta: p.theta,
            phi: p.phi,
            theta_degrees: p.theta.to_degrees(),
            phi_degrees: p.phi.to_degrees(),
        }
    }
}

impl SphericalPointInput {
    pub fn to_core(&self) -> Result<SphericalPoint, async_graphql::Error> {
        SphericalPoint::new(self.r, self.theta, self.phi)
            .map_err(|e| async_graphql::Error::new(e.to_string()))
    }
}

impl ConeInput {
    pub fn to_core(&self) -> Result<Cone, async_graphql::Error> {
        let apex = self.apex.to_core()?;
        let axis = self.axis.to_core()?;
        Cone::new(apex, axis, self.half_angle)
            .map_err(|e| async_graphql::Error::new(e.to_string()))
    }
}

impl CapInput {
    pub fn to_core(&self) -> Result<Cap, async_graphql::Error> {
        let center = self.center.to_core()?;
        Cap::new(center, self.half_angle)
            .map_err(|e| async_graphql::Error::new(e.to_string()))
    }
}

impl ShellInput {
    pub fn to_core(&self) -> Result<Shell, async_graphql::Error> {
        Shell::new(self.inner, self.outer)
            .map_err(|e| async_graphql::Error::new(e.to_string()))
    }
}

impl BandInput {
    pub fn to_core(&self) -> Result<Band, async_graphql::Error> {
        Band::new(self.phi_min, self.phi_max)
            .map_err(|e| async_graphql::Error::new(e.to_string()))
    }
}

impl WedgeInput {
    pub fn to_core(&self) -> Result<Wedge, async_graphql::Error> {
        Ok(Wedge::new(self.theta_min, self.theta_max))
    }
}

impl RegionInput {
    pub fn to_core(&self) -> Result<Region, async_graphql::Error> {
        let set_count = [
            self.cone.is_some(),
            self.cap.is_some(),
            self.shell.is_some(),
            self.band.is_some(),
            self.wedge.is_some(),
            self.intersection.is_some(),
            self.union.is_some(),
        ]
        .iter()
        .filter(|&&v| v)
        .count();

        if set_count == 0 {
            return Err(async_graphql::Error::new(
                "RegionInput must have exactly one field set, but none were provided",
            ));
        }
        if set_count > 1 {
            return Err(async_graphql::Error::new(format!(
                "RegionInput must have exactly one field set, but {set_count} were provided",
            )));
        }

        if let Some(cone) = &self.cone {
            return Ok(Region::Cone(cone.to_core()?));
        }
        if let Some(cap) = &self.cap {
            return Ok(Region::Cap(cap.to_core()?));
        }
        if let Some(shell) = &self.shell {
            return Ok(Region::Shell(shell.to_core()?));
        }
        if let Some(band) = &self.band {
            return Ok(Region::Band(band.to_core()?));
        }
        if let Some(wedge) = &self.wedge {
            return Ok(Region::Wedge(wedge.to_core()?));
        }
        if let Some(regions) = &self.intersection {
            let converted: Result<Vec<Region>, _> =
                regions.iter().map(|r| r.to_core()).collect();
            return Ok(Region::Intersection(converted?));
        }
        if let Some(regions) = &self.union {
            let converted: Result<Vec<Region>, _> =
                regions.iter().map(|r| r.to_core()).collect();
            return Ok(Region::Union(converted?));
        }

        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn sp_input(r: f64, theta: f64, phi: f64) -> SphericalPointInput {
        SphericalPointInput { r, theta, phi }
    }

    #[test]
    fn spherical_point_input_to_core_roundtrip() {
        let input = sp_input(2.0, 1.0, FRAC_PI_4);
        let core = input.to_core().unwrap();
        assert!((core.r - 2.0).abs() < 1e-12);
        assert!((core.theta - 1.0).abs() < 1e-12);
        assert!((core.phi - FRAC_PI_4).abs() < 1e-12);

        let output = SphericalPointOutput::from(&core);
        assert!((output.r - 2.0).abs() < 1e-12);
        assert!((output.theta - 1.0).abs() < 1e-12);
        assert!((output.phi - FRAC_PI_4).abs() < 1e-12);
        assert!((output.theta_degrees - 1.0_f64.to_degrees()).abs() < 1e-9);
        assert!((output.phi_degrees - FRAC_PI_4.to_degrees()).abs() < 1e-9);
    }

    #[test]
    fn region_input_cone_converts() {
        let region = RegionInput {
            cone: Some(ConeInput {
                apex: sp_input(0.0, 0.0, 0.0),
                axis: sp_input(1.0, 0.5, FRAC_PI_2),
                half_angle: FRAC_PI_4,
            }),
            cap: None,
            shell: None,
            band: None,
            wedge: None,
            intersection: None,
            union: None,
        };
        let core = region.to_core().unwrap();
        assert!(matches!(core, Region::Cone(_)));
    }

    #[test]
    fn region_input_intersection_recursive() {
        let shell_region = RegionInput {
            cone: None,
            cap: None,
            shell: Some(ShellInput {
                inner: 1.0,
                outer: 5.0,
            }),
            band: None,
            wedge: None,
            intersection: None,
            union: None,
        };
        let band_region = RegionInput {
            cone: None,
            cap: None,
            shell: None,
            band: Some(BandInput {
                phi_min: FRAC_PI_4,
                phi_max: FRAC_PI_2,
            }),
            wedge: None,
            intersection: None,
            union: None,
        };
        let compound = RegionInput {
            cone: None,
            cap: None,
            shell: None,
            band: None,
            wedge: None,
            intersection: Some(vec![shell_region, band_region]),
            union: None,
        };

        let core = compound.to_core().unwrap();
        match core {
            Region::Intersection(regions) => {
                assert_eq!(regions.len(), 2);
                assert!(matches!(regions[0], Region::Shell(_)));
                assert!(matches!(regions[1], Region::Band(_)));
            }
            other => panic!("expected Intersection, got {other:?}"),
        }
    }

    #[test]
    fn invalid_inputs_produce_errors() {
        let bad_point = sp_input(-1.0, 0.0, 0.0);
        assert!(bad_point.to_core().is_err());

        let bad_shell = ShellInput {
            inner: 5.0,
            outer: 1.0,
        };
        assert!(bad_shell.to_core().is_err());

        let bad_band = BandInput {
            phi_min: PI,
            phi_max: 0.1,
        };
        assert!(bad_band.to_core().is_err());

        let empty_region = RegionInput {
            cone: None,
            cap: None,
            shell: None,
            band: None,
            wedge: None,
            intersection: None,
            union: None,
        };
        assert!(empty_region.to_core().is_err());

        let multi_region = RegionInput {
            cone: Some(ConeInput {
                apex: sp_input(0.0, 0.0, 0.0),
                axis: sp_input(1.0, 0.0, FRAC_PI_2),
                half_angle: FRAC_PI_4,
            }),
            cap: None,
            shell: Some(ShellInput {
                inner: 1.0,
                outer: 5.0,
            }),
            band: None,
            wedge: None,
            intersection: None,
            union: None,
        };
        assert!(multi_region.to_core().is_err());
    }
}
