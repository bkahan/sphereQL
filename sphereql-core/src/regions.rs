use std::f64::consts::PI;

use crate::distance::angular_distance;
use crate::error::SphereQlError;
use crate::types::SphericalPoint;

pub trait Contains {
    fn contains(&self, point: &SphericalPoint) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cone {
    pub apex: SphericalPoint,
    pub axis: SphericalPoint,
    pub half_angle: f64,
}

impl Cone {
    pub fn new(
        apex: SphericalPoint,
        axis: SphericalPoint,
        half_angle: f64,
    ) -> Result<Self, SphereQlError> {
        if half_angle <= 0.0 || half_angle > PI {
            return Err(SphereQlError::InvalidConeAngle(half_angle));
        }
        Ok(Self {
            apex,
            axis,
            half_angle,
        })
    }
}

impl Contains for Cone {
    fn contains(&self, point: &SphericalPoint) -> bool {
        let point_unit = SphericalPoint::new_unchecked(1.0, point.theta, point.phi);
        let axis_unit = SphericalPoint::new_unchecked(1.0, self.axis.theta, self.axis.phi);
        angular_distance(&point_unit, &axis_unit) <= self.half_angle
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cap {
    pub center: SphericalPoint,
    pub half_angle: f64,
}

impl Cap {
    pub fn new(center: SphericalPoint, half_angle: f64) -> Result<Self, SphereQlError> {
        if half_angle <= 0.0 || half_angle > PI {
            return Err(SphereQlError::InvalidConeAngle(half_angle));
        }
        Ok(Self { center, half_angle })
    }
}

impl Contains for Cap {
    fn contains(&self, point: &SphericalPoint) -> bool {
        let point_unit = SphericalPoint::new_unchecked(1.0, point.theta, point.phi);
        let center_unit = SphericalPoint::new_unchecked(1.0, self.center.theta, self.center.phi);
        angular_distance(&point_unit, &center_unit) <= self.half_angle
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Shell {
    pub inner: f64,
    pub outer: f64,
}

impl Shell {
    pub fn new(inner: f64, outer: f64) -> Result<Self, SphereQlError> {
        if inner < 0.0 || inner >= outer {
            return Err(SphereQlError::InvalidShellBounds { inner, outer });
        }
        Ok(Self { inner, outer })
    }
}

impl Contains for Shell {
    fn contains(&self, point: &SphericalPoint) -> bool {
        point.r >= self.inner && point.r <= self.outer
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Band {
    pub phi_min: f64,
    pub phi_max: f64,
}

impl Band {
    pub fn new(phi_min: f64, phi_max: f64) -> Result<Self, SphereQlError> {
        if phi_min < 0.0 || phi_min >= phi_max || phi_max > PI {
            return Err(SphereQlError::InvalidBandBounds { phi_min, phi_max });
        }
        Ok(Self { phi_min, phi_max })
    }
}

impl Contains for Band {
    fn contains(&self, point: &SphericalPoint) -> bool {
        point.phi >= self.phi_min && point.phi <= self.phi_max
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Wedge {
    pub theta_min: f64,
    pub theta_max: f64,
}

impl Wedge {
    pub fn new(theta_min: f64, theta_max: f64) -> Self {
        Self {
            theta_min,
            theta_max,
        }
    }

    fn wraps(&self) -> bool {
        self.theta_min > self.theta_max
    }
}

impl Contains for Wedge {
    fn contains(&self, point: &SphericalPoint) -> bool {
        if self.wraps() {
            point.theta >= self.theta_min || point.theta <= self.theta_max
        } else {
            point.theta >= self.theta_min && point.theta <= self.theta_max
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Region {
    Cone(Cone),
    Cap(Cap),
    Shell(Shell),
    Band(Band),
    Wedge(Wedge),
    Intersection(Vec<Region>),
    Union(Vec<Region>),
}

impl Region {
    pub fn intersection(regions: Vec<Region>) -> Self {
        Region::Intersection(regions)
    }

    pub fn union(regions: Vec<Region>) -> Self {
        Region::Union(regions)
    }
}

impl Contains for Region {
    fn contains(&self, point: &SphericalPoint) -> bool {
        match self {
            Region::Cone(c) => c.contains(point),
            Region::Cap(c) => c.contains(point),
            Region::Shell(s) => s.contains(point),
            Region::Band(b) => b.contains(point),
            Region::Wedge(w) => w.contains(point),
            Region::Intersection(regions) => regions.iter().all(|r| r.contains(point)),
            Region::Union(regions) => regions.iter().any(|r| r.contains(point)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn point(r: f64, theta: f64, phi: f64) -> SphericalPoint {
        SphericalPoint::new_unchecked(r, theta, phi)
    }

    // --- Cone tests ---

    #[test]
    fn cone_contains_point_inside() {
        let cone = Cone::new(point(0.0, 0.0, 0.0), point(1.0, 0.0, FRAC_PI_4), FRAC_PI_2).unwrap();
        let p = point(2.0, 0.0, FRAC_PI_4);
        assert!(cone.contains(&p));
    }

    #[test]
    fn cone_excludes_point_outside() {
        let cone = Cone::new(point(0.0, 0.0, 0.0), point(1.0, 0.0, 0.1), 0.05).unwrap();
        let p = point(1.0, PI, FRAC_PI_2);
        assert!(!cone.contains(&p));
    }

    #[test]
    fn cone_contains_point_on_boundary() {
        let half = FRAC_PI_4;
        let cone = Cone::new(point(0.0, 0.0, 0.0), point(1.0, 0.0, 0.0), half).unwrap();
        // axis is at phi=0 (north pole), point at phi=half_angle should be on boundary
        let p = point(1.0, 0.0, half);
        assert!(cone.contains(&p));
    }

    #[test]
    fn cone_various_half_angles() {
        // Narrow cone
        let narrow = Cone::new(point(0.0, 0.0, 0.0), point(1.0, 0.0, FRAC_PI_2), 0.01).unwrap();
        let near = point(1.0, 0.0, FRAC_PI_2);
        let far = point(1.0, 0.0, FRAC_PI_2 + 0.1);
        assert!(narrow.contains(&near));
        assert!(!narrow.contains(&far));

        // Full hemisphere
        let wide = Cone::new(point(0.0, 0.0, 0.0), point(1.0, 0.0, FRAC_PI_2), FRAC_PI_2).unwrap();
        assert!(wide.contains(&point(1.0, 0.5, FRAC_PI_2 + 0.3)));
    }

    #[test]
    fn cone_invalid_half_angle() {
        assert!(Cone::new(point(0.0, 0.0, 0.0), point(1.0, 0.0, 0.0), 0.0).is_err());
        assert!(Cone::new(point(0.0, 0.0, 0.0), point(1.0, 0.0, 0.0), -0.1).is_err());
        assert!(Cone::new(point(0.0, 0.0, 0.0), point(1.0, 0.0, 0.0), PI + 0.1).is_err());
        // PI is valid
        assert!(Cone::new(point(0.0, 0.0, 0.0), point(1.0, 0.0, 0.0), PI).is_ok());
    }

    // --- Cap tests ---

    #[test]
    fn cap_contains_point_inside() {
        let cap = Cap::new(point(1.0, 0.0, FRAC_PI_2), FRAC_PI_4).unwrap();
        let p = point(5.0, 0.0, FRAC_PI_2);
        assert!(cap.contains(&p));
    }

    #[test]
    fn cap_excludes_point_outside() {
        let cap = Cap::new(point(1.0, 0.0, 0.1), 0.05).unwrap();
        let p = point(1.0, PI, PI - 0.1);
        assert!(!cap.contains(&p));
    }

    #[test]
    fn cap_ignores_radius() {
        let cap = Cap::new(point(1.0, 0.0, FRAC_PI_2), FRAC_PI_4).unwrap();
        let near = point(0.1, 0.0, FRAC_PI_2);
        let far = point(1000.0, 0.0, FRAC_PI_2);
        assert!(cap.contains(&near));
        assert!(cap.contains(&far));
    }

    #[test]
    fn cap_invalid_half_angle() {
        assert!(Cap::new(point(1.0, 0.0, 0.0), 0.0).is_err());
        assert!(Cap::new(point(1.0, 0.0, 0.0), -1.0).is_err());
    }

    // --- Shell tests ---

    #[test]
    fn shell_contains_point_inside() {
        let shell = Shell::new(1.0, 5.0).unwrap();
        assert!(shell.contains(&point(3.0, 0.0, 0.0)));
    }

    #[test]
    fn shell_excludes_point_outside() {
        let shell = Shell::new(1.0, 5.0).unwrap();
        assert!(!shell.contains(&point(0.5, 0.0, 0.0)));
        assert!(!shell.contains(&point(6.0, 0.0, 0.0)));
    }

    #[test]
    fn shell_boundary_inclusive() {
        let shell = Shell::new(1.0, 5.0).unwrap();
        assert!(shell.contains(&point(1.0, 0.0, 0.0)));
        assert!(shell.contains(&point(5.0, 0.0, 0.0)));
    }

    #[test]
    fn shell_invalid_bounds() {
        assert!(Shell::new(5.0, 1.0).is_err());
        assert!(Shell::new(3.0, 3.0).is_err());
        assert!(Shell::new(-1.0, 5.0).is_err());
    }

    // --- Band tests ---

    #[test]
    fn band_contains_point_inside() {
        let band = Band::new(FRAC_PI_4, 3.0 * FRAC_PI_4).unwrap();
        assert!(band.contains(&point(1.0, 0.0, FRAC_PI_2)));
    }

    #[test]
    fn band_excludes_point_outside() {
        let band = Band::new(FRAC_PI_4, FRAC_PI_2).unwrap();
        assert!(!band.contains(&point(1.0, 0.0, 0.1)));
        assert!(!band.contains(&point(1.0, 0.0, PI - 0.1)));
    }

    #[test]
    fn band_boundary_inclusive() {
        let band = Band::new(FRAC_PI_4, FRAC_PI_2).unwrap();
        assert!(band.contains(&point(1.0, 0.0, FRAC_PI_4)));
        assert!(band.contains(&point(1.0, 0.0, FRAC_PI_2)));
    }

    #[test]
    fn band_poles() {
        // Band covering north pole area
        let north = Band::new(0.0 + 0.001, FRAC_PI_4).unwrap();
        assert!(north.contains(&point(1.0, 0.0, 0.01)));
        assert!(!north.contains(&point(1.0, 0.0, FRAC_PI_2)));

        // Band covering south pole area
        let south = Band::new(3.0 * FRAC_PI_4, PI).unwrap();
        assert!(south.contains(&point(1.0, 0.0, PI - 0.1)));
        assert!(!south.contains(&point(1.0, 0.0, FRAC_PI_4)));
    }

    #[test]
    fn band_invalid_bounds() {
        assert!(Band::new(FRAC_PI_2, FRAC_PI_4).is_err());
        assert!(Band::new(FRAC_PI_4, FRAC_PI_4).is_err());
        assert!(Band::new(-0.1, FRAC_PI_2).is_err());
        assert!(Band::new(0.0, PI + 0.1).is_err());
    }

    // --- Wedge tests ---

    #[test]
    fn wedge_contains_normal_range() {
        let wedge = Wedge::new(0.5, 2.0);
        assert!(wedge.contains(&point(1.0, 1.0, FRAC_PI_2)));
        assert!(!wedge.contains(&point(1.0, 3.0, FRAC_PI_2)));
    }

    #[test]
    fn wedge_wraparound() {
        // 350° to 10° in radians: ~6.1087 to ~0.1745
        let theta_min = 350.0_f64.to_radians();
        let theta_max = 10.0_f64.to_radians();
        let wedge = Wedge::new(theta_min, theta_max);

        let inside_high = point(1.0, 355.0_f64.to_radians(), FRAC_PI_2);
        let inside_low = point(1.0, 5.0_f64.to_radians(), FRAC_PI_2);
        let outside = point(1.0, 180.0_f64.to_radians(), FRAC_PI_2);

        assert!(wedge.contains(&inside_high));
        assert!(wedge.contains(&inside_low));
        assert!(!wedge.contains(&outside));
    }

    #[test]
    fn wedge_boundary_inclusive() {
        let wedge = Wedge::new(1.0, 2.0);
        assert!(wedge.contains(&point(1.0, 1.0, FRAC_PI_2)));
        assert!(wedge.contains(&point(1.0, 2.0, FRAC_PI_2)));
    }

    // --- Compound region tests ---

    #[test]
    fn intersection_shell_and_band() {
        let shell = Region::Shell(Shell::new(1.0, 5.0).unwrap());
        let band = Region::Band(Band::new(FRAC_PI_4, 3.0 * FRAC_PI_4).unwrap());
        let region = Region::intersection(vec![shell, band]);

        // Inside both
        assert!(region.contains(&point(3.0, 0.0, FRAC_PI_2)));
        // Inside shell but outside band
        assert!(!region.contains(&point(3.0, 0.0, 0.1)));
        // Inside band but outside shell
        assert!(!region.contains(&point(10.0, 0.0, FRAC_PI_2)));
    }

    #[test]
    fn union_two_caps() {
        let cap_a = Region::Cap(Cap::new(point(1.0, 0.0, 0.1), 0.2).unwrap());
        let cap_b = Region::Cap(Cap::new(point(1.0, 0.0, PI - 0.1), 0.2).unwrap());
        let region = Region::union(vec![cap_a, cap_b]);

        // Near north pole (cap_a)
        assert!(region.contains(&point(1.0, 0.0, 0.05)));
        // Near south pole (cap_b)
        assert!(region.contains(&point(1.0, 0.0, PI - 0.05)));
        // Equator (neither)
        assert!(!region.contains(&point(1.0, 0.0, FRAC_PI_2)));
    }

    #[test]
    fn empty_intersection_contains_everything() {
        let region = Region::intersection(vec![]);
        assert!(region.contains(&point(1.0, 0.0, FRAC_PI_2)));
    }

    #[test]
    fn empty_union_contains_nothing() {
        let region = Region::union(vec![]);
        assert!(!region.contains(&point(1.0, 0.0, FRAC_PI_2)));
    }

    // --- Region enum dispatch ---

    #[test]
    fn region_dispatches_to_inner_types() {
        let shell_region = Region::Shell(Shell::new(1.0, 5.0).unwrap());
        assert!(shell_region.contains(&point(3.0, 0.0, 0.0)));
        assert!(!shell_region.contains(&point(10.0, 0.0, 0.0)));

        let wedge_region = Region::Wedge(Wedge::new(0.5, 2.0));
        assert!(wedge_region.contains(&point(1.0, 1.0, FRAC_PI_2)));
    }
}
