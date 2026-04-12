use crate::item::{SpatialItem, SpatialQueryResult};
use sphereql_core::{Band, Cap, Contains, SphericalPoint, Wedge, angular_distance};
use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

pub struct SectorIndex<T: SpatialItem> {
    theta_divisions: usize,
    phi_divisions: usize,
    sectors: Vec<Vec<T>>,
    item_map: HashMap<T::Id, usize>,
}

impl<T: SpatialItem> SectorIndex<T> {
    pub fn new(theta_divisions: usize, phi_divisions: usize) -> Self {
        assert!(theta_divisions >= 1, "theta_divisions must be >= 1");
        assert!(phi_divisions >= 1, "phi_divisions must be >= 1");

        let total = theta_divisions * phi_divisions;
        let mut sectors = Vec::with_capacity(total);
        for _ in 0..total {
            sectors.push(Vec::new());
        }

        Self {
            theta_divisions,
            phi_divisions,
            sectors,
            item_map: HashMap::new(),
        }
    }

    pub fn insert(&mut self, item: T) {
        let pos = item.position();
        let idx = self.sector_index(pos.theta, pos.phi);
        self.item_map.insert(item.id().clone(), idx);
        self.sectors[idx].push(item);
    }

    pub fn remove(&mut self, id: &T::Id) -> Option<T> {
        let sector_idx = self.item_map.remove(id)?;
        let sector = &mut self.sectors[sector_idx];
        let pos = sector.iter().position(|item| item.id() == id)?;
        Some(sector.swap_remove(pos))
    }

    pub fn get(&self, id: &T::Id) -> Option<&T> {
        let &sector_idx = self.item_map.get(id)?;
        self.sectors[sector_idx].iter().find(|item| item.id() == id)
    }

    pub fn len(&self) -> usize {
        self.item_map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.item_map.is_empty()
    }

    pub fn query_band(&self, band: &Band) -> SpatialQueryResult<T> {
        let mut items = Vec::new();
        let mut total_scanned = 0;

        for phi_idx in 0..self.phi_divisions {
            let (phi_min, phi_max) = self.phi_range(phi_idx);
            if phi_max < band.phi_min || phi_min > band.phi_max {
                continue;
            }
            for theta_idx in 0..self.theta_divisions {
                let sector_idx = theta_idx * self.phi_divisions + phi_idx;
                for item in &self.sectors[sector_idx] {
                    total_scanned += 1;
                    if band.contains(item.position()) {
                        items.push(item.clone());
                    }
                }
            }
        }

        SpatialQueryResult {
            items,
            total_scanned,
        }
    }

    pub fn query_wedge(&self, wedge: &Wedge) -> SpatialQueryResult<T> {
        let mut items = Vec::new();
        let mut total_scanned = 0;

        for theta_idx in 0..self.theta_divisions {
            let (t_min, t_max) = self.theta_range(theta_idx);
            if !self.theta_range_overlaps_wedge(t_min, t_max, wedge) {
                continue;
            }
            for phi_idx in 0..self.phi_divisions {
                let sector_idx = theta_idx * self.phi_divisions + phi_idx;
                for item in &self.sectors[sector_idx] {
                    total_scanned += 1;
                    if wedge.contains(item.position()) {
                        items.push(item.clone());
                    }
                }
            }
        }

        SpatialQueryResult {
            items,
            total_scanned,
        }
    }

    pub fn query_cone(
        &self,
        cone_center: &SphericalPoint,
        half_angle: f64,
    ) -> SpatialQueryResult<T> {
        let mut items = Vec::new();
        let mut total_scanned = 0;
        let threshold = half_angle + self.sector_diagonal();
        let center_unit = SphericalPoint::new_unchecked(1.0, cone_center.theta, cone_center.phi);

        for sector_idx in 0..self.sectors.len() {
            let center = self.sector_center(sector_idx);
            if angular_distance(&center_unit, &center) > threshold {
                continue;
            }
            for item in &self.sectors[sector_idx] {
                total_scanned += 1;
                let item_unit =
                    SphericalPoint::new_unchecked(1.0, item.position().theta, item.position().phi);
                if angular_distance(&center_unit, &item_unit) <= half_angle {
                    items.push(item.clone());
                }
            }
        }

        SpatialQueryResult {
            items,
            total_scanned,
        }
    }

    pub fn query_cap(&self, cap: &Cap) -> SpatialQueryResult<T> {
        let mut items = Vec::new();
        let mut total_scanned = 0;
        let threshold = cap.half_angle + self.sector_diagonal();
        let center_unit = SphericalPoint::new_unchecked(1.0, cap.center.theta, cap.center.phi);

        for sector_idx in 0..self.sectors.len() {
            let center = self.sector_center(sector_idx);
            if angular_distance(&center_unit, &center) > threshold {
                continue;
            }
            for item in &self.sectors[sector_idx] {
                total_scanned += 1;
                if cap.contains(item.position()) {
                    items.push(item.clone());
                }
            }
        }

        SpatialQueryResult {
            items,
            total_scanned,
        }
    }

    /// Returns all items from sectors whose angular center is within
    /// `proxy_threshold` cosine proxy distance of the query direction.
    ///
    /// Cosine proxy = `1 - dot(a, b)` for unit vectors, monotone with
    /// angular distance. The sector's angular diagonal is added as margin
    /// to ensure no edge items are missed.
    ///
    /// This is the fast path for sector-accelerated k-NN: it does only
    /// 3 muls + 2 adds per sector center (no trig), and returns references
    /// without cloning items.
    pub fn items_in_nearby_sectors<'a>(
        &'a self,
        query_cart: &[f64; 3],
        proxy_threshold: f64,
    ) -> Vec<&'a T> {
        let sector_margin = 1.0 - self.sector_diagonal().cos();
        let adjusted = proxy_threshold + sector_margin;
        let mut items = Vec::new();

        for sector_idx in 0..self.sectors.len() {
            let center = self.sector_center(sector_idx);
            let center_cart = center.unit_cartesian();
            let dot = query_cart[0] * center_cart[0]
                + query_cart[1] * center_cart[1]
                + query_cart[2] * center_cart[2];
            let proxy = 1.0 - dot.clamp(-1.0, 1.0);

            if proxy <= adjusted {
                items.extend(self.sectors[sector_idx].iter());
            }
        }

        items
    }

    pub fn all_items(&self) -> Vec<&T> {
        self.sectors.iter().flat_map(|s| s.iter()).collect()
    }

    fn sector_index(&self, theta: f64, phi: f64) -> usize {
        let theta_idx = ((theta / TAU) * self.theta_divisions as f64).floor() as usize;
        let phi_idx = ((phi / PI) * self.phi_divisions as f64).floor() as usize;

        let theta_idx = theta_idx.min(self.theta_divisions - 1);
        let phi_idx = phi_idx.min(self.phi_divisions - 1);

        theta_idx * self.phi_divisions + phi_idx
    }

    fn theta_range(&self, theta_idx: usize) -> (f64, f64) {
        let step = TAU / self.theta_divisions as f64;
        (theta_idx as f64 * step, (theta_idx + 1) as f64 * step)
    }

    fn phi_range(&self, phi_idx: usize) -> (f64, f64) {
        let step = PI / self.phi_divisions as f64;
        (phi_idx as f64 * step, (phi_idx + 1) as f64 * step)
    }

    pub(crate) fn sector_center(&self, sector_idx: usize) -> SphericalPoint {
        let theta_idx = sector_idx / self.phi_divisions;
        let phi_idx = sector_idx % self.phi_divisions;

        let (t_min, t_max) = self.theta_range(theta_idx);
        let (p_min, p_max) = self.phi_range(phi_idx);

        SphericalPoint::new_unchecked(1.0, (t_min + t_max) / 2.0, (p_min + p_max) / 2.0)
    }

    pub(crate) fn sector_diagonal(&self) -> f64 {
        let d_theta = TAU / self.theta_divisions as f64;
        let d_phi = PI / self.phi_divisions as f64;
        // Conservative upper bound: treat the angular extents as legs of a right triangle
        // on the sphere. The actual angular diagonal is at most this for small sectors,
        // and this overestimates for large sectors, which is the safe direction.
        (d_theta * d_theta + d_phi * d_phi).sqrt()
    }

    fn theta_range_overlaps_wedge(&self, t_min: f64, t_max: f64, wedge: &Wedge) -> bool {
        if wedge.theta_min <= wedge.theta_max {
            t_max >= wedge.theta_min && t_min <= wedge.theta_max
        } else {
            // Wrapping wedge: overlaps if sector is in the high range OR the low range
            t_max >= wedge.theta_min || t_min <= wedge.theta_max
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
    #[derive(Debug, Clone)]
    struct TestItem {
        id: u32,
        pos: SphericalPoint,
    }

    impl SpatialItem for TestItem {
        type Id = u32;
        fn id(&self) -> &u32 {
            &self.id
        }
        fn position(&self) -> &SphericalPoint {
            &self.pos
        }
    }

    fn item(id: u32, theta: f64, phi: f64) -> TestItem {
        TestItem {
            id,
            pos: SphericalPoint::new_unchecked(1.0, theta, phi),
        }
    }

    #[test]
    fn insert_and_get() {
        let mut index = SectorIndex::new(4, 4);
        index.insert(item(1, 0.5, 0.5));
        index.insert(item(2, 3.0, 1.5));

        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());
        assert!(index.get(&1).is_some());
        assert!(index.get(&2).is_some());
        assert!(index.get(&99).is_none());
    }

    #[test]
    fn correct_sector_placement() {
        let mut index = SectorIndex::new(4, 2);
        let a = item(1, 0.1, 0.1);
        let b = item(2, FRAC_PI_2 + 0.1, FRAC_PI_2 + 0.1);

        index.insert(a);
        index.insert(b);

        assert_eq!(index.sectors[0].len(), 1);
        assert_eq!(index.sectors[0][0].id, 1);
        assert_eq!(index.sectors[3].len(), 1);
        assert_eq!(index.sectors[3][0].id, 2);
    }

    #[test]
    fn remove_item() {
        let mut index = SectorIndex::new(4, 4);
        index.insert(item(1, 0.5, 0.5));
        index.insert(item(2, 0.5, 0.5));

        let removed = index.remove(&1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, 1);
        assert_eq!(index.len(), 1);
        assert!(index.get(&1).is_none());
        assert!(index.get(&2).is_some());
    }

    #[test]
    fn remove_nonexistent() {
        let mut index: SectorIndex<TestItem> = SectorIndex::new(4, 4);
        assert!(index.remove(&99).is_none());
    }

    #[test]
    fn query_band_filters_by_phi() {
        let mut index = SectorIndex::new(4, 8);
        index.insert(item(1, 1.0, 0.3));
        index.insert(item(2, 2.0, FRAC_PI_2));
        index.insert(item(3, 3.0, PI - 0.1));

        let band = Band::new(FRAC_PI_4, 3.0 * FRAC_PI_4).unwrap();
        let result = index.query_band(&band);

        let ids: Vec<u32> = result.items.iter().map(|i| i.id).collect();
        assert!(ids.contains(&2));
        assert!(!ids.contains(&1));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn query_band_skips_sectors() {
        let mut index = SectorIndex::new(4, 8);
        for i in 0..100 {
            index.insert(item(i, (i as f64 * 0.06) % TAU, 0.05));
        }
        index.insert(item(999, 1.0, FRAC_PI_2));

        let band = Band::new(FRAC_PI_4, 3.0 * FRAC_PI_4).unwrap();
        let result = index.query_band(&band);

        assert_eq!(result.items.len(), 1);
        assert!(result.total_scanned < index.len());
    }

    #[test]
    fn query_wedge_filters_by_theta() {
        let mut index = SectorIndex::new(8, 4);
        index.insert(item(1, 0.5, FRAC_PI_2));
        index.insert(item(2, 3.0, FRAC_PI_2));
        index.insert(item(3, 5.5, FRAC_PI_2));

        let wedge = Wedge::new(0.2, 1.0).unwrap();
        let result = index.query_wedge(&wedge);

        let ids: Vec<u32> = result.items.iter().map(|i| i.id).collect();
        assert!(ids.contains(&1));
        assert!(!ids.contains(&2));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn query_wedge_wraparound() {
        let mut index = SectorIndex::new(8, 4);
        index.insert(item(1, 6.0, FRAC_PI_2));
        index.insert(item(2, 0.1, FRAC_PI_2));
        index.insert(item(3, 3.0, FRAC_PI_2));

        let wedge = Wedge::new(5.5, 0.3).unwrap();
        let result = index.query_wedge(&wedge);

        let ids: Vec<u32> = result.items.iter().map(|i| i.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn query_cone_returns_nearby_items() {
        let mut index = SectorIndex::new(8, 8);
        let center = SphericalPoint::new_unchecked(1.0, 1.0, FRAC_PI_2);

        index.insert(item(1, 1.0, FRAC_PI_2));
        index.insert(item(2, 1.05, FRAC_PI_2));
        index.insert(item(3, 4.0, FRAC_PI_2));

        let result = index.query_cone(&center, 0.2);

        let ids: Vec<u32> = result.items.iter().map(|i| i.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn query_cone_skips_distant_sectors() {
        let mut index = SectorIndex::new(16, 16);
        for i in 0..50 {
            let theta = PI + (i as f64 * 0.02);
            index.insert(item(i, theta, FRAC_PI_2));
        }
        index.insert(item(999, 0.1, FRAC_PI_4));

        let center = SphericalPoint::new_unchecked(1.0, 0.1, FRAC_PI_4);
        let result = index.query_cone(&center, 0.3);

        assert_eq!(result.items.len(), 1);
        assert!(result.total_scanned < index.len());
    }

    #[test]
    fn query_cap() {
        let mut index = SectorIndex::new(8, 8);
        index.insert(item(1, 0.5, 0.5));
        index.insert(item(2, 0.6, 0.5));
        index.insert(item(3, 3.5, 2.5));

        let cap = Cap::new(SphericalPoint::new_unchecked(1.0, 0.5, 0.5), 0.3).unwrap();
        let result = index.query_cap(&cap);

        let ids: Vec<u32> = result.items.iter().map(|i| i.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn all_items_returns_everything() {
        let mut index = SectorIndex::new(4, 4);
        index.insert(item(1, 0.5, 0.5));
        index.insert(item(2, 3.0, 1.5));
        index.insert(item(3, 5.0, 2.5));

        let all = index.all_items();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn single_sector_index() {
        let mut index = SectorIndex::new(1, 1);
        index.insert(item(1, 0.0, 0.0));
        index.insert(item(2, 3.0, FRAC_PI_2));
        index.insert(item(3, 5.0, PI));

        assert_eq!(index.len(), 3);
        assert_eq!(index.sectors.len(), 1);
        assert_eq!(index.sectors[0].len(), 3);

        let band = Band::new(FRAC_PI_4, 3.0 * FRAC_PI_4).unwrap();
        let result = index.query_band(&band);
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.items[0].id, 2);
    }

    #[test]
    fn items_at_sector_boundaries() {
        let mut index = SectorIndex::new(4, 4);
        index.insert(item(1, 0.0, PI));
        index.insert(item(2, TAU - 0.001, FRAC_PI_2));

        assert_eq!(index.len(), 2);
        assert!(index.get(&1).is_some());
        assert!(index.get(&2).is_some());
    }

    #[test]
    fn empty_index() {
        let index: SectorIndex<TestItem> = SectorIndex::new(4, 4);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!(index.all_items().is_empty());
    }

    #[test]
    #[should_panic(expected = "theta_divisions must be >= 1")]
    fn zero_theta_divisions_panics() {
        SectorIndex::<TestItem>::new(0, 4);
    }

    #[test]
    #[should_panic(expected = "phi_divisions must be >= 1")]
    fn zero_phi_divisions_panics() {
        SectorIndex::<TestItem>::new(4, 0);
    }

    #[test]
    fn items_in_nearby_sectors_returns_local_items() {
        let mut index = SectorIndex::new(8, 8);
        index.insert(item(1, 0.5, FRAC_PI_2));
        index.insert(item(2, PI + 0.5, FRAC_PI_2));

        let query_point = SphericalPoint::new_unchecked(1.0, 0.5, FRAC_PI_2);
        let query_cart = query_point.unit_cartesian();

        let results = index.items_in_nearby_sectors(&query_cart, 0.5);
        let ids: Vec<u32> = results.iter().map(|i| i.id).collect();
        assert!(ids.contains(&1));
        assert!(!ids.contains(&2));
    }
}
