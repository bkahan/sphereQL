use crate::item::{NearestResult, SpatialItem, SpatialQueryResult};
use crate::sector::SectorIndex;
use crate::shell::ShellIndex;
use sphereql_core::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub struct SpatialIndex<T: SpatialItem> {
    shell: ShellIndex<T>,
    sector: SectorIndex<T>,
}

pub struct SpatialIndexBuilder {
    shell_boundaries: Vec<f64>,
    theta_divisions: usize,
    phi_divisions: usize,
}

impl SpatialIndexBuilder {
    pub fn new() -> Self {
        Self {
            shell_boundaries: Vec::new(),
            theta_divisions: 12,
            phi_divisions: 6,
        }
    }

    pub fn shell_boundary(mut self, r: f64) -> Self {
        self.shell_boundaries.push(r);
        self
    }

    pub fn uniform_shells(mut self, count: usize, max_r: f64) -> Self {
        for i in 0..=count {
            self.shell_boundaries
                .push(max_r * i as f64 / count as f64);
        }
        self
    }

    pub fn theta_divisions(mut self, n: usize) -> Self {
        self.theta_divisions = n;
        self
    }

    pub fn phi_divisions(mut self, n: usize) -> Self {
        self.phi_divisions = n;
        self
    }

    pub fn build<T: SpatialItem>(self) -> SpatialIndex<T> {
        let mut shell_builder = ShellIndex::<T>::builder();
        for &b in &self.shell_boundaries {
            shell_builder = shell_builder.boundary(b);
        }

        SpatialIndex {
            shell: shell_builder.build(),
            sector: SectorIndex::new(self.theta_divisions, self.phi_divisions),
        }
    }
}

impl Default for SpatialIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: SpatialItem> SpatialIndex<T> {
    pub fn builder() -> SpatialIndexBuilder {
        SpatialIndexBuilder::new()
    }

    pub fn insert(&mut self, item: T) {
        self.sector.insert(item.clone());
        self.shell.insert(item);
    }

    pub fn remove(&mut self, id: &T::Id) -> Option<T> {
        self.sector.remove(id);
        self.shell.remove(id)
    }

    pub fn get(&self, id: &T::Id) -> Option<&T> {
        self.shell.get(id)
    }

    pub fn len(&self) -> usize {
        self.shell.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shell.is_empty()
    }

    pub fn update(&mut self, item: T) {
        self.remove(item.id());
        self.insert(item);
    }

    pub fn query_cone(&self, cone: &Cone) -> SpatialQueryResult<T> {
        let axis_unit = SphericalPoint::new_unchecked(1.0, cone.axis.theta, cone.axis.phi);
        let candidates = self.sector.query_cone(&axis_unit, cone.half_angle);
        let mut items = Vec::new();

        for item in &candidates.items {
            if cone.contains(item.position()) {
                items.push(item.clone());
            }
        }

        SpatialQueryResult {
            total_scanned: candidates.total_scanned,
            items,
        }
    }

    pub fn query_shell(&self, shell: &Shell) -> SpatialQueryResult<T> {
        self.shell.query_shell(shell)
    }

    pub fn query_band(&self, band: &Band) -> SpatialQueryResult<T> {
        self.sector.query_band(band)
    }

    pub fn query_region(&self, region: &Region) -> SpatialQueryResult<T> {
        let all = self.shell.all_items();
        let total_scanned = all.len();
        let items = all
            .into_iter()
            .filter(|item| region.contains(item.position()))
            .cloned()
            .collect();

        SpatialQueryResult {
            items,
            total_scanned,
        }
    }

    pub fn nearest(&self, point: &SphericalPoint, k: usize) -> Vec<NearestResult<T>> {
        if k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<DistanceEntry<T>> = BinaryHeap::new();

        for item in self.shell.all_items() {
            let distance = angular_distance(point, item.position());
            if heap.len() < k {
                heap.push(DistanceEntry {
                    item: item.clone(),
                    distance,
                });
            } else if let Some(farthest) = heap.peek() {
                if distance < farthest.distance {
                    heap.pop();
                    heap.push(DistanceEntry {
                        item: item.clone(),
                        distance,
                    });
                }
            }
        }

        heap.into_sorted_vec()
            .into_iter()
            .map(|e| NearestResult {
                item: e.item,
                distance: e.distance,
            })
            .collect()
    }

    pub fn within_distance(
        &self,
        point: &SphericalPoint,
        max_dist: f64,
    ) -> SpatialQueryResult<T> {
        let all = self.shell.all_items();
        let total_scanned = all.len();
        let items = all
            .into_iter()
            .filter(|item| angular_distance(point, item.position()) <= max_dist)
            .cloned()
            .collect();

        SpatialQueryResult {
            items,
            total_scanned,
        }
    }

    pub fn all_items(&self) -> Vec<&T> {
        self.shell.all_items()
    }
}

struct DistanceEntry<T: SpatialItem> {
    item: T,
    distance: f64,
}

impl<T: SpatialItem> PartialEq for DistanceEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T: SpatialItem> Eq for DistanceEntry<T> {}

impl<T: SpatialItem> PartialOrd for DistanceEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Max-heap by distance so we can evict the farthest when maintaining top-k closest
impl<T: SpatialItem> Ord for DistanceEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    #[derive(Debug, Clone)]
    struct TestItem {
        id: u64,
        pos: SphericalPoint,
    }

    impl SpatialItem for TestItem {
        type Id = u64;
        fn id(&self) -> &u64 {
            &self.id
        }
        fn position(&self) -> &SphericalPoint {
            &self.pos
        }
    }

    fn point(r: f64, theta: f64, phi: f64) -> SphericalPoint {
        SphericalPoint::new_unchecked(r, theta, phi)
    }

    fn item(id: u64, r: f64, theta: f64, phi: f64) -> TestItem {
        TestItem {
            id,
            pos: point(r, theta, phi),
        }
    }

    fn build_test_index() -> SpatialIndex<TestItem> {
        let mut idx = SpatialIndexBuilder::new()
            .shell_boundary(1.0)
            .shell_boundary(5.0)
            .theta_divisions(4)
            .phi_divisions(3)
            .build::<TestItem>();

        idx.insert(item(1, 0.5, 0.3, 0.5));
        idx.insert(item(2, 2.0, 1.0, FRAC_PI_2));
        idx.insert(item(3, 3.0, 3.0, FRAC_PI_4));
        idx.insert(item(4, 7.0, 5.0, PI - 0.2));
        idx.insert(item(5, 1.5, 0.5, 0.8));
        idx
    }

    #[test]
    fn insert_and_len() {
        let idx = build_test_index();
        assert_eq!(idx.len(), 5);
        assert!(!idx.is_empty());
    }

    #[test]
    fn get_returns_inserted_item() {
        let idx = build_test_index();
        let got = idx.get(&2).unwrap();
        assert_eq!(*got.id(), 2);
    }

    #[test]
    fn get_missing_returns_none() {
        let idx = build_test_index();
        assert!(idx.get(&99).is_none());
    }

    #[test]
    fn remove_returns_item_and_decrements_len() {
        let mut idx = build_test_index();
        let removed = idx.remove(&3).unwrap();
        assert_eq!(*removed.id(), 3);
        assert_eq!(idx.len(), 4);
        assert!(idx.get(&3).is_none());
    }

    #[test]
    fn remove_missing_returns_none() {
        let mut idx = build_test_index();
        assert!(idx.remove(&99).is_none());
    }

    #[test]
    fn query_shell_returns_correct_items() {
        let idx = build_test_index();
        let shell = Shell::new(1.0, 4.0).unwrap();
        let result = idx.query_shell(&shell);

        let ids: Vec<u64> = result.items.iter().map(|i| *i.id()).collect();
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert!(ids.contains(&5));
        assert!(!ids.contains(&1));
        assert!(!ids.contains(&4));
    }

    #[test]
    fn query_cone_returns_correct_items() {
        let idx = build_test_index();
        let cone = Cone::new(
            point(0.0, 0.0, 0.0),
            point(1.0, 0.5, 0.8),
            0.5,
        )
        .unwrap();

        let result = idx.query_cone(&cone);
        for item in &result.items {
            assert!(cone.contains(item.position()));
        }
    }

    #[test]
    fn query_band_returns_correct_items() {
        let idx = build_test_index();
        let band = Band::new(0.3, 1.0).unwrap();
        let result = idx.query_band(&band);

        for item in &result.items {
            let phi = item.position().phi;
            assert!(phi >= 0.3 && phi <= 1.0);
        }
    }

    #[test]
    fn query_region_compound() {
        let idx = build_test_index();
        let shell = Region::Shell(Shell::new(1.0, 4.0).unwrap());
        let band = Region::Band(Band::new(FRAC_PI_4, FRAC_PI_2 + 0.1).unwrap());
        let region = Region::intersection(vec![shell, band]);

        let result = idx.query_region(&region);
        for item in &result.items {
            assert!(region.contains(item.position()));
        }
        assert_eq!(result.total_scanned, 5);
    }

    #[test]
    fn nearest_returns_k_closest_sorted() {
        let mut idx = SpatialIndexBuilder::new()
            .theta_divisions(4)
            .phi_divisions(3)
            .build::<TestItem>();

        let center_theta = 1.0;
        let center_phi = FRAC_PI_2;

        idx.insert(item(1, 1.0, center_theta, center_phi));
        idx.insert(item(2, 1.0, center_theta + 0.1, center_phi));
        idx.insert(item(3, 1.0, center_theta + 0.5, center_phi));
        idx.insert(item(4, 1.0, center_theta + 1.0, center_phi));
        idx.insert(item(5, 1.0, center_theta + 2.0, center_phi));

        let query = point(1.0, center_theta, center_phi);
        let results = idx.nearest(&query, 3);

        assert_eq!(results.len(), 3);
        assert!(results[0].distance <= results[1].distance);
        assert!(results[1].distance <= results[2].distance);
        assert_eq!(*results[0].item.id(), 1);
        assert_eq!(*results[1].item.id(), 2);
        assert_eq!(*results[2].item.id(), 3);
    }

    #[test]
    fn nearest_k_larger_than_index() {
        let mut idx = SpatialIndexBuilder::new()
            .theta_divisions(4)
            .phi_divisions(3)
            .build::<TestItem>();

        idx.insert(item(1, 1.0, 0.5, FRAC_PI_2));
        idx.insert(item(2, 1.0, 1.0, FRAC_PI_2));

        let results = idx.nearest(&point(1.0, 0.5, FRAC_PI_2), 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn nearest_zero_k() {
        let idx = build_test_index();
        let results = idx.nearest(&point(1.0, 0.5, FRAC_PI_2), 0);
        assert!(results.is_empty());
    }

    #[test]
    fn within_distance_returns_all_in_range() {
        let mut idx = SpatialIndexBuilder::new()
            .theta_divisions(4)
            .phi_divisions(3)
            .build::<TestItem>();

        let center = point(1.0, 1.0, FRAC_PI_2);
        idx.insert(item(1, 1.0, 1.0, FRAC_PI_2));
        idx.insert(item(2, 1.0, 1.1, FRAC_PI_2));
        idx.insert(item(3, 1.0, 1.0 + PI, FRAC_PI_2));

        let result = idx.within_distance(&center, 0.2);
        let ids: Vec<u64> = result.items.iter().map(|i| *i.id()).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn update_replaces_item_position() {
        let mut idx = build_test_index();
        let old = idx.get(&2).unwrap();
        let old_phi = old.position().phi;

        let new_phi = old_phi + 0.5;
        idx.update(item(2, 2.0, 1.0, new_phi));

        let updated = idx.get(&2).unwrap();
        assert!((updated.position().phi - new_phi).abs() < 1e-12);
        assert_eq!(idx.len(), 5);
    }

    #[test]
    fn all_items_returns_everything() {
        let idx = build_test_index();
        assert_eq!(idx.all_items().len(), 5);
    }

    #[test]
    fn empty_index_queries() {
        let idx = SpatialIndexBuilder::new()
            .theta_divisions(4)
            .phi_divisions(3)
            .build::<TestItem>();

        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);

        let shell = Shell::new(0.5, 3.0).unwrap();
        assert!(idx.query_shell(&shell).items.is_empty());

        let cone = Cone::new(point(0.0, 0.0, 0.0), point(1.0, 0.0, FRAC_PI_2), FRAC_PI_4).unwrap();
        assert!(idx.query_cone(&cone).items.is_empty());

        let band = Band::new(0.1, FRAC_PI_2).unwrap();
        assert!(idx.query_band(&band).items.is_empty());

        let region = Region::Shell(Shell::new(0.5, 3.0).unwrap());
        assert!(idx.query_region(&region).items.is_empty());

        assert!(idx.nearest(&point(1.0, 0.0, FRAC_PI_2), 5).is_empty());
        assert!(idx.within_distance(&point(1.0, 0.0, FRAC_PI_2), 1.0).items.is_empty());
        assert!(idx.all_items().is_empty());
    }
}
