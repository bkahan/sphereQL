use crate::composite::{SpatialIndex, SpatialIndexBuilder};
use crate::item::{NearestResult, SpatialItem, SpatialQueryResult};
use sphereql_core::*;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CacheKey {
    Cone {
        center_bits: [u64; 3],
        half_angle_bits: u64,
    },
    Shell {
        inner_bits: u64,
        outer_bits: u64,
    },
    Band {
        phi_min_bits: u64,
        phi_max_bits: u64,
    },
    Region(u64),
    Nearest {
        point_bits: [u64; 3],
        k: usize,
    },
    WithinDistance {
        point_bits: [u64; 3],
        max_dist_bits: u64,
    },
}

impl CacheKey {
    fn from_cone(cone: &Cone) -> Self {
        CacheKey::Cone {
            center_bits: [
                cone.axis.r.to_bits(),
                cone.axis.theta.to_bits(),
                cone.axis.phi.to_bits(),
            ],
            half_angle_bits: cone.half_angle.to_bits(),
        }
    }

    fn from_shell(shell: &Shell) -> Self {
        CacheKey::Shell {
            inner_bits: shell.inner.to_bits(),
            outer_bits: shell.outer.to_bits(),
        }
    }

    fn from_band(band: &Band) -> Self {
        CacheKey::Band {
            phi_min_bits: band.phi_min.to_bits(),
            phi_max_bits: band.phi_max.to_bits(),
        }
    }

    fn from_region(region: &Region) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        Self::hash_region(region, &mut hasher);
        CacheKey::Region(hasher.finish())
    }

    fn from_nearest(point: &SphericalPoint, k: usize) -> Self {
        CacheKey::Nearest {
            point_bits: point_to_bits(point),
            k,
        }
    }

    fn from_within_distance(point: &SphericalPoint, max_dist: f64) -> Self {
        CacheKey::WithinDistance {
            point_bits: point_to_bits(point),
            max_dist_bits: max_dist.to_bits(),
        }
    }

    fn hash_region(region: &Region, hasher: &mut impl Hasher) {
        std::mem::discriminant(region).hash(hasher);
        match region {
            Region::Cone(c) => {
                c.apex.r.to_bits().hash(hasher);
                c.apex.theta.to_bits().hash(hasher);
                c.apex.phi.to_bits().hash(hasher);
                c.axis.r.to_bits().hash(hasher);
                c.axis.theta.to_bits().hash(hasher);
                c.axis.phi.to_bits().hash(hasher);
                c.half_angle.to_bits().hash(hasher);
            }
            Region::Cap(c) => {
                c.center.r.to_bits().hash(hasher);
                c.center.theta.to_bits().hash(hasher);
                c.center.phi.to_bits().hash(hasher);
                c.half_angle.to_bits().hash(hasher);
            }
            Region::Shell(s) => {
                s.inner.to_bits().hash(hasher);
                s.outer.to_bits().hash(hasher);
            }
            Region::Band(b) => {
                b.phi_min.to_bits().hash(hasher);
                b.phi_max.to_bits().hash(hasher);
            }
            Region::Wedge(w) => {
                w.theta_min.to_bits().hash(hasher);
                w.theta_max.to_bits().hash(hasher);
            }
            Region::Intersection(regions) | Region::Union(regions) => {
                regions.len().hash(hasher);
                for r in regions {
                    Self::hash_region(r, hasher);
                }
            }
        }
    }
}

fn point_to_bits(p: &SphericalPoint) -> [u64; 3] {
    [p.r.to_bits(), p.theta.to_bits(), p.phi.to_bits()]
}

struct CacheEntry<T: SpatialItem> {
    generation: u64,
    result: CachedResult<T>,
}

#[derive(Clone)]
enum CachedResult<T: SpatialItem> {
    Query(SpatialQueryResult<T>),
    Nearest(Vec<NearestResult<T>>),
}

pub struct CachedIndexBuilder {
    inner_builder: SpatialIndexBuilder,
    cache_capacity: usize,
}

impl CachedIndexBuilder {
    pub fn new() -> Self {
        Self {
            inner_builder: SpatialIndexBuilder::new(),
            cache_capacity: 128,
        }
    }

    pub fn shell_boundary(mut self, r: f64) -> Self {
        self.inner_builder = self.inner_builder.shell_boundary(r);
        self
    }

    pub fn uniform_shells(mut self, count: usize, max_r: f64) -> Self {
        self.inner_builder = self.inner_builder.uniform_shells(count, max_r);
        self
    }

    pub fn theta_divisions(mut self, n: usize) -> Self {
        self.inner_builder = self.inner_builder.theta_divisions(n);
        self
    }

    pub fn phi_divisions(mut self, n: usize) -> Self {
        self.inner_builder = self.inner_builder.phi_divisions(n);
        self
    }

    pub fn cache_capacity(mut self, cap: usize) -> Self {
        self.cache_capacity = cap;
        self
    }

    pub fn build<T: SpatialItem>(self) -> CachedIndex<T> {
        CachedIndex {
            inner: self.inner_builder.build(),
            cache: HashMap::new(),
            lru_order: VecDeque::new(),
            capacity: self.cache_capacity,
            generation: 0,
            cache_hits: 0,
            cache_lookups: 0,
        }
    }
}

impl Default for CachedIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// LRU-capped query-result cache wrapping a [`SpatialIndex`].
///
/// **LRU-touch cost.** `lru_order` is a `VecDeque<CacheKey>` with
/// `retain` / `position + remove` / `push_back` on each hit or miss —
/// all O(capacity). The default capacity is 128, so 128 ops per touch
/// is negligible for any workload this cache was designed for. If you
/// set `cache_capacity` above a few thousand, consider swapping this
/// module's Vec-based LRU for an `indexmap::IndexMap` keyed variant
/// (O(1) move-to-back via `shift_remove` + re-insert at tail).
pub struct CachedIndex<T: SpatialItem> {
    inner: SpatialIndex<T>,
    cache: HashMap<CacheKey, CacheEntry<T>>,
    lru_order: VecDeque<CacheKey>,
    capacity: usize,
    generation: u64,
    cache_hits: usize,
    cache_lookups: usize,
}

impl<T: SpatialItem> CachedIndex<T> {
    pub fn builder() -> CachedIndexBuilder {
        CachedIndexBuilder::new()
    }

    pub fn insert(&mut self, item: T) {
        self.inner.insert(item);
        self.invalidate();
    }

    pub fn remove(&mut self, id: &T::Id) -> Option<T> {
        let result = self.inner.remove(id);
        if result.is_some() {
            self.invalidate();
        }
        result
    }

    pub fn update(&mut self, item: T) {
        self.inner.update(item);
        self.invalidate();
    }

    pub fn get(&self, id: &T::Id) -> Option<&T> {
        self.inner.get(id)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn cache_hit_rate(&self) -> (usize, usize) {
        (self.cache_hits, self.cache_lookups)
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
    }

    pub fn query_cone(&mut self, cone: &Cone) -> SpatialQueryResult<T> {
        let key = CacheKey::from_cone(cone);
        if let Some(CachedResult::Query(result)) = self.cache_lookup(&key) {
            return result;
        }
        let result = self.inner.query_cone(cone);
        self.cache_insert(key, CachedResult::Query(result.clone()));
        result
    }

    pub fn query_shell(&mut self, shell: &Shell) -> SpatialQueryResult<T> {
        let key = CacheKey::from_shell(shell);
        if let Some(CachedResult::Query(result)) = self.cache_lookup(&key) {
            return result;
        }
        let result = self.inner.query_shell(shell);
        self.cache_insert(key, CachedResult::Query(result.clone()));
        result
    }

    pub fn query_band(&mut self, band: &Band) -> SpatialQueryResult<T> {
        let key = CacheKey::from_band(band);
        if let Some(CachedResult::Query(result)) = self.cache_lookup(&key) {
            return result;
        }
        let result = self.inner.query_band(band);
        self.cache_insert(key, CachedResult::Query(result.clone()));
        result
    }

    pub fn query_region(&mut self, region: &Region) -> SpatialQueryResult<T> {
        let key = CacheKey::from_region(region);
        if let Some(CachedResult::Query(result)) = self.cache_lookup(&key) {
            return result;
        }
        let result = self.inner.query_region(region);
        self.cache_insert(key, CachedResult::Query(result.clone()));
        result
    }

    pub fn nearest(&mut self, point: &SphericalPoint, k: usize) -> Vec<NearestResult<T>> {
        let key = CacheKey::from_nearest(point, k);
        if let Some(CachedResult::Nearest(results)) = self.cache_lookup(&key) {
            return results;
        }
        let result = self.inner.nearest(point, k);
        self.cache_insert(key, CachedResult::Nearest(result.clone()));
        result
    }

    pub fn within_distance(
        &mut self,
        point: &SphericalPoint,
        max_dist: f64,
    ) -> SpatialQueryResult<T> {
        let key = CacheKey::from_within_distance(point, max_dist);
        if let Some(CachedResult::Query(result)) = self.cache_lookup(&key) {
            return result;
        }
        let result = self.inner.within_distance(point, max_dist);
        self.cache_insert(key, CachedResult::Query(result.clone()));
        result
    }

    /// Returns a cloned result if the cache has a valid (non-stale) entry.
    /// Handles LRU touch on hit and lazy eviction of stale entries.
    fn cache_lookup(&mut self, key: &CacheKey) -> Option<CachedResult<T>> {
        self.cache_lookups += 1;

        if let Some(entry) = self.cache.get(key) {
            if entry.generation == self.generation {
                self.cache_hits += 1;
                let result = entry.result.clone();
                self.touch_lru(key);
                return Some(result);
            }
            let key_owned = key.clone();
            self.cache.remove(&key_owned);
            self.lru_order.retain(|k| k != &key_owned);
        }
        None
    }

    fn cache_insert(&mut self, key: CacheKey, result: CachedResult<T>) {
        if self.capacity == 0 {
            return;
        }

        if self.cache.contains_key(&key) {
            self.touch_lru(&key);
        } else {
            while self.cache.len() >= self.capacity {
                self.evict_lru();
            }
            self.lru_order.push_back(key.clone());
        }

        self.cache.insert(
            key,
            CacheEntry {
                generation: self.generation,
                result,
            },
        );
    }

    fn touch_lru(&mut self, key: &CacheKey) {
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
            self.lru_order.push_back(key.clone());
        }
    }

    fn evict_lru(&mut self) {
        if let Some(key) = self.lru_order.pop_front() {
            self.cache.remove(&key);
        }
    }

    fn invalidate(&mut self) {
        self.generation += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

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

    fn make_item(id: u64, r: f64, theta: f64, phi: f64) -> TestItem {
        TestItem {
            id,
            pos: SphericalPoint::new_unchecked(r, theta, phi),
        }
    }

    fn build_populated_index() -> CachedIndex<TestItem> {
        let mut index = CachedIndexBuilder::new()
            .shell_boundary(2.0)
            .shell_boundary(5.0)
            .theta_divisions(4)
            .phi_divisions(4)
            .cache_capacity(16)
            .build::<TestItem>();

        index.insert(make_item(1, 1.0, 0.5, 0.5));
        index.insert(make_item(2, 3.0, 1.0, 1.0));
        index.insert(make_item(3, 4.0, 2.0, FRAC_PI_2));
        index.insert(make_item(4, 6.0, 3.0, 2.0));

        index.cache_hits = 0;
        index.cache_lookups = 0;
        index
    }

    #[test]
    fn cache_hit_on_repeated_query() {
        let mut index = build_populated_index();
        let shell = Shell::new(0.5, 2.5).unwrap();

        let first = index.query_shell(&shell);
        let second = index.query_shell(&shell);

        assert_eq!(first.items.len(), second.items.len());
        let (hits, lookups) = index.cache_hit_rate();
        assert_eq!(lookups, 2);
        assert_eq!(hits, 1);
    }

    #[test]
    fn cache_invalidated_on_insert() {
        let mut index = build_populated_index();
        let shell = Shell::new(0.5, 2.5).unwrap();

        let first = index.query_shell(&shell);
        assert_eq!(first.items.len(), 1);

        index.insert(make_item(10, 1.5, 0.5, 0.5));
        let second = index.query_shell(&shell);
        assert_eq!(second.items.len(), 2);

        let (hits, lookups) = index.cache_hit_rate();
        assert_eq!(lookups, 2);
        assert_eq!(hits, 0);
    }

    #[test]
    fn cache_invalidated_on_remove() {
        let mut index = build_populated_index();
        let shell = Shell::new(0.5, 2.5).unwrap();

        let first = index.query_shell(&shell);
        assert_eq!(first.items.len(), 1);

        index.remove(&1);
        let second = index.query_shell(&shell);
        assert_eq!(second.items.len(), 0);

        let (hits, _) = index.cache_hit_rate();
        assert_eq!(hits, 0);
    }

    #[test]
    fn lru_eviction() {
        let mut index = CachedIndexBuilder::new()
            .shell_boundary(2.0)
            .shell_boundary(5.0)
            .cache_capacity(2)
            .build::<TestItem>();

        index.insert(make_item(1, 1.0, 0.5, 0.5));
        index.insert(make_item(2, 3.0, 1.0, 1.0));
        index.insert(make_item(3, 6.0, 2.0, 2.0));
        index.cache_hits = 0;
        index.cache_lookups = 0;

        let shell_a = Shell::new(0.5, 1.5).unwrap();
        let shell_b = Shell::new(2.5, 4.0).unwrap();
        let shell_c = Shell::new(5.0, 7.0).unwrap();

        index.query_shell(&shell_a);
        index.query_shell(&shell_b);
        index.query_shell(&shell_c);

        index.query_shell(&shell_b);
        index.query_shell(&shell_c);
        index.query_shell(&shell_a);

        let (hits, lookups) = index.cache_hit_rate();
        assert_eq!(lookups, 6);
        assert_eq!(hits, 2);
    }

    #[test]
    fn generation_counter_increments_on_mutations() {
        let mut index = CachedIndexBuilder::new()
            .cache_capacity(16)
            .build::<TestItem>();

        assert_eq!(index.generation, 0);

        index.insert(make_item(1, 1.0, 0.5, 0.5));
        assert_eq!(index.generation, 1);

        index.update(make_item(1, 2.0, 0.5, 0.5));
        assert_eq!(index.generation, 2);

        index.remove(&1);
        assert_eq!(index.generation, 3);

        index.remove(&999);
        assert_eq!(index.generation, 3);
    }

    #[test]
    fn stale_entries_not_returned() {
        let mut index = build_populated_index();
        let shell = Shell::new(0.5, 2.5).unwrap();

        index.query_shell(&shell);
        let gen_before = index.generation;

        index.insert(make_item(20, 1.8, 0.5, 0.5));
        assert!(index.generation > gen_before);

        let (hits_before, _) = index.cache_hit_rate();
        index.query_shell(&shell);
        let (hits_after, _) = index.cache_hit_rate();
        assert_eq!(hits_after, hits_before);
    }

    #[test]
    fn query_band_works() {
        let mut index = build_populated_index();
        let band = Band::new(0.3, 0.7).unwrap();

        let result = index.query_band(&band);
        let ids: Vec<u64> = result.items.iter().map(|i| *i.id()).collect();
        assert!(ids.contains(&1));
    }

    #[test]
    fn query_cone_works() {
        let mut index = build_populated_index();
        let cone = Cone::new(
            SphericalPoint::new_unchecked(0.0, 0.0, 0.0),
            SphericalPoint::new_unchecked(1.0, 0.5, 0.5),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();

        let result = index.query_cone(&cone);
        assert!(!result.items.is_empty() || result.total_scanned > 0);
    }

    #[test]
    fn query_region_works() {
        let mut index = build_populated_index();
        let region = Region::Shell(Shell::new(0.5, 2.5).unwrap());

        let result = index.query_region(&region);
        let ids: Vec<u64> = result.items.iter().map(|i| *i.id()).collect();
        assert!(ids.contains(&1));
        assert!(!ids.contains(&4));
    }

    #[test]
    fn nearest_works() {
        let mut index = build_populated_index();
        let point = SphericalPoint::new_unchecked(1.0, 0.5, 0.5);

        let results = index.nearest(&point, 2);
        assert!(results.len() <= 2);
        if results.len() == 2 {
            assert!(results[0].distance <= results[1].distance);
        }
    }

    #[test]
    fn nearest_cached() {
        let mut index = build_populated_index();
        let point = SphericalPoint::new_unchecked(1.0, 0.5, 0.5);

        let first = index.nearest(&point, 2);
        let second = index.nearest(&point, 2);

        assert_eq!(first.len(), second.len());
        let (hits, lookups) = index.cache_hit_rate();
        assert_eq!(hits, 1);
        assert_eq!(lookups, 2);
    }

    #[test]
    fn within_distance_works() {
        let mut index = build_populated_index();
        let point = SphericalPoint::new_unchecked(1.0, 0.5, 0.5);

        let result = index.within_distance(&point, 1.0);
        for item in &result.items {
            let dist = sphereql_core::angular_distance(item.position(), &point);
            assert!(dist <= 1.0 + f64::EPSILON);
        }
    }

    #[test]
    fn clear_cache_resets() {
        let mut index = build_populated_index();
        let shell = Shell::new(0.5, 2.5).unwrap();

        index.query_shell(&shell);
        index.query_shell(&shell);
        assert_eq!(index.cache_hit_rate(), (1, 2));

        index.clear_cache();
        assert!(index.cache.is_empty());
        assert!(index.lru_order.is_empty());
    }

    #[test]
    fn get_delegates_without_caching() {
        let index = build_populated_index();

        let item = index.get(&1);
        assert!(item.is_some());
        assert_eq!(*item.unwrap().id(), 1);

        assert!(index.get(&999).is_none());

        let (_, lookups) = index.cache_hit_rate();
        assert_eq!(lookups, 0);
    }

    #[test]
    fn builder_defaults() {
        let builder = CachedIndexBuilder::new();
        assert_eq!(builder.cache_capacity, 128);
    }

    #[test]
    fn builder_with_all_options() {
        let index = CachedIndexBuilder::new()
            .shell_boundary(1.0)
            .shell_boundary(3.0)
            .uniform_shells(4, 10.0)
            .theta_divisions(8)
            .phi_divisions(6)
            .cache_capacity(64)
            .build::<TestItem>();

        assert_eq!(index.capacity, 64);
        assert!(index.is_empty());
    }

    #[test]
    fn len_and_is_empty() {
        let mut index = CachedIndexBuilder::new()
            .cache_capacity(8)
            .build::<TestItem>();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        index.insert(make_item(1, 1.0, 0.5, 0.5));
        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn zero_capacity_cache_never_stores() {
        let mut index = CachedIndexBuilder::new()
            .cache_capacity(0)
            .build::<TestItem>();

        index.insert(make_item(1, 1.0, 0.5, 0.5));
        index.cache_hits = 0;
        index.cache_lookups = 0;

        let shell = Shell::new(0.5, 2.5).unwrap();
        index.query_shell(&shell);
        index.query_shell(&shell);

        let (hits, lookups) = index.cache_hit_rate();
        assert_eq!(hits, 0);
        assert_eq!(lookups, 2);
    }
}
