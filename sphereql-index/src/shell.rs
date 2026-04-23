use crate::item::{SpatialItem, SpatialQueryResult};
use sphereql_core::{Contains, Shell};
use std::collections::HashMap;

/// Builder for constructing a [`ShellIndex`] with custom radial boundaries.
pub struct ShellIndexBuilder {
    boundaries: Vec<f64>,
}

impl ShellIndexBuilder {
    pub fn new() -> Self {
        Self {
            boundaries: Vec::new(),
        }
    }

    pub fn boundary(mut self, r: f64) -> Self {
        self.boundaries.push(r);
        self
    }

    pub fn uniform_boundaries(mut self, count: usize, max_r: f64) -> Self {
        for i in 0..=count {
            self.boundaries.push(max_r * i as f64 / count as f64);
        }
        self
    }

    pub fn build<T: SpatialItem>(mut self) -> ShellIndex<T> {
        self.boundaries.sort_by(|a, b| a.total_cmp(b));
        self.boundaries.dedup();

        let bucket_count = if self.boundaries.is_empty() {
            1
        } else {
            self.boundaries.len() + 1
        };

        ShellIndex {
            boundaries: self.boundaries,
            buckets: (0..bucket_count).map(|_| Vec::new()).collect(),
            item_map: HashMap::new(),
        }
    }
}

impl Default for ShellIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Radial shell index that partitions items into concentric spherical shells.
///
/// Items are placed into buckets bounded by configurable radial boundaries.
/// Shell queries only scan the overlapping buckets, making radial filtering
/// sub-linear for indices with many shells.
pub struct ShellIndex<T: SpatialItem> {
    boundaries: Vec<f64>,
    buckets: Vec<Vec<T>>,
    item_map: HashMap<T::Id, usize>,
}

impl<T: SpatialItem> ShellIndex<T> {
    pub fn builder() -> ShellIndexBuilder {
        ShellIndexBuilder::new()
    }

    pub fn insert(&mut self, item: T) {
        let bucket = self.bucket_index(item.position().r);
        self.item_map.insert(item.id().clone(), bucket);
        self.buckets[bucket].push(item);
    }

    pub fn remove(&mut self, id: &T::Id) -> Option<T> {
        let bucket_idx = self.item_map.remove(id)?;
        let bucket = &mut self.buckets[bucket_idx];
        let pos = bucket.iter().position(|item| item.id() == id)?;
        Some(bucket.swap_remove(pos))
    }

    pub fn get(&self, id: &T::Id) -> Option<&T> {
        let &bucket_idx = self.item_map.get(id)?;
        self.buckets[bucket_idx].iter().find(|item| item.id() == id)
    }

    pub fn len(&self) -> usize {
        self.item_map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.item_map.is_empty()
    }

    pub fn query_shell(&self, shell: &Shell) -> SpatialQueryResult<T> {
        let start = self.bucket_index(shell.inner);
        let end = self.bucket_index(shell.outer);

        let mut items = Vec::new();
        let mut total_scanned = 0;

        for bucket in &self.buckets[start..=end] {
            total_scanned += bucket.len();
            for item in bucket {
                if shell.contains(item.position()) {
                    items.push(item.clone());
                }
            }
        }

        SpatialQueryResult {
            items,
            total_scanned,
        }
    }

    pub fn all_items(&self) -> Vec<&T> {
        self.buckets.iter().flat_map(|b| b.iter()).collect()
    }

    fn bucket_index(&self, r: f64) -> usize {
        match self.boundaries.binary_search_by(|b| b.total_cmp(&r)) {
            // Exactly on a boundary: place in the bucket after it
            Ok(i) => (i + 1).min(self.buckets.len() - 1),
            Err(i) => i.min(self.buckets.len() - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sphereql_core::SphericalPoint;

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

    fn make_item(id: u64, r: f64) -> TestItem {
        TestItem {
            id,
            pos: SphericalPoint {
                r,
                theta: 1.0,
                phi: 0.5,
            },
        }
    }

    #[test]
    fn insert_places_items_in_correct_buckets() {
        let mut index: ShellIndex<TestItem> = ShellIndexBuilder::new()
            .boundary(1.0)
            .boundary(2.0)
            .boundary(5.0)
            .build();

        index.insert(make_item(1, 0.5));
        index.insert(make_item(2, 1.5));
        index.insert(make_item(3, 3.0));
        index.insert(make_item(4, 10.0));

        assert_eq!(index.len(), 4);
        assert!(index.get(&1).is_some());
        assert!(index.get(&2).is_some());
        assert!(index.get(&3).is_some());
        assert!(index.get(&4).is_some());
    }

    #[test]
    fn query_shell_returns_matching_items() {
        let mut index: ShellIndex<TestItem> = ShellIndexBuilder::new()
            .boundary(1.0)
            .boundary(3.0)
            .boundary(5.0)
            .build();

        index.insert(make_item(1, 0.5));
        index.insert(make_item(2, 2.0));
        index.insert(make_item(3, 4.0));
        index.insert(make_item(4, 6.0));

        let shell = Shell::new(1.5, 4.5).unwrap();
        let result = index.query_shell(&shell);

        let ids: Vec<u64> = result.items.iter().map(|i| *i.id()).collect();
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&1));
        assert!(!ids.contains(&4));
    }

    #[test]
    fn query_shell_does_not_scan_irrelevant_buckets() {
        let mut index: ShellIndex<TestItem> = ShellIndexBuilder::new()
            .boundary(1.0)
            .boundary(3.0)
            .boundary(5.0)
            .build();

        index.insert(make_item(1, 0.5));
        index.insert(make_item(2, 2.0));
        index.insert(make_item(3, 4.0));
        index.insert(make_item(4, 6.0));

        let shell = Shell::new(1.5, 2.5).unwrap();
        let result = index.query_shell(&shell);

        assert!(result.total_scanned < index.len());
    }

    #[test]
    fn remove_item() {
        let mut index: ShellIndex<TestItem> = ShellIndexBuilder::new().boundary(1.0).build();

        index.insert(make_item(1, 0.5));
        index.insert(make_item(2, 1.5));

        let removed = index.remove(&1);
        assert!(removed.is_some());
        assert_eq!(*removed.unwrap().id(), 1);
        assert!(index.get(&1).is_none());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut index: ShellIndex<TestItem> = ShellIndexBuilder::new().boundary(1.0).build();

        assert!(index.remove(&42).is_none());
    }

    #[test]
    fn uniform_boundaries_creates_correct_count() {
        let builder = ShellIndexBuilder::new().uniform_boundaries(4, 8.0);
        assert_eq!(builder.boundaries, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn item_exactly_on_boundary() {
        let mut index: ShellIndex<TestItem> = ShellIndexBuilder::new().boundary(2.0).build();

        index.insert(make_item(1, 2.0));
        assert!(index.get(&1).is_some());

        let shell = Shell::new(1.5, 2.5).unwrap();
        let result = index.query_shell(&shell);
        assert_eq!(result.items.len(), 1);
        assert_eq!(*result.items[0].id(), 1);
    }

    #[test]
    fn empty_index_queries() {
        let index: ShellIndex<TestItem> =
            ShellIndexBuilder::new().boundary(1.0).boundary(5.0).build();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        let shell = Shell::new(0.5, 3.0).unwrap();
        let result = index.query_shell(&shell);
        assert!(result.items.is_empty());
        assert_eq!(result.total_scanned, 0);
    }

    #[test]
    fn all_items_returns_everything() {
        let mut index: ShellIndex<TestItem> = ShellIndexBuilder::new().boundary(1.0).build();

        index.insert(make_item(1, 0.5));
        index.insert(make_item(2, 1.5));
        index.insert(make_item(3, 3.0));

        let all = index.all_items();
        assert_eq!(all.len(), 3);
    }
}
