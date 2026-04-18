use std::collections::HashSet;

use sphereql_core::SphericalPoint;

use crate::traits::{DimensionMapper, LayoutStrategy};
use crate::types::LayoutEntry;

pub struct ManagedLayout<T> {
    items: Vec<T>,
    positions: Vec<SphericalPoint>,
    dirty: HashSet<usize>,
    needs_full_reflow: bool,
}

impl<T: Clone + Send + Sync> ManagedLayout<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            positions: Vec::new(),
            dirty: HashSet::new(),
            needs_full_reflow: false,
        }
    }

    pub fn add(&mut self, item: T) {
        let idx = self.items.len();
        self.items.push(item);
        self.positions.push(SphericalPoint::origin());
        self.dirty.insert(idx);
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.items.len() {
            return None;
        }

        let item = self.items.remove(index);
        let _ = self.positions.remove(index);
        self.dirty.remove(&index);

        if index != self.items.len() {
            self.needs_full_reflow = true;
            let shifted: HashSet<usize> = self
                .dirty
                .iter()
                .map(|&i| if i > index { i - 1 } else { i })
                .collect();
            self.dirty = shifted;
        }
        self.dirty.retain(|&i| i < self.items.len());

        Some(item)
    }

    pub fn mark_dirty(&mut self, index: usize) {
        if index < self.items.len() {
            self.dirty.insert(index);
        }
    }

    pub fn reflow(
        &mut self,
        strategy: &dyn LayoutStrategy<T>,
        mapper: &dyn DimensionMapper<Item = T>,
    ) {
        let result = strategy.layout(&self.items, mapper);
        for (i, entry) in result.entries.into_iter().enumerate() {
            if i < self.positions.len() {
                self.positions[i] = entry.position;
            }
        }
        self.dirty.clear();
        self.needs_full_reflow = false;
    }

    pub fn reflow_incremental(
        &mut self,
        strategy: &dyn LayoutStrategy<T>,
        mapper: &dyn DimensionMapper<Item = T>,
    ) {
        if self.needs_full_reflow {
            self.reflow(strategy, mapper);
            return;
        }

        if self.dirty.is_empty() {
            return;
        }

        let dirty_indices: Vec<usize> = self.dirty.iter().copied().collect();
        let dirty_items: Vec<T> = dirty_indices
            .iter()
            .map(|&i| self.items[i].clone())
            .collect();

        let result = strategy.layout(&dirty_items, mapper);
        for (i, entry) in dirty_indices.iter().zip(result.entries) {
            self.positions[*i] = entry.position;
        }

        self.dirty.clear();
    }

    pub fn items(&self) -> &[T] {
        &self.items
    }

    pub fn positions(&self) -> &[SphericalPoint] {
        &self.positions
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn get_entry(&self, index: usize) -> Option<LayoutEntry<&T>> {
        if index >= self.items.len() {
            return None;
        }
        Some(LayoutEntry {
            item: &self.items[index],
            position: self.positions[index],
        })
    }
}

impl<T: Clone + Send + Sync> Default for ManagedLayout<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::DimensionMapper;
    use crate::types::{LayoutQuality, LayoutResult};
    use std::f64::consts::FRAC_PI_2;

    struct IdentityMapper;

    impl DimensionMapper for IdentityMapper {
        type Item = u32;
        fn map(&self, _item: &u32) -> SphericalPoint {
            SphericalPoint::origin()
        }
    }

    struct DeterministicStrategy;

    impl LayoutStrategy<u32> for DeterministicStrategy {
        fn layout(
            &self,
            items: &[u32],
            _mapper: &dyn DimensionMapper<Item = u32>,
        ) -> LayoutResult<u32> {
            let entries = items
                .iter()
                .map(|&item| LayoutEntry {
                    item,
                    position: SphericalPoint::new_unchecked(1.0, item as f64 * 0.1, FRAC_PI_2),
                })
                .collect();
            LayoutResult {
                entries,
                quality: LayoutQuality::default(),
            }
        }
    }

    #[test]
    fn new_layout_is_empty() {
        let layout: ManagedLayout<u32> = ManagedLayout::new();
        assert!(layout.is_empty());
        assert_eq!(layout.len(), 0);
        assert!(layout.items().is_empty());
        assert!(layout.positions().is_empty());
    }

    #[test]
    fn add_increases_len() {
        let mut layout = ManagedLayout::new();
        layout.add(1u32);
        assert_eq!(layout.len(), 1);
        layout.add(2);
        layout.add(3);
        assert_eq!(layout.len(), 3);
        assert!(!layout.is_empty());
    }

    #[test]
    fn remove_returns_item_and_decrements() {
        let mut layout = ManagedLayout::new();
        layout.add(10u32);
        layout.add(20);
        layout.add(30);
        assert_eq!(layout.len(), 3);

        let removed = layout.remove(1);
        assert_eq!(removed, Some(20));
        assert_eq!(layout.len(), 2);
        assert_eq!(layout.items(), &[10, 30]);

        let removed = layout.remove(1);
        assert_eq!(removed, Some(30));
        assert_eq!(layout.len(), 1);

        assert_eq!(layout.remove(5), None);
    }

    #[test]
    fn incremental_reflow_only_updates_dirty() {
        let mut layout = ManagedLayout::new();
        layout.add(1u32);
        layout.add(2);
        layout.add(3);

        layout.reflow(&DeterministicStrategy, &IdentityMapper);
        let pos_0_after_full = layout.positions()[0];
        let pos_2_after_full = layout.positions()[2];

        layout.mark_dirty(1);

        layout.reflow_incremental(&DeterministicStrategy, &IdentityMapper);

        assert_eq!(layout.positions()[0].theta, pos_0_after_full.theta);
        assert_eq!(layout.positions()[2].theta, pos_2_after_full.theta);
        assert!(layout.positions()[1].theta.is_finite());
    }

    #[test]
    fn full_reflow_updates_all_positions() {
        let mut layout = ManagedLayout::new();
        layout.add(5u32);
        layout.add(10);

        assert_eq!(layout.positions()[0].theta, 0.0);
        assert_eq!(layout.positions()[1].theta, 0.0);

        layout.reflow(&DeterministicStrategy, &IdentityMapper);

        assert!((layout.positions()[0].theta - 0.5).abs() < 1e-12);
        assert!((layout.positions()[1].theta - 1.0).abs() < 1e-12);
    }

    #[test]
    fn incremental_falls_back_to_full_when_needed() {
        let mut layout = ManagedLayout::new();
        layout.add(1u32);
        layout.add(2);
        layout.add(3);

        layout.reflow(&DeterministicStrategy, &IdentityMapper);

        layout.remove(0);
        assert_eq!(layout.items(), &[2, 3]);

        layout.reflow_incremental(&DeterministicStrategy, &IdentityMapper);

        assert!((layout.positions()[0].theta - 0.2).abs() < 1e-12);
        assert!((layout.positions()[1].theta - 0.3).abs() < 1e-12);
    }

    #[test]
    fn get_entry_returns_correct_data() {
        let mut layout = ManagedLayout::new();
        layout.add(42u32);
        layout.reflow(&DeterministicStrategy, &IdentityMapper);

        let entry = layout.get_entry(0).unwrap();
        assert_eq!(*entry.item, 42);
        assert!((entry.position.theta - 4.2).abs() < 1e-12);

        assert!(layout.get_entry(1).is_none());
    }
}
