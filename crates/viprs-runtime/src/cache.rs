//! Pipeline-owned caches for reusable operation output tiles.
//!
//! These caches sit at adapter level because they are part of execution policy:
//! they trade memory for less recomputation when compiled pipeline nodes are
//! revisited during one run.

#![allow(clippy::significant_drop_tightening)]
// REASON: cache guards intentionally stay alive across validation and commit steps.

use std::{
    collections::HashMap,
    num::NonZeroUsize,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{
    adapters::lock_instrumentation,
    domain::{error::ViprsError, image::Region},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    op_id: usize,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

impl CacheKey {
    const fn new(op_id: usize, region: Region) -> Self {
        Self {
            op_id,
            x: region.x,
            y: region.y,
            width: region.width,
            height: region.height,
        }
    }
}

struct CacheEntry {
    tile: Arc<[u8]>,
    last_used: AtomicU64,
}

impl CacheEntry {
    const fn new(tile: Arc<[u8]>, last_used: u64) -> Self {
        Self {
            tile,
            last_used: AtomicU64::new(last_used),
        }
    }

    fn touch(&self, stamp: u64) {
        self.last_used.store(stamp, Ordering::Relaxed);
    }

    fn last_used(&self) -> u64 {
        self.last_used.load(Ordering::Relaxed)
    }
}

struct CacheState {
    tiles: HashMap<CacheKey, Arc<CacheEntry>>,
    total_bytes: usize,
}

impl CacheState {
    fn new() -> Self {
        Self {
            tiles: HashMap::new(),
            total_bytes: 0,
        }
    }
}

struct SharedCache {
    state: RwLock<CacheState>,
    access_clock: AtomicU64,
}

impl SharedCache {
    fn new() -> Self {
        Self {
            state: RwLock::new(CacheState::new()),
            access_clock: AtomicU64::new(0),
        }
    }

    fn next_stamp(&self) -> u64 {
        self.access_clock.fetch_add(1, Ordering::Relaxed) + 1
    }
}

/// Pipeline-owned tile cache keyed by compiled operation id and exact region.
///
/// # Concurrency model
///
/// The cache uses `Arc<RwLock<CacheState>>` because reads dominate once a hot
/// region has been populated. Lookups take a shared read lock long enough to
/// clone the immutable `Arc<[u8]>` tile payload; LRU recency lives in
/// `CacheEntry::last_used`, so cache hits touch only an atomic, not the map
/// lock. Cache misses compute outside the lock and re-enter with a write lock
/// for insertion/eviction, so rayon workers never hold the lock while
/// executing pixel code.
#[derive(Clone)]
/// The `OperationTileCache` type provides concrete adapter functionality in the `adapters` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::cache::OperationTileCache>();
/// ```
pub struct OperationTileCache {
    max_bytes: NonZeroUsize,
    shared: Arc<SharedCache>,
}

impl OperationTileCache {
    /// Create a cache with an upper bound on retained tile bytes.
    ///
    /// Use this when a compiled pipeline is expected to revisit the same node
    /// outputs and recomputation would be more expensive than holding tiles in
    /// memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use viprs::adapters::cache::OperationTileCache;
    ///
    /// let cache = OperationTileCache::new(NonZeroUsize::new(1024).unwrap());
    /// assert_eq!(cache.max_bytes().get(), 1024);
    /// ```
    #[must_use]
    pub fn new(max_bytes: NonZeroUsize) -> Self {
        Self {
            max_bytes,
            shared: Arc::new(SharedCache::new()),
        }
    }

    /// Return the configured byte budget for this cache.
    ///
    /// This helps schedulers and builders report or reuse the cache budget
    /// without peeking into internal state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use viprs::adapters::cache::OperationTileCache;
    ///
    /// let cache = OperationTileCache::new(NonZeroUsize::new(256).unwrap());
    /// assert_eq!(cache.max_bytes().get(), 256);
    /// ```
    #[must_use]
    pub const fn max_bytes(&self) -> NonZeroUsize {
        self.max_bytes
    }

    /// Look up a cached tile for a compiled operation and output region.
    ///
    /// This solves repeated tile reads in fan-out or revisit-heavy pipelines by
    /// sharing immutable tile buffers across worker threads.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use viprs::adapters::cache::OperationTileCache;
    /// use viprs::domain::image::Region;
    ///
    /// let cache = OperationTileCache::new(NonZeroUsize::new(64).unwrap());
    /// let cached = cache.get(3, Region::new(0, 0, 1, 1))?;
    /// assert!(cached.is_none());
    /// # Ok::<(), viprs::domain::error::ViprsError>(())
    /// ```
    pub fn get(&self, op_id: usize, region: Region) -> Result<Option<Arc<[u8]>>, ViprsError> {
        let key = CacheKey::new(op_id, region);
        let cached = {
            lock_instrumentation::record_lock_acquisition();
            let state = self.shared.state.read().map_err(|_| {
                ViprsError::Scheduler("operation tile cache read lock poisoned".into())
            })?;
            state.tiles.get(&key).cloned()
        };

        if let Some(entry) = cached {
            entry.touch(self.shared.next_stamp());
            return Ok(Some(Arc::clone(&entry.tile)));
        }

        Ok(None)
    }

    /// Insert a tile produced by a compiled operation, evicting older entries if needed.
    ///
    /// This keeps tile reuse bounded by memory while preserving a simple LRU
    /// policy for operation outputs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::{num::NonZeroUsize, sync::Arc};
    /// use viprs::adapters::cache::OperationTileCache;
    /// use viprs::domain::image::Region;
    ///
    /// let cache = OperationTileCache::new(NonZeroUsize::new(64).unwrap());
    /// cache.insert(1, Region::new(0, 0, 1, 1), Arc::from([255_u8]))?;
    /// assert!(cache.get(1, Region::new(0, 0, 1, 1))?.is_some());
    /// # Ok::<(), viprs::domain::error::ViprsError>(())
    /// ```
    pub fn insert(&self, op_id: usize, region: Region, tile: Arc<[u8]>) -> Result<(), ViprsError> {
        let key = CacheKey::new(op_id, region);
        let stamp = self.shared.next_stamp();
        lock_instrumentation::record_lock_acquisition();
        let mut state = self.shared.state.write().map_err(|_| {
            ViprsError::Scheduler("operation tile cache write lock poisoned".into())
        })?;

        if let Some(existing) = state.tiles.get(&key) {
            existing.touch(stamp);
            return Ok(());
        }

        let tile_bytes = tile.len();
        state.total_bytes += tile_bytes;
        state
            .tiles
            .insert(key, Arc::new(CacheEntry::new(tile, stamp)));
        while state.total_bytes > self.max_bytes.get() {
            let evicted = state
                .tiles
                .iter()
                .min_by_key(|(_, entry)| entry.last_used())
                .map(|(key, _)| *key);
            if let Some(evicted) = evicted {
                if let Some(entry) = state.tiles.remove(&evicted) {
                    state.total_bytes = state.total_bytes.saturating_sub(entry.tile.len());
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Drop every cached tile.
    ///
    /// This is useful after a profiled run or before releasing a pipeline when
    /// the caller wants to reclaim memory immediately.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::{num::NonZeroUsize, sync::Arc};
    /// use viprs::adapters::cache::OperationTileCache;
    /// use viprs::domain::image::Region;
    ///
    /// let cache = OperationTileCache::new(NonZeroUsize::new(64).unwrap());
    /// cache.insert(1, Region::new(0, 0, 1, 1), Arc::from([1_u8]))?;
    /// cache.clear()?;
    /// assert!(cache.get(1, Region::new(0, 0, 1, 1))?.is_none());
    /// # Ok::<(), viprs::domain::error::ViprsError>(())
    /// ```
    pub fn clear(&self) -> Result<(), ViprsError> {
        lock_instrumentation::record_lock_acquisition();
        let mut state = self.shared.state.write().map_err(|_| {
            ViprsError::Scheduler("operation tile cache write lock poisoned".into())
        })?;
        state.tiles.clear();
        state.total_bytes = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_returns_inserted_tile() {
        let cache = OperationTileCache::new(NonZeroUsize::new(8).unwrap());
        let region = Region::new(0, 0, 2, 2);
        let tile: Arc<[u8]> = Arc::from([1, 2, 3, 4]);

        cache.insert(7, region, Arc::clone(&tile)).unwrap();
        let cached = cache.get(7, region).unwrap().unwrap();

        assert_eq!(cached.as_ref(), &[1, 2, 3, 4]);
        assert!(Arc::ptr_eq(&tile, &cached));
    }

    #[test]
    fn cache_evicts_least_recently_used_tile_when_over_byte_budget() {
        let cache = OperationTileCache::new(NonZeroUsize::new(2).unwrap());
        let a = Region::new(0, 0, 1, 1);
        let b = Region::new(1, 0, 1, 1);
        let c = Region::new(2, 0, 1, 1);

        cache.insert(1, a, Arc::from([10u8])).unwrap();
        cache.insert(1, b, Arc::from([20u8])).unwrap();
        let _ = cache.get(1, a).unwrap();
        cache.insert(1, c, Arc::from([30u8])).unwrap();

        assert!(cache.get(1, a).unwrap().is_some());
        assert!(cache.get(1, b).unwrap().is_none());
        assert!(cache.get(1, c).unwrap().is_some());
    }

    #[test]
    fn cache_drops_tile_larger_than_budget() {
        let cache = OperationTileCache::new(NonZeroUsize::new(2).unwrap());
        let region = Region::new(0, 0, 2, 2);

        cache.insert(9, region, Arc::from([1u8, 2, 3, 4])).unwrap();

        assert!(cache.get(9, region).unwrap().is_none());
    }

    #[test]
    fn clear_drops_all_tiles() {
        let cache = OperationTileCache::new(NonZeroUsize::new(2).unwrap());
        let region = Region::new(0, 0, 1, 1);

        cache.insert(3, region, Arc::from([42u8])).unwrap();
        cache.clear().unwrap();

        assert!(cache.get(3, region).unwrap().is_none());
    }
}
