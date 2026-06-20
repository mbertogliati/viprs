//! Tile Cache image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

#![allow(clippy::significant_drop_tightening)]
// REASON: cache-entry borrows intentionally remain scoped until tile publication completes.

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
    domain::{
        error::{SourceError, ViprsError},
        image::{DemandHint, Region},
    },
    ports::source::{ImageSource, RandomAccessSource},
};

/// Cache key: exact tile coordinates. `x` and `y` are `i32` because `Region`
/// allows negative coordinates (border extension). Width and height are `u32`.
type CacheKey = (i32, i32, u32, u32);

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

/// LRU tile cache that promotes a `SequentialSource` to `RandomAccessSource`.
///
/// `TileCache<S>` wraps any source `S` — sequential or random — and services
/// `read_region` calls from an in-memory LRU cache bounded by `max_bytes`.
/// On a cache miss it calls `S::read_region` to fill the tile, stores it, and
/// evicts older tiles until total memory falls below `max_bytes`.
///
/// # Why source wrapper and not a pipeline node
///
/// The scheduler is stateless with respect to access patterns. Making it aware of
/// a `TileCache` node would require teaching the scheduler about eviction policy,
/// byte limits, and ordering — that is cache-manager complexity that belongs in a
/// source, not a scheduler. A source wrapper keeps the scheduler transparent and
/// the cache composable: `TileCache<JpegDecoder<U8>>` is a valid `ImageSource`
/// without any pipeline changes.
///
/// # Eviction: LRU by bytes
///
/// Tiles are not uniform in size (tile geometry depends on the op above). A
/// byte-bounded LRU is more predictable for large images: the caller can say
/// "use at most 256 MiB for this stage" and the limit is respected regardless
/// of tile dimensions. A tile-count limit would require knowing tile sizes in
/// advance.
///
/// # Thread safety
///
/// `TileCache` uses a shared `RwLock` over the cache map plus an atomic access
/// clock. Cache hits take only a read lock long enough to clone the immutable
/// `Arc<[u8]>` tile; recency updates go through `CacheEntry::last_used`, so hot
/// lookups do not contend on a global writer. Cache misses compute outside the
/// lock and re-enter with a write lock only for insertion/eviction.
///
/// # Usage
///
/// ```ignore
/// // Promote a JPEG decoder (SequentialSource) to RandomAccessSource:
/// let cache: TileCache<JpegDecoder<U8>> = TileCache::new(decoder, max_bytes);
/// // Now cache: RandomAccessSource — safe to pass to Affine.
/// ```
pub struct TileCache<S: ImageSource> {
    /// The upstream source. May be sequential or random-access.
    inner: S,
    /// Maximum total bytes of cached tile data before eviction triggers.
    max_bytes: NonZeroUsize,
    shared: SharedCache,
}

impl<S: ImageSource> TileCache<S> {
    /// Constructs a new `TileCache` wrapping `inner` with a memory limit of
    /// `max_bytes`. The cache starts empty; tiles are populated on first access.
    ///
    /// # Panics
    ///
    /// Does not panic. `max_bytes` is a `NonZeroUsize` — zero-byte cache is
    /// rejected at the type level.
    pub fn new(inner: S, max_bytes: NonZeroUsize) -> Self {
        Self {
            inner,
            max_bytes,
            shared: SharedCache::new(),
        }
    }

    /// Returns the configured byte limit for the LRU eviction policy.
    pub const fn max_bytes(&self) -> NonZeroUsize {
        self.max_bytes
    }

    /// Returns a shared reference to the wrapped source.
    pub const fn inner(&self) -> &S {
        &self.inner
    }

    /// Consumes the cache and returns the inner source.
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S: ImageSource> ImageSource for TileCache<S> {
    type Format = S::Format;

    fn width(&self) -> u32 {
        self.inner.width()
    }

    fn height(&self) -> u32 {
        self.inner.height()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    /// Reports `SmallTile` — the cache is optimised for random small-tile access.
    /// Ops that request large strips will still work but will cause more evictions.
    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let key: CacheKey = (region.x, region.y, region.width, region.height);

        let cached = {
            lock_instrumentation::record_lock_acquisition();
            let state =
                self.shared
                    .state
                    .read()
                    .map_err(|_| SourceError::TileCacheMutexPoisoned {
                        phase: "lookup",
                        x: region.x,
                        y: region.y,
                        width: region.width,
                        height: region.height,
                    })?;
            state.tiles.get(&key).cloned()
        };
        if let Some(entry) = cached {
            entry.touch(self.shared.next_stamp());
            output.copy_from_slice(entry.tile.as_ref());
            return Ok(());
        }

        // Cache miss: read from the inner source outside the lock to allow
        // parallelism when the inner source is thread-safe (e.g. MemorySource).
        self.inner.read_region(region, output)?;

        // Re-acquire the lock to insert; check again in case another thread
        // raced and already populated this key.
        {
            let stamp = self.shared.next_stamp();
            lock_instrumentation::record_lock_acquisition();
            let mut state =
                self.shared
                    .state
                    .write()
                    .map_err(|_| SourceError::TileCacheMutexPoisoned {
                        phase: "insert",
                        x: region.x,
                        y: region.y,
                        width: region.width,
                        height: region.height,
                    })?;
            if let Some(existing) = state.tiles.get(&key) {
                existing.touch(stamp);
                output.copy_from_slice(existing.tile.as_ref());
            } else {
                let tile = Arc::<[u8]>::from(&*output);
                state.total_bytes += tile.len();
                state
                    .tiles
                    .insert(key, Arc::new(CacheEntry::new(tile, stamp)));

                while state.total_bytes > self.max_bytes.get() {
                    let evicted = state
                        .tiles
                        .iter()
                        .min_by_key(|(_, entry)| entry.last_used())
                        .map(|(cache_key, _)| *cache_key);
                    if let Some(evicted) = evicted {
                        if let Some(entry) = state.tiles.remove(&evicted) {
                            state.total_bytes = state.total_bytes.saturating_sub(entry.tile.len());
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        Ok(())
    }
}

/// `TileCache` serves any region in any order — that is its sole purpose.
/// This impl is what allows `TileCache<S: SequentialSource>` to be passed
/// to ops that require `RandomAccessSource`.
impl<S: ImageSource> RandomAccessSource for TileCache<S> {}

/// Convenience: if the inner source is already random-access, `TileCache`
/// still satisfies `RandomAccessSource` (covered by the blanket above).
/// No separate impl needed.

/// When the inner source is sequential, `TileCache` breaks the sequential
/// contract — it does NOT implement `SequentialSource`. This is intentional:
/// callers that need sequential guarantees (e.g. for streaming throughput)
/// must use the inner source directly, not through the cache.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::sources::memory::MemorySource;
    use crate::adapters::sources::zero::ZeroSource;
    use crate::domain::format::U8;

    fn make_cache() -> TileCache<ZeroSource<U8>> {
        let inner = ZeroSource::<U8>::new(64, 64, 1);
        TileCache::new(inner, NonZeroUsize::new(1024 * 1024).unwrap())
    }

    #[test]
    fn tile_cache_dimensions_forward_to_inner() {
        let cache = make_cache();
        assert_eq!(cache.width(), 64);
        assert_eq!(cache.height(), 64);
        assert_eq!(cache.bands(), 1);
    }

    #[test]
    fn tile_cache_demand_hint_is_small_tile() {
        let cache = make_cache();
        assert_eq!(cache.demand_hint(), DemandHint::SmallTile);
    }

    #[test]
    fn tile_cache_max_bytes_round_trips() {
        let limit = NonZeroUsize::new(512 * 1024).unwrap();
        let inner = ZeroSource::<U8>::new(8, 8, 1);
        let cache = TileCache::new(inner, limit);
        assert_eq!(cache.max_bytes(), limit);
    }

    #[test]
    fn tile_cache_into_inner_returns_source() {
        let inner = ZeroSource::<U8>::new(4, 4, 3);
        let cache = TileCache::new(inner, NonZeroUsize::new(1).unwrap());
        let recovered = cache.into_inner();
        assert_eq!(ImageSource::width(&recovered), 4);
    }

    /// Confirms that `TileCache<S>` satisfies `RandomAccessSource` at the type
    /// level. If this compiles, the trait bound is correctly implemented.
    #[test]
    fn tile_cache_satisfies_random_access_bound() {
        fn needs_random_access<T: RandomAccessSource>(_: &T) {}
        let cache = make_cache();
        needs_random_access(&cache);
    }

    /// Cache hit: a second call to `read_region` for the same tile must return
    /// the same bytes and not call the inner source a second time. We verify
    /// correctness by using a `MemorySource` with known data and checking the
    /// returned bytes match on both calls.
    #[test]
    fn tile_cache_read_region_returns_correct_pixels() {
        // 4x4 single-band image with sequential values 0..15
        let data: Vec<u8> = (0u8..16).collect();
        let inner = MemorySource::<U8>::new(4, 4, 1, data).unwrap();
        let cache = TileCache::new(inner, NonZeroUsize::new(1024).unwrap());

        let region = Region::new(0, 0, 2, 2);
        let mut out1 = vec![0u8; 4];
        let mut out2 = vec![0u8; 4];

        cache.read_region(region, &mut out1).unwrap();
        cache.read_region(region, &mut out2).unwrap();

        // pixel(0,0)=0, pixel(1,0)=1, pixel(0,1)=4, pixel(1,1)=5
        assert_eq!(out1, vec![0, 1, 4, 5]);
        assert_eq!(out2, out1, "second read should match first (cache hit)");
    }

    /// LRU eviction: with a `max_bytes` of exactly one tile (4 bytes), adding a
    /// second tile must evict the first.
    #[test]
    fn tile_cache_evicts_old_tiles_when_over_limit() {
        let data: Vec<u8> = (0u8..16).collect();
        let inner = MemorySource::<U8>::new(4, 4, 1, data).unwrap();
        // max_bytes = 4: exactly one 2x2 tile fits.
        let cache = TileCache::new(inner, NonZeroUsize::new(4).unwrap());

        let region_a = Region::new(0, 0, 2, 2); // tile A: bytes [0, 1, 4, 5]
        let region_b = Region::new(2, 0, 2, 2); // tile B: bytes [2, 3, 6, 7]

        let mut out = vec![0u8; 4];
        cache.read_region(region_a, &mut out).unwrap();

        {
            let lru = cache.shared.state.read().unwrap();
            assert_eq!(
                lru.tiles.len(),
                1,
                "one tile should be cached after first read"
            );
            assert_eq!(lru.total_bytes, 4);
        }

        cache.read_region(region_b, &mut out).unwrap();

        {
            let lru = cache.shared.state.read().unwrap();
            assert_eq!(
                lru.tiles.len(),
                1,
                "tile A should have been evicted to make room for tile B"
            );
            assert_eq!(lru.total_bytes, 4);
            let key_a = (0i32, 0i32, 2u32, 2u32);
            assert!(
                !lru.tiles.contains_key(&key_a),
                "tile A should no longer be in cache"
            );
        }
    }

    /// LRU order: a hit on an older tile should promote it, preventing eviction.
    #[test]
    fn tile_cache_hit_promotes_tile_preventing_eviction() {
        let data: Vec<u8> = (0u8..64).collect();
        let inner = MemorySource::<U8>::new(8, 8, 1, data).unwrap();
        // max_bytes = 8: fits two 2x2 tiles (4 bytes each).
        let cache = TileCache::new(inner, NonZeroUsize::new(8).unwrap());

        let region_a = Region::new(0, 0, 2, 2); // tile A
        let region_b = Region::new(2, 0, 2, 2); // tile B
        let region_c = Region::new(4, 0, 2, 2); // tile C

        let mut out = vec![0u8; 4];

        // Load A then B — order is [B (front), A (back)].
        cache.read_region(region_a, &mut out).unwrap();
        cache.read_region(region_b, &mut out).unwrap();

        // Hit A — order becomes [A (front), B (back)].
        cache.read_region(region_a, &mut out).unwrap();

        // Load C — B should be evicted (back of order), not A.
        cache.read_region(region_c, &mut out).unwrap();

        let lru = cache.shared.state.read().unwrap();
        let key_a = (0i32, 0i32, 2u32, 2u32);
        let key_b = (2i32, 0i32, 2u32, 2u32);
        let key_c = (4i32, 0i32, 2u32, 2u32);

        assert!(lru.tiles.contains_key(&key_a), "A should still be cached");
        assert!(
            !lru.tiles.contains_key(&key_b),
            "B should have been evicted"
        );
        assert!(lru.tiles.contains_key(&key_c), "C should be cached");
    }

    #[test]
    fn tile_cache_returns_typed_error_after_mutex_poison() {
        let cache = make_cache();
        let region = Region::new(0, 0, 2, 2);
        let mut out = vec![0u8; 4];

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = cache.shared.state.write().expect("test lock acquisition");
            panic!("poison cache lock");
        }));

        let err = cache
            .read_region(region, &mut out)
            .expect_err("poisoned mutex must return typed source error");
        assert!(
            matches!(
                err,
                ViprsError::Source(SourceError::TileCacheMutexPoisoned {
                    phase: "lookup",
                    ..
                })
            ),
            "expected poisoned lookup source error, got: {err:?}"
        );
    }

    #[cfg(feature = "lock_instrumentation")]
    #[test]
    fn tile_cache_counts_two_locks_on_cold_miss_and_one_on_warm_hit() {
        use crate::adapters::lock_instrumentation::{TileLockScope, snapshot};

        let data: Vec<u8> = (0u8..16).collect();
        let cache = TileCache::new(
            MemorySource::<U8>::new(4, 4, 1, data).unwrap(),
            NonZeroUsize::new(1024).unwrap(),
        );
        let region = Region::new(0, 0, 2, 2);
        let mut out = vec![0u8; 4];

        crate::adapters::lock_instrumentation::reset();
        {
            let _tile_scope = TileLockScope::new();
            cache.read_region(region, &mut out).unwrap();
        }
        let cold = snapshot();
        assert_eq!(cold.tile_count, 1);
        assert_eq!(cold.total_lock_acquisitions, 2);
        assert_eq!(cold.max_locks_per_tile, 2);

        crate::adapters::lock_instrumentation::reset();
        {
            let _tile_scope = TileLockScope::new();
            cache.read_region(region, &mut out).unwrap();
        }
        let warm = snapshot();
        assert_eq!(warm.tile_count, 1);
        assert_eq!(warm.total_lock_acquisitions, 1);
        assert_eq!(warm.max_locks_per_tile, 1);
    }
}
