//! Counting allocator — tracks number of allocations and total bytes.
//!
//! Activated by the `count-alloc` feature flag. When active, wraps the system
//! allocator and increments atomic counters on every alloc/dealloc.
//!
//! Usage: call `reset()` before the measured section, then `snapshot()` after.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);

pub struct CountingAllocator;

// SAFETY: Delegates all operations to the System allocator which is sound.
// The atomic counters are plain relaxed increments — no memory ordering needed
// for correctness since they are only read after the measured section completes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && TRACKING_ENABLED.load(Ordering::Relaxed) {
            let size = layout.size() as u64;
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(size, Ordering::Relaxed);
            let live = increase_live_bytes(size);
            // Update peak (relaxed CAS loop)
            let mut peak = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
            while live > peak {
                match PEAK_LIVE_BYTES.compare_exchange_weak(
                    peak,
                    live,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => peak = actual,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACKING_ENABLED.load(Ordering::Relaxed) {
            let size = layout.size() as u64;
            decrease_live_bytes(size);
        }
        unsafe { System.dealloc(ptr, layout) };
    }
}

fn increase_live_bytes(size: u64) -> u64 {
    update_live_bytes(|current| current.saturating_add(size))
}

fn decrease_live_bytes(size: u64) -> u64 {
    update_live_bytes(|current| current.saturating_sub(size))
}

fn update_live_bytes(update: impl Fn(u64) -> u64) -> u64 {
    let mut current = LIVE_BYTES.load(Ordering::Relaxed);
    loop {
        let next = update(current);
        match LIVE_BYTES.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => return next,
            Err(actual) => current = actual,
        }
    }
}

/// Reset all counters to zero. Call before the measured section.
pub fn reset() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
}

pub fn set_enabled(enabled: bool) {
    TRACKING_ENABLED.store(enabled, Ordering::Relaxed);
}

pub struct CountingSession;

impl CountingSession {
    pub fn start() -> Self {
        reset();
        set_enabled(true);
        Self
    }
}

impl Drop for CountingSession {
    fn drop(&mut self) {
        set_enabled(false);
    }
}

/// Snapshot current counters.
pub struct AllocStats {
    pub alloc_count: u64,
    pub alloc_bytes: u64,
    pub peak_live_bytes: u64,
}

pub fn snapshot() -> AllocStats {
    AllocStats {
        alloc_count: ALLOC_COUNT.load(Ordering::Relaxed),
        alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        peak_live_bytes: PEAK_LIVE_BYTES.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::{CountingSession, decrease_live_bytes, increase_live_bytes, reset, snapshot};

    #[test]
    fn counts_allocations_only_while_session_is_active() {
        reset();
        let inactive = vec![1u8; 32];
        std::hint::black_box(inactive);
        let inactive_stats = snapshot();
        assert_eq!(inactive_stats.alloc_count, 0);

        let active_stats = {
            let _session = CountingSession::start();
            let active = vec![1u8; 64];
            std::hint::black_box(active);
            snapshot()
        };
        assert!(active_stats.alloc_count > 0);
        assert!(active_stats.alloc_bytes >= 64);

        reset();
        let after = vec![1u8; 16];
        std::hint::black_box(after);
        let after_stats = snapshot();
        assert_eq!(after_stats.alloc_count, 0);
    }

    #[test]
    fn live_bytes_helpers_saturate_instead_of_wrapping() {
        reset();
        assert_eq!(decrease_live_bytes(8), 0);
        assert_eq!(increase_live_bytes(4), 4);
        assert_eq!(decrease_live_bytes(10), 0);
    }
}
