#[cfg(all(test, feature = "lock_instrumentation"))]
use std::sync::{Mutex, MutexGuard};
#[cfg(feature = "lock_instrumentation")]
use std::{
    cell::{Cell, RefCell},
    sync::atomic::{AtomicU64, Ordering},
};

#[cfg(feature = "lock_instrumentation")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LockInstrumentationSnapshot {
    pub tile_count: u64,
    pub total_lock_acquisitions: u64,
    pub max_locks_per_tile: u64,
}

#[cfg(feature = "lock_instrumentation")]
static TILE_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "lock_instrumentation")]
static TOTAL_LOCK_ACQUISITIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "lock_instrumentation")]
static MAX_LOCKS_PER_TILE: AtomicU64 = AtomicU64::new(0);
#[cfg(all(test, feature = "lock_instrumentation"))]
static TEST_RUN_GUARD: Mutex<()> = Mutex::new(());
#[cfg(all(test, feature = "lock_instrumentation"))]
thread_local! {
    static RUN_GUARD_DEPTH: Cell<u32> = const { Cell::new(0) };
}

#[cfg(feature = "lock_instrumentation")]
thread_local! {
    static TILE_SCOPE_DEPTH: Cell<u32> = const { Cell::new(0) };
    static TILE_LOCK_COUNT: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "lock_instrumentation")]
fn update_max_lock_count(candidate: u64) {
    let mut observed = MAX_LOCKS_PER_TILE.load(Ordering::Relaxed);
    while candidate > observed {
        match MAX_LOCKS_PER_TILE.compare_exchange_weak(
            observed,
            candidate,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(current) => observed = current,
        }
    }
}

#[cfg(feature = "lock_instrumentation")]
fn finish_current_tile() {
    TILE_LOCK_COUNT.with(|counts| {
        if let Some(count) = counts.borrow_mut().pop() {
            TILE_COUNT.fetch_add(1, Ordering::Relaxed);
            TOTAL_LOCK_ACQUISITIONS.fetch_add(count, Ordering::Relaxed);
            update_max_lock_count(count);
        }
    });
}

#[cfg(feature = "lock_instrumentation")]
pub(crate) fn reset() {
    TILE_COUNT.store(0, Ordering::Relaxed);
    TOTAL_LOCK_ACQUISITIONS.store(0, Ordering::Relaxed);
    MAX_LOCKS_PER_TILE.store(0, Ordering::Relaxed);
    TILE_SCOPE_DEPTH.with(|depth| depth.set(0));
    TILE_LOCK_COUNT.with(|counts| counts.borrow_mut().clear());
}

#[cfg(not(feature = "lock_instrumentation"))]
pub const fn reset() {}

#[cfg(all(test, feature = "lock_instrumentation"))]
pub(crate) struct RunGuard(Option<MutexGuard<'static, ()>>);

#[cfg(not(all(test, feature = "lock_instrumentation")))]
pub struct RunGuard;

#[cfg(all(test, feature = "lock_instrumentation"))]
pub(crate) fn prepare_run() -> RunGuard {
    let mut should_reset = false;
    let mut should_lock = false;
    RUN_GUARD_DEPTH.with(|depth| {
        if depth.get() == 0 {
            should_reset = true;
            should_lock = true;
        }
        depth.set(depth.get() + 1);
    });

    let guard = if should_lock {
        Some(
            TEST_RUN_GUARD
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    } else {
        None
    };
    if should_reset {
        reset();
    }
    RunGuard(guard)
}

#[cfg(not(all(test, feature = "lock_instrumentation")))]
pub fn prepare_run() -> RunGuard {
    reset();
    RunGuard
}

#[cfg(all(test, feature = "lock_instrumentation"))]
impl Drop for RunGuard {
    fn drop(&mut self) {
        RUN_GUARD_DEPTH.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0, "run guard underflow");
            depth.set(current.saturating_sub(1));
        });
    }
}

#[cfg(feature = "lock_instrumentation")]
pub(crate) fn record_lock_acquisition() {
    TILE_LOCK_COUNT.with(|counts| {
        if let Some(current) = counts.borrow_mut().last_mut() {
            *current += 1;
        }
    });
}

#[cfg(not(feature = "lock_instrumentation"))]
pub const fn record_lock_acquisition() {}

#[cfg(feature = "lock_instrumentation")]
pub(crate) fn snapshot() -> LockInstrumentationSnapshot {
    LockInstrumentationSnapshot {
        tile_count: TILE_COUNT.load(Ordering::Relaxed),
        total_lock_acquisitions: TOTAL_LOCK_ACQUISITIONS.load(Ordering::Relaxed),
        max_locks_per_tile: MAX_LOCKS_PER_TILE.load(Ordering::Relaxed),
    }
}

#[cfg(not(feature = "lock_instrumentation"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub struct LockInstrumentationSnapshot;

#[cfg(not(feature = "lock_instrumentation"))]
#[allow(dead_code)]
pub const fn snapshot() -> LockInstrumentationSnapshot {
    LockInstrumentationSnapshot
}

pub struct TileLockScope;

impl TileLockScope {
    #[must_use]
    pub(crate) fn new() -> Self {
        #[cfg(feature = "lock_instrumentation")]
        {
            TILE_SCOPE_DEPTH.with(|depth| depth.set(depth.get() + 1));
            TILE_LOCK_COUNT.with(|counts| counts.borrow_mut().push(0));
        }

        Self
    }
}

impl Drop for TileLockScope {
    fn drop(&mut self) {
        #[cfg(feature = "lock_instrumentation")]
        {
            TILE_SCOPE_DEPTH.with(|depth| {
                let current = depth.get();
                debug_assert!(current > 0, "tile lock scope underflow");
                depth.set(current.saturating_sub(1));
            });
            finish_current_tile();
        }
    }
}
