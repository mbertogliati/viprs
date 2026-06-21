#![cfg(test)]
#![allow(dead_code)]

//! Test-only helpers for tracking allocations and running isolated measurement subprocesses.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
const ALLOC_STATS_PREFIX: &str = "VIPRS_ALLOC_STATS";

pub(crate) struct CountingAllocator;

// SAFETY: Delegates every allocation primitive to `System`. The extra atomic
// bookkeeping does not affect the allocator contract and uses relaxed ordering
// because tests read the counters only after the measured decode completes.
unsafe impl GlobalAlloc for CountingAllocator {
    // SAFETY: callers uphold `GlobalAlloc::alloc` by passing a valid `Layout`; this implementation forwards that exact request to `System` and only updates atomics alongside it.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards the exact `layout` request to the system allocator.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let size = layout.size() as u64;
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(size, Ordering::Relaxed);
            let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
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

    // SAFETY: callers uphold `GlobalAlloc::dealloc` by passing the original pointer/layout pair from `alloc`; this implementation forwards the same pair to `System` after updating counters.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size() as u64;
        let _ = LIVE_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
            Some(live.saturating_sub(size))
        });
        // SAFETY: forwards the original pointer/layout pair back to the system allocator.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocStats {
    pub(crate) alloc_count: u64,
    pub(crate) alloc_bytes: u64,
    pub(crate) peak_live_bytes: u64,
}

pub(crate) fn reset_alloc_stats() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
}

#[must_use]
pub(crate) fn alloc_stats() -> AllocStats {
    AllocStats {
        alloc_count: ALLOC_COUNT.load(Ordering::Relaxed),
        alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        peak_live_bytes: PEAK_LIVE_BYTES.load(Ordering::Relaxed),
    }
}

pub(crate) fn should_run_alloc_stats_child(env_var: &str) -> bool {
    std::env::var_os(env_var).is_some()
}

pub(crate) fn emit_alloc_stats(stats: AllocStats) {
    println!(
        "{ALLOC_STATS_PREFIX} alloc_count={} alloc_bytes={} peak_live_bytes={}",
        stats.alloc_count, stats.alloc_bytes, stats.peak_live_bytes
    );
}

pub(crate) fn run_alloc_stats_child(test_name: &str, env_var: &str) -> AllocStats {
    let output = Command::new(std::env::current_exe().unwrap())
        .env(env_var, "1")
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "allocation child run failed for {test_name}: stdout=\n{}\nstderr=\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let combined_output = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let metrics_line = combined_output
        .lines()
        .find_map(|line| {
            line.split_once(ALLOC_STATS_PREFIX)
                .map(|(_, metrics)| metrics)
        })
        .unwrap_or_else(|| panic!("missing alloc stats for {test_name}: {combined_output}"));

    parse_alloc_stats(metrics_line.trim())
}

fn parse_alloc_stats(line: &str) -> AllocStats {
    let mut stats = AllocStats {
        alloc_count: 0,
        alloc_bytes: 0,
        peak_live_bytes: 0,
    };

    for field in line.split_whitespace() {
        let Some((key, value)) = field.split_once('=') else {
            continue;
        };
        match key {
            "alloc_count" => stats.alloc_count = value.parse().unwrap(),
            "alloc_bytes" => stats.alloc_bytes = value.parse().unwrap(),
            "peak_live_bytes" => stats.peak_live_bytes = value.parse().unwrap(),
            _ => {}
        }
    }

    stats
}
