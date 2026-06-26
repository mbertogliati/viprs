#[cfg(target_os = "linux")]
use std::fs;
/// Web-service benchmark scenarios.
///
/// Each scenario simulates a real web-service image processing pattern:
/// receive bytes → decode → transform pipeline → encode → return bytes.
use std::hint::black_box;
use std::time::{Duration, Instant};

use viprs::adapters::image_api::ImagePipeline2;

/// Result of a single scenario run (multiple iterations).
pub struct ScenarioResult {
    pub name: String,
    pub iterations: u32,
    pub latencies_ns: Vec<u64>,
    pub wall_total_ns: u64,
    pub output_bytes: usize,
    pub peak_rss_kb: u64,
}

/// Thumbnail-bytes: decode from buffer → thumbnail(400) → encode WebP.
pub fn thumbnail_bytes(input_bytes: &[u8], iterations: u32) -> ScenarioResult {
    let mut latencies = Vec::with_capacity(iterations as usize);
    let mut output_bytes = 0;
    let mut peak_rss = 0;

    // Warmup
    for _ in 0..3 {
        let out = run_thumbnail_bytes(input_bytes);
        black_box(&out);
    }

    let rss_before = current_rss_kb();
    for _ in 0..iterations {
        let start = Instant::now();
        let out = run_thumbnail_bytes(input_bytes);
        let elapsed = start.elapsed();
        latencies.push(elapsed.as_nanos() as u64);
        output_bytes = out.len();
        peak_rss = peak_rss.max(current_rss_kb().saturating_sub(rss_before));
        black_box(&out);
    }

    ScenarioResult {
        name: "thumbnail-bytes".to_string(),
        iterations,
        wall_total_ns: latencies.iter().sum(),
        latencies_ns: latencies,
        output_bytes,
        peak_rss_kb: peak_rss,
    }
}

/// Pipeline-bytes: decode → thumbnail(800) + sharpen + linear(1.1, 5) → encode JPEG q85.
pub fn pipeline_bytes(input_bytes: &[u8], iterations: u32) -> ScenarioResult {
    let mut latencies = Vec::with_capacity(iterations as usize);
    let mut output_bytes = 0;
    let mut peak_rss = 0;

    // Warmup
    for _ in 0..3 {
        let out = run_pipeline_bytes(input_bytes);
        black_box(&out);
    }

    let rss_before = current_rss_kb();
    for _ in 0..iterations {
        let start = Instant::now();
        let out = run_pipeline_bytes(input_bytes);
        let elapsed = start.elapsed();
        latencies.push(elapsed.as_nanos() as u64);
        output_bytes = out.len();
        peak_rss = peak_rss.max(current_rss_kb().saturating_sub(rss_before));
        black_box(&out);
    }

    ScenarioResult {
        name: "pipeline-bytes".to_string(),
        iterations,
        wall_total_ns: latencies.iter().sum(),
        latencies_ns: latencies,
        output_bytes,
        peak_rss_kb: peak_rss,
    }
}

/// Concurrent-N: run N parallel thumbnail-bytes requests and measure throughput.
pub fn concurrent(
    input_bytes: &[u8],
    concurrency: u32,
    iterations_per_thread: u32,
) -> ScenarioResult {
    use std::sync::Arc;
    use std::thread;

    let shared_bytes = Arc::new(input_bytes.to_vec());
    let mut handles = Vec::with_capacity(concurrency as usize);

    // Warmup
    let _ = run_thumbnail_bytes(input_bytes);

    let wall_start = Instant::now();
    let rss_before = current_rss_kb();

    for _ in 0..concurrency {
        let bytes = Arc::clone(&shared_bytes);
        let iters = iterations_per_thread;
        handles.push(thread::spawn(move || {
            let mut latencies = Vec::with_capacity(iters as usize);
            for _ in 0..iters {
                let start = Instant::now();
                let out = run_thumbnail_bytes(&bytes);
                latencies.push(start.elapsed().as_nanos() as u64);
                black_box(&out);
            }
            latencies
        }));
    }

    let mut all_latencies = Vec::new();
    for h in handles {
        match h.join() {
            Ok(lats) => all_latencies.extend(lats),
            Err(_) => eprintln!("  warning: thread panicked during concurrent benchmark"),
        }
    }
    let wall_elapsed = wall_start.elapsed();
    let peak_rss = current_rss_kb().saturating_sub(rss_before);

    let total_requests = concurrency * iterations_per_thread;
    let throughput = total_requests as f64 / wall_elapsed.as_secs_f64();

    eprintln!(
        "  concurrent-{concurrency}: {total_requests} requests in {:.2}s = {throughput:.1} req/s",
        wall_elapsed.as_secs_f64()
    );

    ScenarioResult {
        name: format!("concurrent-{concurrency}"),
        iterations: total_requests,
        wall_total_ns: wall_elapsed.as_nanos() as u64,
        latencies_ns: all_latencies,
        output_bytes: 0,
        peak_rss_kb: peak_rss,
    }
}

/// Large-upload: 8192×8192 image → thumbnail(400).
pub fn large_upload(input_bytes: &[u8], iterations: u32) -> ScenarioResult {
    let mut latencies = Vec::with_capacity(iterations as usize);
    let mut output_bytes = 0;
    let mut peak_rss = 0;

    // Warmup
    let out = run_thumbnail_bytes(input_bytes);
    black_box(&out);

    let rss_before = current_rss_kb();
    for _ in 0..iterations {
        let start = Instant::now();
        let out = run_thumbnail_bytes(input_bytes);
        let elapsed = start.elapsed();
        latencies.push(elapsed.as_nanos() as u64);
        output_bytes = out.len();
        peak_rss = peak_rss.max(current_rss_kb().saturating_sub(rss_before));
        black_box(&out);
    }

    ScenarioResult {
        name: "large-upload".to_string(),
        iterations,
        wall_total_ns: latencies.iter().sum(),
        latencies_ns: latencies,
        output_bytes,
        peak_rss_kb: peak_rss,
    }
}

// ─── Internal helpers ───────────────────────────────────────────────────────

fn run_thumbnail_bytes(input_bytes: &[u8]) -> Vec<u8> {
    ImagePipeline2::from_bytes(input_bytes)
        .and_then(|api| api.thumbnail(400).map_err(Into::into))
        .and_then(|api| api.encode_webp(80))
        .unwrap_or_default()
}

fn run_pipeline_bytes(input_bytes: &[u8]) -> Vec<u8> {
    ImagePipeline2::from_bytes(input_bytes)
        .and_then(|api| api.thumbnail(800).map_err(Into::into))
        .and_then(|api| api.sharpen().map_err(Into::into))
        .and_then(|api| api.linear(1.1, 5.0).map_err(Into::into))
        .and_then(|api| api.encode_jpeg(85))
        .unwrap_or_default()
}

fn current_rss_kb() -> u64 {
    current_resident_kb()
}

#[cfg(target_os = "linux")]
fn current_resident_kb() -> u64 {
    let Ok(statm) = fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    let mut fields = statm.split_whitespace();
    let _size_pages = fields.next();
    let Some(resident_pages) = fields.next().and_then(|value| value.parse::<u64>().ok()) else {
        return 0;
    };
    // SAFETY: `sysconf(_SC_PAGESIZE)` is thread-safe and does not require any additional invariants.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return 0;
    }
    resident_pages.saturating_mul(page_size as u64) / 1024
}

#[cfg(target_os = "macos")]
fn current_resident_kb() -> u64 {
    use std::mem::{MaybeUninit, size_of};

    let mut info = MaybeUninit::<libc::mach_task_basic_info>::uninit();
    let mut count = (size_of::<libc::mach_task_basic_info>() / size_of::<libc::integer_t>())
        as libc::mach_msg_type_number_t;
    // SAFETY: `info` points to valid writable memory for `mach_task_basic_info`, and `count`
    // is initialized to the number of `integer_t` words required by the kernel API.
    // REASON: mach2 crate migration tracked as tech debt; libc version still functional.
    #[allow(deprecated)]
    let result = unsafe {
        libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast::<libc::integer_t>(),
            &mut count,
        )
    };
    if result != libc::KERN_SUCCESS {
        return 0;
    }
    // SAFETY: `task_info` returned success, so the kernel initialized `info`.
    let info = unsafe { info.assume_init() };
    (info.resident_size as u64) / 1024
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn current_resident_kb() -> u64 {
    0
}

/// Compute percentile from a sorted slice of nanosecond timings.
pub fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Format nanoseconds as human-readable duration.
pub fn format_duration(ns: u64) -> String {
    let d = Duration::from_nanos(ns);
    if d.as_secs() > 0 {
        format!("{:.2}s", d.as_secs_f64())
    } else if d.as_millis() > 0 {
        format!("{:.1}ms", d.as_secs_f64() * 1000.0)
    } else {
        format!("{:.1}µs", d.as_nanos() as f64 / 1000.0)
    }
}
