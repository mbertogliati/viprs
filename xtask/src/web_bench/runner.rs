use std::collections::BTreeMap;
/// Web-bench runner: orchestrates scenario execution and reports results.
use std::fs;
use std::path::Path;
use std::process::Command;

use serde::Serialize;

use super::args::{Scenario, WebBenchArgs};
use super::scenarios::{self, ScenarioResult, format_duration, percentile};
use crate::common::repo_root;

#[derive(Serialize)]
struct JsonResult {
    scenario: String,
    iterations: u32,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    output_bytes: usize,
    peak_rss_kb: u64,
    rss_metric: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    throughput_rps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    libvips_throughput_rps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    libvips_p50_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ratio: Option<f64>,
}

#[derive(Clone, Debug)]
struct LibvipsComparison {
    p50_ms: f64,
    throughput_rps: Option<f64>,
}

pub fn run_web_bench(input_path: &Path, args: &WebBenchArgs) {
    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║  viprs web-bench: web-service image processing      ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");
    eprintln!();

    let input_bytes = fs::read(input_path).unwrap_or_else(|e| {
        eprintln!("Failed to read input: {e}");
        std::process::exit(1);
    });

    eprintln!(
        "Input: {} ({} bytes, {:.1} KB)",
        input_path.display(),
        input_bytes.len(),
        input_bytes.len() as f64 / 1024.0
    );
    eprintln!("Iterations: {}", args.iterations);
    eprintln!("Memory metric: rssΔ = sampled resident-set delta from scenario baseline");
    eprintln!();

    let large_input_path = resolve_large_input();
    let large_bytes = large_input_path.as_ref().and_then(|p| fs::read(p).ok());

    let mut results = Vec::new();

    match &args.scenario {
        Scenario::ThumbnailBytes => {
            results.push(run_scenario_thumbnail_bytes(&input_bytes, args));
        }
        Scenario::PipelineBytes => {
            results.push(run_scenario_pipeline_bytes(&input_bytes, args));
        }
        Scenario::Concurrent => {
            results.extend(run_scenario_concurrent(&input_bytes, args));
        }
        Scenario::LargeUpload => {
            if let Some(ref large) = large_bytes {
                results.push(run_scenario_large_upload(large, args));
            } else {
                eprintln!("⚠ Skipping large-upload: no 8192×8192 fixture found");
            }
        }
        Scenario::All => {
            results.push(run_scenario_thumbnail_bytes(&input_bytes, args));
            results.push(run_scenario_pipeline_bytes(&input_bytes, args));
            results.extend(run_scenario_concurrent(&input_bytes, args));
            if let Some(ref large) = large_bytes {
                results.push(run_scenario_large_upload(large, args));
            } else {
                eprintln!("⚠ Skipping large-upload: no 8192×8192 fixture found");
            }
        }
    }

    // Try libvips comparison
    let libvips_available = check_libvips_web_runner();

    if args.json_output {
        print_json_results(&results, libvips_available, input_path, args);
    } else {
        print_table_results(&results, libvips_available, input_path, args);
    }
}

fn run_scenario_thumbnail_bytes(input_bytes: &[u8], args: &WebBenchArgs) -> ScenarioResult {
    eprintln!("▶ thumbnail-bytes: decode → thumbnail(400) → WebP");
    let result = scenarios::thumbnail_bytes(input_bytes, args.iterations);
    print_quick_summary(&result);
    result
}

fn run_scenario_pipeline_bytes(input_bytes: &[u8], args: &WebBenchArgs) -> ScenarioResult {
    eprintln!("▶ pipeline-bytes: decode → thumbnail(800) + sharpen + linear(1.1,5) → JPEG q85");
    let result = scenarios::pipeline_bytes(input_bytes, args.iterations);
    print_quick_summary(&result);
    result
}

fn run_scenario_concurrent(input_bytes: &[u8], args: &WebBenchArgs) -> Vec<ScenarioResult> {
    eprintln!("▶ concurrent: parallel thumbnail-bytes requests");
    let iters_per_thread = (args.iterations / 4).max(5);
    args.concurrency
        .iter()
        .map(|&n| {
            let result = scenarios::concurrent(input_bytes, n, iters_per_thread);
            result
        })
        .collect()
}

fn run_scenario_large_upload(input_bytes: &[u8], args: &WebBenchArgs) -> ScenarioResult {
    eprintln!("▶ large-upload: 8192×8192 → thumbnail(400)");
    // Use fewer iterations for large images
    let iters = (args.iterations / 3).max(5);
    let result = scenarios::large_upload(input_bytes, iters);
    print_quick_summary(&result);
    result
}

fn print_quick_summary(result: &ScenarioResult) {
    let mut sorted = result.latencies_ns.clone();
    sorted.sort_unstable();
    let p50 = percentile(&sorted, 50.0);
    let p95 = percentile(&sorted, 95.0);
    eprintln!(
        "  p50={} p95={} output={}B rssΔ={}KB",
        format_duration(p50),
        format_duration(p95),
        result.output_bytes,
        result.peak_rss_kb
    );
    eprintln!();
}

fn print_table_results(
    results: &[ScenarioResult],
    libvips_available: bool,
    input_path: &Path,
    args: &WebBenchArgs,
) {
    eprintln!("┌─────────────────────────────────────────────────────────────────────┐");
    eprintln!("│ RESULTS SUMMARY                                                     │");
    eprintln!("├─────────────────────┬──────────┬──────────┬──────────┬──────────────┤");
    eprintln!("│ Scenario            │ p50      │ p95      │ p99      │ Output       │");
    eprintln!("├─────────────────────┼──────────┼──────────┼──────────┼──────────────┤");

    for result in results {
        let mut sorted = result.latencies_ns.clone();
        sorted.sort_unstable();
        let p50 = format_duration(percentile(&sorted, 50.0));
        let p95 = format_duration(percentile(&sorted, 95.0));
        let p99 = format_duration(percentile(&sorted, 99.0));
        let output = if result.output_bytes > 0 {
            format!("{:.1}KB", result.output_bytes as f64 / 1024.0)
        } else {
            "—".to_string()
        };
        eprintln!(
            "│ {:19} │ {:8} │ {:8} │ {:8} │ {:12} │",
            result.name, p50, p95, p99, output
        );
    }
    eprintln!("└─────────────────────┴──────────┴──────────┴──────────┴──────────────┘");

    if !libvips_available {
        eprintln!();
        eprintln!("ℹ libvips web-runner not built. To enable comparison:");
        eprintln!("  cd tools/bench-vs-libvips && make libvips-web-runner");
    }

    // Libvips comparison via existing runner with workflow op
    if libvips_available {
        eprintln!();
        eprintln!("── libvips comparison (workflow mode) ──");
        if let Some(large_input_path) = resolve_large_input() {
            run_libvips_comparison(results, input_path, large_input_path.as_path(), args);
        } else {
            run_libvips_comparison(results, input_path, input_path, args);
        }
    }
}

fn print_json_results(
    results: &[ScenarioResult],
    libvips_available: bool,
    input_path: &Path,
    args: &WebBenchArgs,
) {
    let comparisons = if libvips_available {
        let large_input_path = resolve_large_input().unwrap_or_else(|| input_path.to_path_buf());
        run_libvips_comparison_collect(results, input_path, large_input_path.as_path(), args)
    } else {
        BTreeMap::new()
    };

    let json_results: Vec<JsonResult> = results
        .iter()
        .map(|r| {
            let mut sorted = r.latencies_ns.clone();
            sorted.sort_unstable();
            let p50 = percentile(&sorted, 50.0);
            let p95 = percentile(&sorted, 95.0);
            let p99 = percentile(&sorted, 99.0);
            let throughput_rps = if r.name.starts_with("concurrent-") {
                Some(r.iterations as f64 / (r.wall_total_ns as f64 / 1_000_000_000.0))
            } else {
                None
            };
            let comparison = comparisons.get(&r.name);
            let libvips_p50_ms = comparison.map(|entry| entry.p50_ms);
            let libvips_throughput_rps = comparison.and_then(|entry| entry.throughput_rps);
            let ratio = libvips_p50_ms
                .map(|baseline| p50 as f64 / 1_000_000.0 / baseline.max(f64::EPSILON));

            JsonResult {
                scenario: r.name.clone(),
                iterations: r.iterations,
                p50_ms: p50 as f64 / 1_000_000.0,
                p95_ms: p95 as f64 / 1_000_000.0,
                p99_ms: p99 as f64 / 1_000_000.0,
                output_bytes: r.output_bytes,
                peak_rss_kb: r.peak_rss_kb,
                rss_metric: "sampled resident-set delta from scenario baseline",
                throughput_rps,
                libvips_throughput_rps,
                libvips_p50_ms,
                ratio,
            }
        })
        .collect();

    match serde_json::to_string_pretty(&json_results) {
        Ok(json) => println!("{json}"),
        Err(e) => eprintln!("Failed to serialize JSON: {e}"),
    }
}

fn check_libvips_web_runner() -> bool {
    let runner_path = repo_root().join("tools/bench-vs-libvips/libvips-web-runner");
    runner_path.exists()
}

fn run_libvips_comparison(
    results: &[ScenarioResult],
    input_path: &Path,
    large_input_path: &Path,
    args: &WebBenchArgs,
) {
    let comparisons = run_libvips_comparison_collect(results, input_path, large_input_path, args);
    for result in results {
        if let Some(comparison) = comparisons.get(&result.name) {
            let mut sorted = result.latencies_ns.clone();
            sorted.sort_unstable();
            let viprs_p50_ms = percentile(&sorted, 50.0) as f64 / 1_000_000.0;
            let ratio = viprs_p50_ms / comparison.p50_ms.max(f64::EPSILON);
            if let Some(throughput) = comparison.throughput_rps {
                eprintln!(
                    "  {:19} libvips p50={:.1}ms ratio={:.3}x throughput={:.1} req/s",
                    result.name, comparison.p50_ms, ratio, throughput
                );
            } else {
                eprintln!(
                    "  {:19} libvips p50={:.1}ms ratio={:.3}x",
                    result.name, comparison.p50_ms, ratio
                );
            }
        }
    }
}

fn run_libvips_comparison_collect(
    results: &[ScenarioResult],
    input_path: &Path,
    large_input_path: &Path,
    args: &WebBenchArgs,
) -> BTreeMap<String, LibvipsComparison> {
    let runner_path = repo_root().join("tools/bench-vs-libvips/libvips-web-runner");
    if !runner_path.exists() {
        return BTreeMap::new();
    }

    let mut comparisons = BTreeMap::new();

    for result in results {
        let mut cmd = Command::new(&runner_path);
        let scenario_input = if result.name == "large-upload" {
            large_input_path
        } else {
            input_path
        };
        cmd.arg(scenario_input);

        let scenario_args: Vec<String> = match result.name.as_str() {
            "thumbnail-bytes" => vec!["thumbnail-bytes".into(), "400".into()],
            "pipeline-bytes" => vec!["pipeline-bytes".into(), "800".into(), "85".into()],
            "large-upload" => vec!["large-upload".into(), "400".into()],
            label if label.starts_with("concurrent-") => {
                let concurrency = label.trim_start_matches("concurrent-");
                let per_thread_iterations = (args.iterations / 4).max(5).to_string();
                vec![
                    "concurrent".into(),
                    "400".into(),
                    concurrency.into(),
                    per_thread_iterations,
                ]
            }
            _ => continue,
        };
        for a in &scenario_args {
            cmd.arg(a);
        }
        let iterations = if result.name == "large-upload" {
            (args.iterations / 3).max(5)
        } else {
            args.iterations
        };
        cmd.arg("--iterations").arg(iterations.to_string());

        match cmd.output() {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                if let Some(parsed) = parse_libvips_result(&stdout) {
                    comparisons.insert(result.name.clone(), parsed);
                }
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                eprintln!("  libvips {} failed: {}", result.name, stderr.trim());
            }
            Err(e) => {
                eprintln!("  Failed to run libvips web-runner: {e}");
                return comparisons;
            }
        }
    }
    comparisons
}

fn parse_libvips_result(json_str: &str) -> Option<LibvipsComparison> {
    let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let wall_ns = parsed.get("wall_ns")?.as_array()?;
    let mut values: Vec<u64> = wall_ns.iter().filter_map(|v| v.as_u64()).collect();
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let p50_ns = percentile(&values, 50.0);
    let throughput_rps = parsed
        .get("wall_total_ns")
        .and_then(|v| v.as_u64())
        .filter(|total| *total > 0)
        .map(|total| {
            let iterations = parsed
                .get("iterations")
                .and_then(|v| v.as_u64())
                .unwrap_or(values.len() as u64);
            iterations as f64 / (total as f64 / 1_000_000_000.0)
        });
    Some(LibvipsComparison {
        p50_ms: p50_ns as f64 / 1_000_000.0,
        throughput_rps,
    })
}

fn resolve_large_input() -> Option<std::path::PathBuf> {
    let root = repo_root();
    let candidates = [
        "tests/fixtures/images/bench_8192x8192.jpg",
        "tests/fixtures/images/bench_8192x8192.png",
        "tests/fixtures/images/bench_8192x8192.webp",
    ];
    for c in &candidates {
        let p = root.join(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}
