mod bench;
mod common;
mod counting_alloc;
mod docker;
mod perf;
mod profile;
mod web_bench;

use std::env;

#[global_allocator]
static ALLOC: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    let help_requested = args
        .first()
        .map(|a| a == "--help" || a == "-h")
        .unwrap_or(false);

    if args.is_empty() || help_requested {
        eprintln!("viprs benchmarking and profiling toolchain");
        eprintln!();
        eprintln!("USAGE:");
        eprintln!("  cargo xtask <command> [options]");
        eprintln!("  cargo xtask <command> --help      Show command-specific help");
        eprintln!();
        eprintln!("COMMANDS:");
        eprintln!("  bench    Measure wall-clock latency: viprs vs libvips, p50/p95 side-by-side.");
        eprintln!("           Start here. A ratio ≥ 2x triggers a P-NNN task.");
        eprintln!();
        eprintln!("  profile  Pinpoint WHERE the gap is: CPU flame graph (samply) or per-function");
        eprintln!("           cache-miss table (cachegrind). Profiles both binaries and compares.");
        eprintln!("           Use after bench confirms a gap.");
        eprintln!();
        eprintln!(
            "  perf     Low-level hard metrics: SIMD instruction % (static), heap allocation"
        );
        eprintln!("           count (dhat), aggregate cache counters (Docker). Use to confirm a");
        eprintln!("           hypothesis from profile or validate that a fix worked.");
        eprintln!();
        eprintln!(
            "  web-bench  Web-service workload simulation: bytes→decode→pipeline→encode→bytes."
        );
        eprintln!(
            "             Measures latency, throughput, and concurrency for HTTP-like flows."
        );
        eprintln!();
        eprintln!("TYPICAL WORKFLOW:");
        eprintln!("  1. cargo xtask bench  input.jpg thumbnail 400   # is there a gap?");
        eprintln!("  2. cargo xtask profile input.jpg thumbnail 400   # where is it?");
        eprintln!("  3. cargo xtask perf   input.jpg thumbnail 400 --metrics simd  # confirm fix");
        eprintln!();
        eprintln!("Run 'cargo xtask <command> --help' for full options.");
        std::process::exit(if args.is_empty() { 1 } else { 0 });
    }

    match args[0].as_str() {
        "bench" => bench::run(&args[1..]),
        "profile" => profile::run(&args[1..]),
        "perf" => perf::run(&args[1..]),
        "web-bench" => web_bench::run(&args[1..]),
        other => {
            eprintln!("Unknown command: {other}");
            std::process::exit(1);
        }
    }
}
