//! `cargo xtask perf` — Hard metrics: hardware counters, allocations, SIMD analysis.
//!
//! Three metric categories:
//! - `hw`:    perf stat counters (cache misses, IPC, branches). Requires Docker on Linux.
//! - `alloc`: heap allocation counting (Rust counting allocator + C valgrind/interposer).
//! - `simd`:  static disassembly analysis — ratio of SIMD vs scalar instructions.
//!
//! Usage:
//!   cargo xtask perf <input> <op> [args...] --metrics all|hw|alloc|simd --arch arm64|amd64

mod alloc;
pub mod args;
mod hw;
mod simd;

use crate::common::resolve_input;

pub fn run(perf_args: &[String]) {
    let pa = args::parse_args(perf_args);
    let input_path = resolve_input(&pa.input);

    if !input_path.exists() {
        eprintln!("Input file not found: {}", input_path.display());
        std::process::exit(1);
    }

    println!("=== xtask perf ===");
    println!("Input:      {}", input_path.display());
    println!("Operation:  {} {:?}", pa.op, pa.op_args);
    println!("Iterations: {}", pa.iterations);
    println!("Arch:       {:?}", pa.arch);
    println!("Metrics:    {:?}", pa.metrics);
    println!();

    match pa.metrics {
        args::Metrics::Simd => {
            simd::run_simd_analysis(pa.arch, &pa.op);
        }
        args::Metrics::Alloc => {
            alloc::run_alloc_counting(
                &input_path,
                &pa.op,
                &pa.op_args,
                pa.iterations,
                pa.ai_output,
            );
        }
        args::Metrics::Hw => {
            hw::run_hw_counters(&input_path, &pa.op, &pa.op_args, pa.iterations, pa.arch);
        }
        args::Metrics::All => {
            simd::run_simd_analysis(pa.arch, &pa.op);
            println!();
            alloc::run_alloc_counting(
                &input_path,
                &pa.op,
                &pa.op_args,
                pa.iterations,
                pa.ai_output,
            );
            println!();
            hw::run_hw_counters(&input_path, &pa.op, &pa.op_args, pa.iterations, pa.arch);
        }
    }
}
