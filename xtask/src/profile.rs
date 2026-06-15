//! `cargo xtask profile` — Side-by-side CPU and cache profiling of viprs vs libvips.
//!
//! Two backends:
//!
//! - `--tool samply`      CPU flame graph (local, macOS + Linux, no Docker).
//!                        Produces two JSON files openable in Firefox Profiler.
//!
//! - `--tool cachegrind`  Cache-miss analysis per function (Docker, cross-arch).
//!                        Prints a ranked table of L1/LL misses by function for both.
//!
//! Usage:
//!   cargo xtask profile <input> <op> [args...]
//!                       [--tool samply|cachegrind]
//!                       [--arch arm64|amd64]
//!                       [--iterations N]
//!                       [--ai]

mod args;
mod cachegrind;
mod dhat;
mod samply;

use crate::common::resolve_input;

pub(crate) fn baseline_op_name(op: &str) -> &str {
    if op == "load-exr" { "load" } else { op }
}

pub fn run(raw_args: &[String]) {
    let pa = args::parse_args(raw_args);
    let input_path = resolve_input(&pa.input);

    if !input_path.exists() {
        eprintln!("Input file not found: {}", input_path.display());
        std::process::exit(1);
    }

    if !(pa.ai_output && matches!(pa.tool, args::Tool::Samply)) {
        println!("=== xtask profile ===");
        println!("Input:      {}", input_path.display());
        println!("Operation:  {} {:?}", pa.op, pa.op_args);
        println!("Iterations: {}", pa.iterations);
        println!("Tool:       {:?}", pa.tool);
        println!();
    }

    match pa.tool {
        args::Tool::Samply => {
            samply::run_samply(
                &input_path,
                &pa.op,
                &pa.op_args,
                pa.iterations,
                pa.ai_output,
            );
        }
        args::Tool::Cachegrind => {
            cachegrind::run_cachegrind(&input_path, &pa.op, &pa.op_args, pa.iterations, pa.arch);
        }
        args::Tool::Dhat => {
            dhat::run_dhat(&input_path, &pa.op, &pa.op_args, pa.iterations, pa.arch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::baseline_op_name;

    #[test]
    fn baseline_op_name_maps_load_exr_to_load() {
        assert_eq!(baseline_op_name("load-exr"), "load");
        assert_eq!(baseline_op_name("thumbnail"), "thumbnail");
    }
}
