use std::path::Path;
use std::process::Command;

use serde::Deserialize;

use crate::bench;
use crate::common::repo_root;
use crate::counting_alloc::{self, AllocStats, CountingSession};

const AI_ALLOC_SITE_LIMIT: usize = 15;
const AI_CALLER_DEPTH: usize = 3;
const CALLER_CONTINUATION_PREFIX: &str = "                                        ";

pub fn run_alloc_counting(
    input: &Path,
    op: &str,
    op_args: &[String],
    iterations: usize,
    ai_output: bool,
) {
    println!("--- Allocation analysis (viprs only) ---");
    println!();
    println!("  Totals come from xtask's in-process counting allocator.");
    println!("  C library allocations (libpng, libjpeg, etc.) are INVISIBLE here.");
    println!("  If the Rust total looks suspiciously small, the real allocator pressure");
    println!("  is in native code. Use the Docker dhat tool to profile the full process:");
    println!();
    println!("       cargo xtask profile <input> <op> --tool dhat");
    println!();
    let stats = run_counting_alloc_in_process(input, op, op_args, iterations);
    println!("  Total allocations:    {}", stats.alloc_count);
    println!("  Total bytes allocated:{}", stats.alloc_bytes);
    println!(
        "  Per-iteration allocs: {}",
        stats.alloc_count / iterations as u64
    );
    println!(
        "  Per-iteration bytes:  {}",
        stats.alloc_bytes / iterations as u64
    );
    println!("  Peak live bytes:      {}", stats.peak_live_bytes);

    if ai_output {
        println!();
        println!("  Collecting dhat call stacks for --ai...");
        println!("  View results at: https://nnethercote.github.io/dh_view/dh_view.html");
        run_dhat_callsite_summary(input, op, op_args);
    }
    println!();
}

fn run_counting_alloc_in_process(
    input: &Path,
    op: &str,
    op_args: &[String],
    iterations: usize,
) -> AllocStats {
    let _session = CountingSession::start();
    bench::run_viprs_alloc_only(input, op, op_args, iterations);
    let stats = counting_alloc::snapshot();
    drop(_session);
    stats
}

fn run_dhat_callsite_summary(input: &Path, op: &str, op_args: &[String]) {
    let repo = repo_root();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&repo);
    cmd.args(["run", "--example", "dhat_profile", "--release", "--"]);
    cmd.arg(input);
    cmd.arg(op);
    for arg in op_args {
        cmd.arg(arg);
    }

    let _ = std::fs::remove_file(repo.join("dhat-heap.json"));

    match cmd.output() {
        Ok(o) if o.status.success() => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let stdout = String::from_utf8_lossy(&o.stdout);

            for line in stderr.lines().chain(stdout.lines()) {
                if line.starts_with("dhat:")
                    || line.starts_with("Running")
                    || line.starts_with("Done")
                {
                    println!("  {line}");
                }
            }

            let total_line = stderr
                .lines()
                .chain(stdout.lines())
                .find(|line| line.contains("Total:"));
            if let Some(line) = total_line {
                println!();
                println!("  Summary: {line}");
            }

            let json_path = repo.join("dhat-heap.json");
            if json_path.exists() {
                println!();
                println!("  Full profile: {}", json_path.display());
                println!("  View at: https://nnethercote.github.io/dh_view/dh_view.html");

                if let Ok(content) = std::fs::read_to_string(&json_path) {
                    if content.len() > 100 {
                        println!("  Profile size: {} KB", content.len() / 1024);
                    }
                }

                match load_dhat_profile(&json_path) {
                    Ok(profile) => {
                        println!();
                        print!("{}", render_ai_summary(&profile));
                    }
                    Err(err) => {
                        eprintln!("  Failed to render --ai allocation summary: {err}");
                    }
                }
            }
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!("  dhat_profile example failed:\n{stderr}");
        }
        Err(e) => {
            eprintln!("  Failed to run dhat_profile: {e}");
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DhatProfile {
    #[serde(default)]
    pps: Vec<ProgramPoint>,
    #[serde(default)]
    ftbl: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ProgramPoint {
    #[serde(default)]
    tb: u64,
    #[serde(default)]
    tbk: u64,
    #[serde(default)]
    mb: u64,
    #[serde(default)]
    gb: u64,
    #[serde(default)]
    fs: Vec<usize>,
}

fn load_dhat_profile(json_path: &Path) -> Result<DhatProfile, String> {
    let content = std::fs::read_to_string(json_path)
        .map_err(|err| format!("read {}: {err}", json_path.display()))?;
    serde_json::from_str(&content).map_err(|err| format!("parse {}: {err}", json_path.display()))
}

fn render_ai_summary(profile: &DhatProfile) -> String {
    let mut pps: Vec<&ProgramPoint> = profile.pps.iter().collect();
    pps.sort_by(|left, right| right.tb.cmp(&left.tb));

    let total_bytes: u64 = profile.pps.iter().map(|pp| pp.tb).sum();
    let total_allocs: u64 = profile.pps.iter().map(|pp| pp.tbk).sum();
    let peak_live_bytes: u64 = profile.pps.iter().map(|pp| pp.gb).sum();

    let mut out = String::new();
    out.push_str("--- allocation sites (top 15 by total bytes) ---\n");
    out.push_str("  #  bytes      allocs  max_live  caller\n");

    for (index, pp) in pps.into_iter().take(AI_ALLOC_SITE_LIMIT).enumerate() {
        let frames = resolve_callers(pp, &profile.ftbl);
        let caller = frames.first().map(String::as_str).unwrap_or("(unknown)");
        out.push_str(&format!(
            "  {:>2}  {:>10}  {:>6}  {:>8}  {caller}\n",
            index + 1,
            pp.tb,
            pp.tbk,
            pp.mb
        ));

        for frame in frames.iter().skip(1) {
            out.push_str(&format!("{CALLER_CONTINUATION_PREFIX}→ {frame}\n"));
        }
    }

    out.push_str("--- totals ---\n");
    out.push_str(&format!("  total_bytes: {total_bytes}\n"));
    out.push_str(&format!("  total_allocs: {total_allocs}\n"));
    out.push_str(&format!("  peak_live_bytes: {peak_live_bytes}\n"));
    out
}

fn resolve_callers(pp: &ProgramPoint, ftbl: &[String]) -> Vec<String> {
    pp.fs
        .iter()
        .filter_map(|&index| ftbl.get(index))
        .filter(|frame| !should_skip_frame(frame))
        .take(AI_CALLER_DEPTH)
        .cloned()
        .collect()
}

fn should_skip_frame(frame: &str) -> bool {
    let lower = frame.to_ascii_lowercase();
    lower.contains("below main")
        || frame.contains("alloc::alloc::")
        || frame.contains("__rust_alloc")
}

#[cfg(test)]
mod tests {
    use super::{
        DhatProfile, ProgramPoint, render_ai_summary, resolve_callers,
        run_counting_alloc_in_process, should_skip_frame,
    };
    use crate::common::resolve_input;

    #[test]
    fn resolve_callers_filters_allocator_frames_and_limits_depth() {
        let profile = DhatProfile {
            pps: vec![ProgramPoint {
                tb: 4_194_304,
                tbk: 1,
                mb: 4_194_304,
                gb: 4_194_304,
                fs: vec![0, 1, 2, 3, 4],
            }],
            ftbl: vec![
                "alloc::alloc::alloc".to_string(),
                "viprs::adapters::scheduler::rayon_scheduler::allocate_buffer".to_string(),
                "rayon_core::scope::scope".to_string(),
                "viprs::adapters::scheduler::rayon_scheduler::run_concurrent".to_string(),
                "below main".to_string(),
            ],
        };

        let callers = resolve_callers(&profile.pps[0], &profile.ftbl);

        assert_eq!(
            callers,
            vec![
                "viprs::adapters::scheduler::rayon_scheduler::allocate_buffer",
                "rayon_core::scope::scope",
                "viprs::adapters::scheduler::rayon_scheduler::run_concurrent",
            ]
        );
    }

    #[test]
    fn render_ai_summary_sorts_by_total_bytes_and_prints_totals() {
        let profile = DhatProfile {
            pps: vec![
                ProgramPoint {
                    tb: 128,
                    tbk: 2,
                    mb: 64,
                    gb: 32,
                    fs: vec![0],
                },
                ProgramPoint {
                    tb: 1024,
                    tbk: 4,
                    mb: 256,
                    gb: 96,
                    fs: vec![1, 2],
                },
            ],
            ftbl: vec![
                "viprs::small_alloc".to_string(),
                "viprs::big_alloc".to_string(),
                "caller::frame".to_string(),
            ],
        };

        let summary = render_ai_summary(&profile);

        assert!(summary.contains("--- allocation sites (top 15 by total bytes) ---"));
        assert!(summary.contains("  1        1024       4       256  viprs::big_alloc"));
        assert!(summary.contains("→ caller::frame"));
        assert!(summary.contains("  total_bytes: 1152"));
        assert!(summary.contains("  total_allocs: 6"));
        assert!(summary.contains("  peak_live_bytes: 128"));
    }

    #[test]
    fn should_skip_frame_matches_known_noise() {
        assert!(should_skip_frame("below main"));
        assert!(should_skip_frame("alloc::alloc::exchange_malloc"));
        assert!(should_skip_frame("__rust_alloc"));
        assert!(!should_skip_frame("viprs::domain::ops::resize::run"));
    }

    #[test]
    fn in_process_counting_reports_allocations_for_invert() {
        let input = resolve_input("tests/fixtures/images/bench_512x512.jpg");
        let stats = run_counting_alloc_in_process(&input, "invert", &[], 1);

        assert!(stats.alloc_count > 0);
        assert!(stats.alloc_bytes > 0);
    }
}
