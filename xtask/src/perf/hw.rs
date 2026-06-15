use std::path::Path;
use std::process::Command;

use super::args::Arch;
use crate::common::repo_root;

pub fn run_hw_counters(input: &Path, op: &str, op_args: &[String], iterations: usize, arch: Arch) {
    println!("--- Hardware counters via Docker (aggregate totals) ---");
    println!("  Target: {}", arch.docker_platform());
    println!("  Backend: cachegrind + DHAT + perf stat (if PMU available)");
    println!();
    println!("  Tip: for per-function cache-miss breakdown use:");
    println!("    cargo xtask profile <input> {op} --tool cachegrind");
    println!();

    if !crate::docker::ensure_docker_running(arch) {
        return;
    }
    if !crate::docker::ensure_image_built(arch) {
        return;
    }

    let input_abs = if input.is_absolute() {
        input.to_path_buf()
    } else {
        repo_root().join(input)
    };
    let input_dir = input_abs.parent().unwrap();
    let input_filename = input_abs.file_name().unwrap().to_string_lossy();

    let mut cmd = Command::new("docker");
    cmd.args(["run", "--rm", "--privileged"]);
    cmd.args(["-v", &format!("{}:/data/input", input_dir.display())]);
    cmd.arg(arch.image_tag());
    cmd.arg(format!("/data/input/{}", input_filename));
    cmd.arg(op);
    for a in op_args {
        cmd.arg(a);
    }
    cmd.args(["--iterations", &iterations.to_string()]);
    cmd.args(["--metrics", "all"]);

    println!(
        "  Running: docker run --privileged {} ...",
        arch.image_tag()
    );
    println!("  (runs cachegrind + DHAT on BOTH viprs and libvips — aggregate totals)");
    println!();

    match cmd.output() {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            println!("{stdout}");
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let stdout = String::from_utf8_lossy(&o.stdout);
            if !stdout.is_empty() {
                println!("{stdout}");
            }
            eprintln!("  Container stderr: {stderr}");
        }
        Err(e) => eprintln!("  Docker run failed: {e}"),
    }
}
