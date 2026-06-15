//! cachegrind-based cache-miss profiling for viprs and libvips.
//!
//! Runs both binaries under valgrind --tool=cachegrind inside a Docker container
//! (for cross-arch support and hermetic results), then calls cg_annotate on each
//! output and prints a side-by-side table sorted by LL data cache misses (DLmr).
//!
//! Docker is required. On macOS, colima/OrbStack/Docker Desktop are auto-detected.

use std::path::Path;
use std::process::Command;

use crate::docker;
use crate::perf::args::Arch;
use crate::profile::baseline_op_name;

pub fn run_cachegrind(input: &Path, op: &str, op_args: &[String], iterations: usize, arch: Arch) {
    println!("--- Cache-miss analysis via cachegrind (Docker) ---");
    println!("  Target: {}", arch.docker_platform());
    println!("  Profiles viprs and libvips, then shows per-function comparison.");
    println!();

    if !docker::ensure_docker_running(arch) {
        return;
    }
    if !docker::ensure_image_built(arch) {
        return;
    }

    let input_abs = if input.is_absolute() {
        input.to_path_buf()
    } else {
        crate::common::repo_root().join(input)
    };
    let input_dir = input_abs.parent().unwrap();
    let input_filename = input_abs.file_name().unwrap().to_string_lossy();

    let mut cmd = Command::new("docker");
    cmd.args(["run", "--rm", "--privileged"]);
    cmd.args(["-v", &format!("{}:/data/input", input_dir.display())]);
    // Override entrypoint to use the dedicated profile script
    cmd.args(["--entrypoint", "/opt/bench/profile.sh"]);
    cmd.arg(arch.image_tag());
    cmd.arg(format!("/data/input/{}", input_filename));
    cmd.arg(baseline_op_name(op));
    for a in op_args {
        cmd.arg(a.as_str());
    }
    cmd.args(["--iterations", &iterations.to_string()]);

    println!(
        "  Running cachegrind on both binaries inside {} ...",
        arch.image_tag()
    );
    println!("  (This is 20-50x slower than normal due to simulation — expected)");
    println!();

    match cmd.output() {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            println!("{stdout}");
        }
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let stderr = String::from_utf8_lossy(&o.stderr);
            if !stdout.is_empty() {
                println!("{stdout}");
            }
            if !stderr.is_empty() {
                eprintln!("  stderr: {stderr}");
            }
        }
        Err(e) => eprintln!("  Docker run failed: {e}"),
    }
}
