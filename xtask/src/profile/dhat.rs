//! dhat-based heap allocation profiling for viprs and libvips.
//!
//! Runs both binaries under valgrind --tool=dhat inside Docker, saves two
//! JSON profiles openable at https://nnethercote.github.io/dh_view/dh_view.html.
//!
//! Any allocation whose call stack passes through process_region or anything
//! under src/domain/ops/ is a bug (violates the zero-alloc rule).

use std::path::Path;
use std::process::Command;

use crate::docker;
use crate::perf::args::Arch;
use crate::profile::baseline_op_name;

pub fn run_dhat(input: &Path, op: &str, op_args: &[String], iterations: usize, arch: Arch) {
    println!("--- Heap allocation profiling via dhat (Docker) ---");
    println!("  Target: {}", arch.docker_platform());
    println!("  Profiles both viprs and libvips with call stacks per allocation site.");
    println!();
    println!("  Rule: any allocation whose stack contains process_region or ops/ is a bug.");
    println!("  Rule: per-iteration alloc count must be 0 for any hot pixel path.");
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

    // Use the dedicated dhat profile script inside the container
    let mut cmd = Command::new("docker");
    cmd.args(["run", "--rm", "--privileged"]);
    cmd.args(["-v", &format!("{}:/data/input", input_dir.display())]);
    // Mount a writable output dir so JSON files survive container exit
    let out_dir = std::env::temp_dir();
    cmd.args(["-v", &format!("{}:/data/output", out_dir.display())]);
    cmd.args(["--entrypoint", "/opt/bench/dhat_profile.sh"]);
    cmd.arg(arch.image_tag());
    cmd.arg(format!("/data/input/{}", input_filename));
    cmd.arg(baseline_op_name(op));
    for a in op_args {
        cmd.arg(a.as_str());
    }
    cmd.args(["--iterations", &iterations.to_string()]);

    println!("  Running dhat inside {} ...", arch.image_tag());
    println!("  Output will be saved to /tmp/viprs_dhat_{op}.json and /tmp/libvips_dhat_{op}.json");
    println!();

    match cmd.output() {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            println!("{stdout}");
            println!();
            println!("  === How to read dhat profiles ===");
            println!();
            println!("  1. Open https://nnethercote.github.io/dh_view/dh_view.html");
            println!("  2. Load /tmp/viprs_dhat_{op}.json → inspect viprs allocation call stacks");
            println!("  3. Reload the page, load /tmp/libvips_dhat_{op}.json for libvips");
            println!();
            println!("  What to look for in viprs:");
            println!(
                "    - Any frame containing 'process_region' or 'domain/ops' → zero-alloc violation"
            );
            println!(
                "    - Allocations that grow linearly with image size → per-tile Vec somewhere"
            );
            println!(
                "    - Alloc sites absent in libvips but present in viprs → unnecessary copies"
            );
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
