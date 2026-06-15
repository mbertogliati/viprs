//! Shared Docker helpers for perf and profile commands.

use std::path::Path;
use std::process::Command;

use crate::perf::args::Arch;

/// Detect Docker runtime and auto-start colima if needed.
/// Returns true if Docker is ready, false if we cannot proceed.
pub fn ensure_docker_running(arch: Arch) -> bool {
    if docker_is_ready() {
        println!("  ✓ Docker is running");
        return true;
    }

    println!("  Docker not responding. Attempting auto-start...");

    let has_colima = Command::new("which")
        .arg("colima")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let has_docker_desktop = Path::new("/Applications/Docker.app").exists();
    let has_orbstack = Command::new("which")
        .arg("orb")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if has_colima {
        println!("  Found colima. Starting...");
        let colima_arch = if arch == Arch::Arm64 {
            "aarch64"
        } else {
            "x86_64"
        };
        let status = Command::new("colima")
            .args([
                "start",
                "--arch",
                colima_arch,
                "--cpu",
                "4",
                "--memory",
                "4",
            ])
            .status();
        match status {
            Ok(s) if s.success() => {
                println!("  ✓ colima started");
                std::thread::sleep(std::time::Duration::from_secs(2));
                if docker_is_ready() {
                    return true;
                }
                eprintln!("  colima started but Docker still not responding.");
                return false;
            }
            Ok(_) => {
                eprintln!("  colima start failed. Try manually:");
                eprintln!("    colima start --arch {colima_arch} --cpu 4 --memory 4");
                return false;
            }
            Err(e) => {
                eprintln!("  Failed to run colima: {e}");
                return false;
            }
        }
    } else if has_orbstack {
        println!("  Found OrbStack. Starting...");
        let status = Command::new("orb").args(["start"]).status();
        if status.map(|s| s.success()).unwrap_or(false) {
            std::thread::sleep(std::time::Duration::from_secs(2));
            if docker_is_ready() {
                println!("  ✓ OrbStack started");
                return true;
            }
        }
        eprintln!("  OrbStack start failed.");
        return false;
    } else if has_docker_desktop {
        println!("  Found Docker Desktop. Starting...");
        let _ = Command::new("open").args(["-a", "Docker"]).status();
        println!("  Waiting for Docker Desktop to initialize (up to 30s)...");
        for i in 0..15 {
            std::thread::sleep(std::time::Duration::from_secs(2));
            if docker_is_ready() {
                println!("  ✓ Docker Desktop ready (took ~{}s)", (i + 1) * 2);
                return true;
            }
        }
        eprintln!("  Docker Desktop did not start in time.");
        return false;
    }

    eprintln!("  No Docker runtime found. Install one of:");
    eprintln!("    brew install colima docker     (recommended, lightweight)");
    eprintln!("    brew install --cask orbstack   (fast alternative)");
    eprintln!("    brew install --cask docker     (Docker Desktop)");
    false
}

/// Ensure the profiling Docker image is built. Auto-builds if missing.
/// Returns true if the image is ready.
pub fn ensure_image_built(arch: Arch) -> bool {
    let exists = Command::new("docker")
        .args(["image", "inspect", arch.image_tag()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if exists {
        println!("  ✓ Image '{}' exists", arch.image_tag());
        return true;
    }

    println!(
        "  Image '{}' not found. Building (this takes ~2-3 min first time)...",
        arch.image_tag()
    );
    println!();

    let repo = crate::common::repo_root();
    let dockerfile = repo.join("docker/Dockerfile");

    if !dockerfile.exists() {
        eprintln!("  docker/Dockerfile not found at: {}", dockerfile.display());
        return false;
    }

    let status = Command::new("docker")
        .current_dir(&repo)
        .args([
            "buildx",
            "build",
            "--platform",
            arch.docker_platform(),
            "-t",
            arch.image_tag(),
            "-f",
            "docker/Dockerfile",
            "--load",
            ".",
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("  ✓ Image '{}' built successfully", arch.image_tag());
            true
        }
        Ok(_) => {
            eprintln!("  Image build failed. Try manually:");
            eprintln!(
                "    docker buildx build --platform {} -t {} -f docker/Dockerfile .",
                arch.docker_platform(),
                arch.image_tag()
            );
            false
        }
        Err(e) => {
            eprintln!("  Failed to run docker buildx: {e}");
            false
        }
    }
}

fn docker_is_ready() -> bool {
    Command::new("docker")
        .args(["info", "--format", "{{.ServerVersion}}"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
