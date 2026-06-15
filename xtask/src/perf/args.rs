pub const DEFAULT_ITERATIONS: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Arm64,
    Amd64,
}

impl Arch {
    pub fn docker_platform(&self) -> &'static str {
        match self {
            Arch::Arm64 => "linux/arm64",
            Arch::Amd64 => "linux/amd64",
        }
    }

    pub fn image_tag(&self) -> &'static str {
        match self {
            Arch::Arm64 => "viprs-perf:arm64",
            Arch::Amd64 => "viprs-perf:amd64",
        }
    }

    pub fn rust_target_triple(&self) -> &'static str {
        match self {
            Arch::Arm64 => "aarch64-apple-darwin",
            Arch::Amd64 => "x86_64-unknown-linux-gnu",
        }
    }

    pub fn is_native(&self) -> bool {
        match self {
            Arch::Arm64 => cfg!(target_arch = "aarch64"),
            Arch::Amd64 => cfg!(target_arch = "x86_64"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metrics {
    All,
    Hw,
    Alloc,
    Simd,
}

pub struct PerfArgs {
    pub input: String,
    pub op: String,
    pub op_args: Vec<String>,
    pub iterations: usize,
    pub metrics: Metrics,
    pub arch: Arch,
    pub ai_output: bool,
}

pub fn parse_args(args: &[String]) -> PerfArgs {
    let help = args
        .first()
        .map(|a| a == "--help" || a == "-h")
        .unwrap_or(false);
    if args.len() < 2 || help {
        eprintln!(
            "cargo xtask perf — low-level hard metrics: SIMD%, allocations, aggregate cache counters"
        );
        eprintln!();
        eprintln!("USAGE:");
        eprintln!("  cargo xtask perf <input> <op> [op_args...]");
        eprintln!("                   [--metrics simd|alloc|hw|all]");
        eprintln!("                   [--arch arm64|amd64]");
        eprintln!("                   [--iterations N]");
        eprintln!("                   [--ai]");
        eprintln!();
        eprintln!("METRICS:");
        eprintln!("  --metrics simd    Static disassembly: ratio of SIMD datapath instructions");
        eprintln!(
            "                    vs scalar datapath instructions in op-matched viprs symbols."
        );
        eprintln!("                    No Docker, instant. Use to verify that a SIMD fix actually");
        eprintln!("                    produced vector instructions in the selected operation.");
        eprintln!(
            "                    Red flag: < 10% on a numeric op → scalar fallback guard exists."
        );
        eprintln!();
        eprintln!("  --metrics alloc   Heap allocation count per iteration via the in-process");
        eprintln!("                    counting allocator. No Docker.");
        eprintln!("                    Any allocation inside process_region is a bug.");
        eprintln!(
            "                    Red flag: per_iter_allocs > 0, or allocs grow with image size."
        );
        eprintln!();
        eprintln!("  --metrics hw      Aggregate cachegrind + DHAT + perf stat on BOTH viprs and");
        eprintln!("                    libvips inside Docker. Gives total D1mr/DLmr/IPC numbers.");
        eprintln!(
            "                    For per-function breakdown use: cargo xtask profile --tool cachegrind"
        );
        eprintln!();
        eprintln!("  --metrics all     (default) Runs simd + alloc + hw in sequence.");
        eprintln!();
        eprintln!("OPTIONS:");
        eprintln!("  --arch arm64|amd64   Target arch for Docker (default: native)");
        eprintln!("  --iterations N       Iterations (default: 20)");
        eprintln!("  --ai                 For --metrics alloc, also run dhat and print the top");
        eprintln!(
            "                       allocation sites as structured text for AI/CLI consumption"
        );
        eprintln!("  --help, -h           Show this help");
        eprintln!();
        eprintln!("EXAMPLES:");
        eprintln!("  # Did the NEON fix actually vectorize?");
        eprintln!(
            "  cargo xtask perf tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 --metrics simd"
        );
        eprintln!();
        eprintln!("  # Are there heap allocations in the pixel path?");
        eprintln!(
            "  cargo xtask perf tests/fixtures/images/bench_2048x2048.jpg resize 0.5 --metrics alloc"
        );
        eprintln!();
        eprintln!("  # Print top allocation call sites as structured text:");
        eprintln!(
            "  cargo xtask perf tests/fixtures/images/bench_2048x2048.jpg resize 0.5 --metrics alloc --ai"
        );
        eprintln!();
        eprintln!("  # Aggregate cache counters for both binaries (Docker)");
        eprintln!(
            "  cargo xtask perf tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 --metrics hw"
        );
        std::process::exit(if help { 0 } else { 1 });
    }

    let input = args[0].clone();
    let op = args[1].clone();
    let mut op_args = Vec::new();
    let mut iterations = DEFAULT_ITERATIONS;
    let mut metrics = Metrics::All;
    let mut ai_output = false;
    let mut arch = if cfg!(target_arch = "aarch64") {
        Arch::Arm64
    } else {
        Arch::Amd64
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" if i + 1 < args.len() => {
                iterations = args[i + 1].parse().unwrap_or(DEFAULT_ITERATIONS);
                i += 2;
            }
            "--metrics" if i + 1 < args.len() => {
                metrics = match args[i + 1].as_str() {
                    "hw" => Metrics::Hw,
                    "alloc" => Metrics::Alloc,
                    "simd" => Metrics::Simd,
                    _ => Metrics::All,
                };
                i += 2;
            }
            "--arch" if i + 1 < args.len() => {
                arch = match args[i + 1].as_str() {
                    "amd64" | "x86_64" | "x86" => Arch::Amd64,
                    _ => Arch::Arm64,
                };
                i += 2;
            }
            "--ai" => {
                ai_output = true;
                i += 1;
            }
            _ => {
                op_args.push(args[i].clone());
                i += 1;
            }
        }
    }

    PerfArgs {
        input,
        op,
        op_args,
        iterations,
        metrics,
        arch,
        ai_output,
    }
}

#[cfg(test)]
mod tests {
    use super::{Arch, DEFAULT_ITERATIONS, Metrics, parse_args};

    #[test]
    fn parse_args_enables_ai_output() {
        let args = vec![
            "input.jpg".to_string(),
            "resize".to_string(),
            "0.5".to_string(),
            "--metrics".to_string(),
            "alloc".to_string(),
            "--ai".to_string(),
        ];

        let parsed = parse_args(&args);

        assert_eq!(parsed.input, "input.jpg");
        assert_eq!(parsed.op, "resize");
        assert_eq!(parsed.op_args, vec!["0.5"]);
        assert_eq!(parsed.metrics, Metrics::Alloc);
        assert!(parsed.ai_output);
    }

    #[test]
    fn parse_args_keeps_defaults_without_ai_flag() {
        let args = vec!["input.jpg".to_string(), "invert".to_string()];

        let parsed = parse_args(&args);

        assert_eq!(parsed.metrics, Metrics::All);
        assert_eq!(parsed.iterations, DEFAULT_ITERATIONS);
        assert!(!parsed.ai_output);
        assert!(matches!(parsed.arch, Arch::Arm64 | Arch::Amd64));
    }
}
