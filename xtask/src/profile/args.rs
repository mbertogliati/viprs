use crate::perf::args::Arch;

pub const DEFAULT_ITERATIONS: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    /// samply — CPU flame graph, works locally on macOS and Linux.
    Samply,
    /// cachegrind — deterministic cache-miss analysis per function, requires Docker.
    Cachegrind,
    /// dhat — heap allocation call stacks per function for both binaries, requires Docker.
    Dhat,
}

pub struct ProfileArgs {
    pub input: String,
    pub op: String,
    pub op_args: Vec<String>,
    pub iterations: usize,
    pub tool: Tool,
    pub arch: Arch,
    pub ai_output: bool,
}

fn normalize_op_alias(op: &str) -> &str {
    match op {
        "load_exr" => "load-exr",
        "extract_area" => "extract-area",
        other => other,
    }
}

pub fn parse_args(args: &[String]) -> ProfileArgs {
    let help = args
        .first()
        .map(|a| a == "--help" || a == "-h")
        .unwrap_or(false);
    if args.len() < 2 || help {
        eprintln!("cargo xtask profile — side-by-side CPU and cache profiling: viprs vs libvips");
        eprintln!();
        eprintln!("USAGE:");
        eprintln!("  cargo xtask profile <input> <op> [op_args...]");
        eprintln!("                      [--tool samply|cachegrind|dhat]");
        eprintln!("                      [--arch arm64|amd64]");
        eprintln!("                      [--iterations N]");
        eprintln!("                      [--ai]");
        eprintln!();
        eprintln!("TOOLS:");
        eprintln!("  --tool samply        (default) CPU flame graph. Works locally on macOS and");
        eprintln!("                       Linux with no Docker required. Saves two JSON files");
        eprintln!("                       (./tmp/viprs_profile_<op>.json and");
        eprintln!("                        ./tmp/libvips_profile_<op>.json)");
        eprintln!("                       openable at https://profiler.firefox.com.");
        eprintln!("                       Install: cargo install samply");
        eprintln!();
        eprintln!("  --tool cachegrind    Per-function L1/LL cache-miss table via Valgrind.");
        eprintln!(
            "                       Requires Docker (auto-started if colima/OrbStack found)."
        );
        eprintln!("                       20-50x slower than real execution — deterministic.");
        eprintln!("                       Shows a ranked table sorted by DLmr (LL data misses):");
        eprintln!("                         Function           libvips DLmr  viprs DLmr  ratio");
        eprintln!("                         reduce_h_u8               210       18000    85.7x");
        eprintln!(
            "                       A high ratio → wrong stride, bad tile size, layout issue."
        );
        eprintln!();
        eprintln!(
            "  --tool dhat          Per-function heap allocation call stacks for both binaries."
        );
        eprintln!("                       Requires Docker. Saves two JSON files openable at");
        eprintln!("                       https://nnethercote.github.io/dh_view/dh_view.html.");
        eprintln!("                       Any allocation in process_region is a bug.");
        eprintln!(
            "                       Profiles libvips (valgrind dhat) and viprs (dhat crate)."
        );
        eprintln!();
        eprintln!("OPTIONS:");
        eprintln!("  --iterations N       Iterations to run (default: 20; cachegrind uses 5)");
        eprintln!("  --arch arm64|amd64   Target arch for Docker/cachegrind (default: native)");
        eprintln!(
            "  --ai                 With --tool samply, print a machine-readable top-functions"
        );
        eprintln!("                       summary from the generated Firefox Profiler JSON.");
        eprintln!("  --help, -h           Show this help");
        eprintln!();
        eprintln!("EXAMPLES:");
        eprintln!("  # Flame graph — find the hottest function in the pixel path");
        eprintln!("  cargo xtask profile tests/fixtures/images/bench_2048x2048.jpg thumbnail 400");
        eprintln!();
        eprintln!("  # Cache-miss table — find the function trashing the cache");
        eprintln!(
            "  cargo xtask profile tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 \\"
        );
        eprintln!("    --tool cachegrind");
        eprintln!();
        eprintln!("  # Same but for x86 via Docker");
        eprintln!("  cargo xtask profile tests/fixtures/images/bench_2048x2048.jpg invert \\");
        eprintln!("    --tool cachegrind --arch amd64");
        eprintln!();
        eprintln!("  # Allocation call stacks — find what's allocating in the pixel path");
        eprintln!(
            "  cargo xtask profile tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 \\"
        );
        eprintln!("    --tool dhat");
        eprintln!();
        eprintln!("  grep -r 'reduce_h' .libvips_repo/libvips/resample/");
        std::process::exit(if help { 0 } else { 1 });
    }

    let input = args[0].clone();
    let op = normalize_op_alias(&args[1]).to_owned();
    let mut op_args = Vec::new();
    let mut iterations = DEFAULT_ITERATIONS;
    let mut tool = Tool::Samply;
    let mut arch = if cfg!(target_arch = "aarch64") {
        Arch::Arm64
    } else {
        Arch::Amd64
    };
    let mut ai_output = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" if i + 1 < args.len() => {
                iterations = args[i + 1].parse().unwrap_or(DEFAULT_ITERATIONS);
                i += 2;
            }
            "--tool" if i + 1 < args.len() => {
                tool = match args[i + 1].as_str() {
                    "cachegrind" | "cg" => Tool::Cachegrind,
                    "dhat" => Tool::Dhat,
                    _ => Tool::Samply,
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

    ProfileArgs {
        input,
        op,
        op_args,
        iterations,
        tool,
        arch,
        ai_output,
    }
}

#[cfg(test)]
mod tests {
    use super::{Tool, parse_args};

    #[test]
    fn parse_args_enables_ai_output_flag() {
        let args = vec![
            "input.jpg".to_owned(),
            "thumbnail".to_owned(),
            "400".to_owned(),
            "--ai".to_owned(),
        ];

        let parsed = parse_args(&args);

        assert!(parsed.ai_output);
        assert_eq!(parsed.tool, Tool::Samply);
        assert_eq!(parsed.op_args, vec!["400".to_owned()]);
    }

    #[test]
    fn parse_args_normalizes_load_exr_alias() {
        let args = vec!["input.exr".to_owned(), "load_exr".to_owned()];

        let parsed = parse_args(&args);

        assert_eq!(parsed.op, "load-exr");
    }
}
