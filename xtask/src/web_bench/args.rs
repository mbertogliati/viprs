/// Command-line arguments for `cargo xtask web-bench`.

pub struct WebBenchArgs {
    pub input: String,
    pub scenario: Scenario,
    pub iterations: u32,
    pub concurrency: Vec<u32>,
    pub json_output: bool,
    #[allow(dead_code)]
    pub threads: Option<usize>,
}

#[derive(Clone, Debug)]
pub enum Scenario {
    ThumbnailBytes,
    PipelineBytes,
    Concurrent,
    LargeUpload,
    All,
}

pub fn parse_args(args: &[String]) -> WebBenchArgs {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(if args.is_empty() { 1 } else { 0 });
    }

    let mut input = String::new();
    let mut scenario = Scenario::All;
    let mut iterations = 30u32;
    let mut concurrency = vec![2, 4, 8, 16];
    let mut json_output = false;
    let mut threads: Option<usize> = None;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--iterations" | "-n" => {
                i += 1;
                iterations = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("--iterations requires a numeric argument");
                    std::process::exit(1);
                });
            }
            "--concurrency" => {
                i += 1;
                concurrency = args
                    .get(i)
                    .map(|s| s.split(',').filter_map(|n| n.parse().ok()).collect())
                    .unwrap_or_else(|| {
                        eprintln!("--concurrency requires comma-separated numbers (e.g. 2,4,8)");
                        std::process::exit(1);
                    });
            }
            "--json" => json_output = true,
            "--threads" => {
                i += 1;
                threads = args.get(i).and_then(|s| s.parse().ok());
            }
            "--scenario" | "-s" => {
                i += 1;
                scenario = match args.get(i).map(|s| s.as_str()) {
                    Some("thumbnail-bytes") => Scenario::ThumbnailBytes,
                    Some("pipeline-bytes") => Scenario::PipelineBytes,
                    Some("concurrent") => Scenario::Concurrent,
                    Some("large-upload") => Scenario::LargeUpload,
                    Some("all") => Scenario::All,
                    Some(other) => {
                        eprintln!("Unknown scenario: {other}");
                        eprintln!(
                            "Valid: thumbnail-bytes, pipeline-bytes, concurrent, large-upload, all"
                        );
                        std::process::exit(1);
                    }
                    None => {
                        eprintln!("--scenario requires an argument");
                        std::process::exit(1);
                    }
                };
            }
            arg if !arg.starts_with('-') && input.is_empty() => {
                input = arg.to_string();
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if input.is_empty() {
        eprintln!("Error: input image path required");
        print_help();
        std::process::exit(1);
    }

    WebBenchArgs {
        input,
        scenario,
        iterations,
        concurrency,
        json_output,
        threads,
    }
}

fn print_help() {
    eprintln!("Web-service benchmark: simulates HTTP image processing workloads");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  cargo xtask web-bench <input_image> [options]");
    eprintln!();
    eprintln!("SCENARIOS:");
    eprintln!("  thumbnail-bytes   JPEG/PNG/WebP bytes → thumbnail(400) → encode WebP → bytes");
    eprintln!("  pipeline-bytes    bytes → thumbnail(800) + sharpen + linear(1.1,5) → JPEG q85");
    eprintln!("  concurrent        N parallel requests of thumbnail-bytes (N=2,4,8,16)");
    eprintln!("  large-upload      8192×8192 image → thumbnail(400) (simulates large upload)");
    eprintln!("  all               Run all scenarios (default)");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  -s, --scenario <name>       Run a specific scenario (default: all)");
    eprintln!("  -n, --iterations <N>        Iterations per scenario (default: 30)");
    eprintln!("  --concurrency <N,N,...>      Concurrency levels (default: 2,4,8,16)");
    eprintln!("  --threads <N>               Pin thread pool size");
    eprintln!("  --json                      Output results as JSON");
    eprintln!();
    eprintln!("EXAMPLE:");
    eprintln!("  cargo xtask web-bench tests/fixtures/images/sample.jpg");
    eprintln!("  cargo xtask web-bench sample.jpg -s concurrent --concurrency 4,8,16 -n 50");
}
