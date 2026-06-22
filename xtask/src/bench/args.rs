use std::path::PathBuf;

use super::helpers::{DEFAULT_ITERATIONS, DEFAULT_VIPRS_CACHE_BYTES, is_workflow_like_op};

pub struct BenchArgs {
    pub input: String,
    pub op: String,
    pub op_args: Vec<String>,
    pub iterations: usize,
    pub threads: usize,
    /// When true, run across the default fixture matrix and print a summary table.
    pub multi_sizes: bool,
    /// Run a named fixture matrix (for example compute-baselines) and print a summary table.
    pub scenario_set: Option<String>,
    /// When true, print machine-readable JSON to stdout instead of the human summary.
    pub json_output: bool,
    /// When true, compare current ratios against a JSON baseline and fail on regressions >10%.
    pub check_regression: bool,
    /// Baseline JSON file produced by `cargo xtask bench --json`.
    pub baseline: Option<PathBuf>,
    /// Include full decode-from-disk in every iteration. Default: true. Opt-out: --no-e2e.
    pub e2e: bool,
    /// When true, print viprs-only per-stage timings for thumbnail pipelines.
    pub profile_stages: bool,
    /// When true, print stage-profile JSON for AI consumers instead of the text table.
    pub ai_output: bool,
    /// When true, preload once and run only the viprs hot loop.
    pub profile_only: bool,
    /// When true, expand the operation into the canonical parameter matrix.
    pub parameter_matrix: bool,
}

fn normalize_operation_alias(op: &str) -> &str {
    match op {
        "extract_area" => "extract-area",
        "load_avif" => "load-avif",
        "load_jpeg" => "load-jpeg",
        "load_tiff" => "load-tiff",
        "load_exr" => "load-exr",
        "load_heif" => "load-heif",
        "load_pdf" => "load-pdf",
        "load_svg" => "load-svg",
        _ => op,
    }
}

fn thumbnail_requires_explicit_width(op: &str, parameter_matrix: bool, op_args: &[String]) -> bool {
    op == "thumbnail"
        && !parameter_matrix
        && op_args
            .first()
            .and_then(|arg| arg.parse::<u32>().ok())
            .is_none()
}

pub fn print_help() {
    let default_threads = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(4);
    let default_cache_mib = DEFAULT_VIPRS_CACHE_BYTES / (1024 * 1024);
    eprintln!(
        r#"cargo xtask bench — head-to-head latency benchmark: viprs vs libvips

USAGE:
    cargo xtask bench <input> <op> [op_args...] [options]

ARGUMENTS:
    <input>        Path to image file, or directory containing bench_NxN.jpg fixtures.
    <op>           Operation to benchmark: load | load-avif (alias: load_avif) | load-exr (alias: load_exr) | load-heif (alias: load_heif) | load-jpeg (alias: load_jpeg) | load-pdf (alias: load_pdf) | load-svg (alias: load_svg) | load-tiff (alias: load_tiff) | save-avif | save-exr | save-gif | save-heif | save-jpeg | save-jp2k | save-tiff | invert | bandmean | add | multiply | subtract | and | equal | linear | cast | flip | gamma | gauss_blur | convolve | sobel | prewitt | laplacian | median_blur | unsharp_mask | colourspace | srgb_to_lab | resize | shrink | shrinkh | shrinkv | affine | similarity | mapim | composite | thumbnail | sharpen | dilate | erode | open | close | histogram | recomb | grey | draw_line | draw_rect | draw_circle | freqfilt | workflow | invert_invert | thumbnail_sharpen | thumbnail_colourspace_cast | thumbnail_gauss_blur | thumbnail_linear | resize_colourspace | embed | extract-area (alias: extract_area) | embed_extract | three_op_chain | perceptual_enhance
    [op_args]      Operation arguments (e.g. scale for resize, hshrink [vshrink] for shrink, factor for shrinkh/shrinkv, affine matrix [a b c d], similarity [scale angle], mapim [dx dy], composite mode [over|atop], width for thumbnail [required unless --matrix], colourspace destination chain [lab|xyz|cmyk|hsv|scrgb|srgb|greyscale ...], bit depth for save-avif: u8|u16, save-tiff compression: none|lzw|deflate|packbits|jpeg, cast target [u8|u16|f32], flip direction [horizontal|vertical], gamma exponent, morphology kernel size for dilate/erode/open/close, workflow/perceptual_enhance: <target_format> [width] or --output-format <target_format> [width]). Composite multi-op scenarios use fixed parameters and do not accept extra args.

OPTIONS:
    --iterations N           Number of timed iterations per scenario  [default: 50]
    --threads N              Pin both viprs and libvips to the same worker count
                             [default: {default_threads}]
    --sizes                  Run across the default fixture matrix:
                             512x512 / 640x480 / 1920x1080 / 2048x2048 / 8192x8192
                             Op-specific exceptions apply:
                               composite -> RGBA PNG fixtures at 512 / 2048 / 8192
                               load-avif -> AVIF fixtures at 512 / 2048 / 8192
                               load-tiff -> TIFF fixtures at 512 / 2048 / 8192
                               load-exr  -> float EXR fixtures at 512 / 2048 / 8192
                               load-heif -> HEIF fixtures at 512 / 2048 / 8192
                               load-pdf  -> PDF fixtures at 512 / 2048 / 8192
                               load-svg  -> SVG fixtures at 512 / 2048 / 8192
                               save-exr  -> float EXR fixtures at 512 / 2048 / 8192
    --scenario-set NAME      Run a named fixture matrix. Supported:
                             compute-baselines | input-diversity | production-workflows
    --json                   Print structured JSON to stdout
    --check-regression       Compare current ratios against --baseline and exit non-zero if
                             any ratio regresses by more than 10%
    --baseline <path>        Baseline JSON file produced by --json
    --matrix                 Expand to the canonical parameter matrix for the selected op.
                             Supported today:
                               thumbnail  -> widths 100 / 200 / 400 / 800
                               gauss_blur -> sigma 0.5 / 1.0 / 2.0 / 5.0
                               median_blur -> kernel sizes 3 / 5
                               colourspace -> lab, lab→srgb, xyz, xyz→srgb, cmyk,
                                               cmyk→srgb, hsv, hsv→srgb, scrgb,
                                               scrgb→srgb

DEFAULTS:
    E2E mode ON              Decode from disk is included in every timed iteration.
                             This reflects real workload cost (codec + processing).
                             Opt out: --no-e2e  (pre-loads pixels to RAM before the loop)

    Internal cache ON        viprs always benchmarks with its internal tile cache enabled
                             at {default_cache_mib} MiB. Each iteration builds a fresh pipeline,
                             so no cache state is shared between iterations.
                             libvips keeps its operation cache enabled for non-load E2E ops,
                             but load/load-* benchmarks disable it to avoid cross-iteration
                             decode reuse on the same file.

OPT-OUT FLAGS:
    --no-e2e                 Pre-load image to RAM once before the loop.
                             Excludes codec decode, but still rebuilds the backend pipeline
                             from the same pre-loaded pixels on every iteration.
    --profile-stages         Print viprs-only per-stage timings for thumbnail.
                             Requires op=thumbnail and --no-e2e.
    --ai                     With --profile-stages, print machine-readable stage-profile JSON.
                             Separate from --json, which controls the main comparison output.
    --profile-only           Skip comparison/reporting and profile only the viprs hot loop.
                             Implies --no-e2e so decode/preload happens before sampling.

EXAMPLES:
    # Full productive benchmark (default — E2E + internal cache enabled):
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg invert

    # Pin both backends to 4 workers:
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg invert --threads 4

    # Average RGB/RGBA bands to a single-band image:
    cargo xtask bench tests/fixtures/images/bench_2048x2048_rgba.png bandmean

    # Arithmetic baselines with fixed scalar RHS defaults:
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg add --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg multiply --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg subtract --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg and --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg equal --no-e2e

    # Kernel-only (pre-loaded pixels, no decode):
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg invert --no-e2e

    # Multi-size table across 512x512 / 640x480 / 1920x1080 / 2048x2048 / 8192x8192:
    cargo xtask bench tests/fixtures/images invert --sizes --iterations 20

    # TIFF decode matrix across 512x512 / 2048x2048 / 8192x8192:
    cargo xtask bench tests/fixtures/images load-tiff --sizes --iterations 20

    # HEIF / AVIF decode matrices across 512x512 / 2048x2048 / 8192x8192:
    cargo xtask bench tests/fixtures/images load-heif --sizes --iterations 20
    cargo xtask bench tests/fixtures/images load-avif --sizes --iterations 20

    # SVG / PDF decode matrices across 512x512 / 2048x2048 / 8192x8192:
    cargo xtask bench tests/fixtures/images load-svg --sizes --iterations 20
    cargo xtask bench tests/fixtures/images load-pdf --sizes --iterations 20

    # Input-diversity compute matrix (grayscale / RGBA / U16 / float across 512 / 2048 / 8192):
    cargo xtask bench tests/fixtures/images invert --scenario-set compute-baselines --iterations 20
    cargo xtask bench tests/fixtures/images colourspace --scenario-set input-diversity --iterations 20

    # Save a machine-readable baseline for CI:
    cargo xtask bench tests/fixtures/images invert --sizes --json > baseline.json

    # Fail the run if any ratio regresses by >10% vs baseline:
    cargo xtask bench tests/fixtures/images invert --sizes --check-regression --baseline baseline.json

    # Thumbnail with width 800:
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg thumbnail 800

    # Canonical thumbnail matrix (100 / 200 / 400 / 800):
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg thumbnail --matrix

    # Canonical Gaussian blur sigma matrix (0.5 / 1.0 / 2.0 / 5.0):
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg gauss_blur --matrix

    # Canonical median blur matrix (3x3 / 5x5):
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg median_blur --matrix --no-e2e

    # Fixed-kernel convolution baselines:
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg convolve --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg sobel --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg prewitt --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg laplacian --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg median_blur 3 --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg unsharp_mask 0.5 3.0 --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg dilate 3 --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg erode 3 --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg open 3 --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg close 3 --no-e2e

    # Canonical colourspace conversion matrix (Lab / XYZ / CMYK / HSV / scRGB):
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg colourspace --matrix --sizes

    # Explicit colourspace route from the input interpretation:
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg colourspace hsv srgb

    # Fixed direct colour conversion baseline:
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg srgb_to_lab --no-e2e

    # Conversion baselines:
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg cast --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg flip --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg gamma --no-e2e

    # Resize to 0.5x:
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg resize 0.5

    # Fixed affine baseline (or override matrix with a b c d):
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg affine --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg affine 1.0 0.2 -0.1 0.95 --no-e2e

    # Similarity baseline (or override scale/angle):
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg similarity --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg similarity 0.9 15 --no-e2e

    # Mapim baseline with a translated coordinate grid (or override dx/dy):
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg mapim --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg mapim 0.25 0.25 --no-e2e

    # Composite baseline on RGBA input (default: over; optional: atop):
    cargo xtask bench tests/fixtures/images/bench_2048x2048_rgba.png composite --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048_rgba.png composite atop --no-e2e

    # Horizontal / vertical integer shrink by factor 5:
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg shrink 5 --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg shrink 5 3 --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg shrinkh 5 --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg shrinkv 5 --no-e2e

    # AVIF encode only (pre-load pixels once, benchmark encoder only):
    cargo xtask bench tests/fixtures/images/sample.avif save-avif --no-e2e

    # AVIF 16-bit encode baseline:
    cargo xtask bench tests/fixtures/images/sample.png save-avif u16 --no-e2e

    # GIF encode baseline:
    cargo xtask bench tests/fixtures/images/bench_512x512.gif save-gif

    # HEIF encode baseline:
    cargo xtask bench tests/fixtures/images/bench_512x512.jpg save-heif --no-e2e

    # OpenEXR encode baseline:
    cargo xtask bench tests/fixtures/images/bench_512x512.exr save-exr

    # OpenEXR decode baseline across EXR fixtures:
    cargo xtask bench tests/fixtures/images load-exr --sizes --iterations 5

    # JPEG 2000 encode baseline:
    cargo xtask bench tests/fixtures/images/bench_512x512.jpg save-jp2k --no-e2e

    # TIFF encode baseline (default: uncompressed):
    cargo xtask bench tests/fixtures/images/bench_512x512.jpg save-tiff --no-e2e

    # JPEG-in-TIFF encode baseline:
    cargo xtask bench tests/fixtures/images/bench_512x512.jpg save-tiff jpeg --no-e2e

    # Production workflow: JPEG→WebP (decode → thumbnail 400px → sharpen → encode):
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg workflow webp

    # Production workflow: JPEG→AVIF with custom width:
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg workflow avif 800

    # Perceptual enhance workflow with explicit output format:
    cargo xtask bench tests/fixtures/images/sample.jpg perceptual_enhance --output-format webp --iterations 40

    # Full production workflow matrix (JPEG→WebP, JPEG→AVIF, PNG→WebP, etc.):
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg workflow --scenario-set production-workflows --iterations 20

    # Multi-op scheduling/caching scenarios across the default fixture matrix:
    cargo xtask bench tests/fixtures/images invert_invert --sizes --iterations 20
    cargo xtask bench tests/fixtures/images thumbnail_sharpen --sizes --iterations 20
    cargo xtask bench tests/fixtures/images thumbnail_colourspace_cast --sizes --iterations 20
    cargo xtask bench tests/fixtures/images embed --sizes --iterations 20
    cargo xtask bench tests/fixtures/images extract-area --sizes --iterations 20
    cargo xtask bench tests/fixtures/images three_op_chain --sizes --iterations 20

    # Representative missing-family baselines:
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg histogram --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg recomb --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg grey --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg draw_line --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg draw_rect --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg draw_circle --no-e2e
    cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg freqfilt --no-e2e

WHAT THE METRICS MEAN:
    p50 / p95   Median and 95th-percentile wall-clock latency. p95 reveals tail latency.
    ratio       viprs / libvips. < 1.0 means viprs wins. E.g. 0.55x = viprs 1.82x faster.
    RSS         Peak resident set size in KB (memory footprint).
"#,
        default_threads = default_threads,
        default_cache_mib = default_cache_mib,
    );
}

pub fn parse_args(args: &[String]) -> BenchArgs {
    if args.len() < 2 || args[0] == "--help" || args[0] == "-h" {
        print_help();
        std::process::exit(if args.len() < 2 { 1 } else { 0 });
    }

    let input = args[0].clone();
    let op = normalize_operation_alias(&args[1]).to_owned();
    let mut iterations = DEFAULT_ITERATIONS;
    let mut threads = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(4);
    let mut op_args = Vec::new();
    let mut multi_sizes = false;
    let mut scenario_set = None;
    let mut json_output = false;
    let mut check_regression = false;
    let mut baseline = None;
    let mut e2e = true;
    let mut profile_stages = false;
    let mut ai_output = false;
    let mut profile_only = false;
    let mut parameter_matrix = false;
    let mut workflow_output_format = None;

    let mut i = 2;
    while i < args.len() {
        if args[i] == "--iterations" && i + 1 < args.len() {
            iterations = args[i + 1].parse().unwrap_or(DEFAULT_ITERATIONS);
            i += 2;
        } else if args[i] == "--threads" && i + 1 < args.len() {
            let parsed = args[i + 1].parse::<usize>().ok();
            let Some(thread_count) = parsed.filter(|count| *count > 0) else {
                eprintln!("--threads requires a non-zero integer worker count");
                std::process::exit(1);
            };
            threads = thread_count;
            i += 2;
        } else if args[i] == "--sizes" || args[i] == "--multi-size" || args[i] == "--multi_size" {
            multi_sizes = true;
            i += 1;
        } else if args[i] == "--scenario-set" && i + 1 < args.len() {
            scenario_set = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--json" {
            json_output = true;
            i += 1;
        } else if args[i] == "--check-regression" {
            check_regression = true;
            i += 1;
        } else if args[i] == "--baseline" && i + 1 < args.len() {
            baseline = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else if args[i] == "--no-e2e" {
            e2e = false;
            i += 1;
        } else if args[i] == "--profile-stages" {
            profile_stages = true;
            i += 1;
        } else if args[i] == "--ai" {
            ai_output = true;
            i += 1;
        } else if args[i] == "--profile-only" {
            profile_only = true;
            i += 1;
        } else if args[i] == "--matrix" {
            parameter_matrix = true;
            i += 1;
        } else if is_workflow_like_op(&op) && args[i] == "--output-format" {
            let Some(format) = args.get(i + 1) else {
                eprintln!("--output-format requires a format value");
                std::process::exit(1);
            };
            workflow_output_format = Some(format.clone());
            i += 2;
        } else if is_workflow_like_op(&op) && args[i].starts_with("--output-format=") {
            let format = args[i]
                .split_once('=')
                .map(|(_, value)| value)
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| {
                    eprintln!("--output-format requires a format value");
                    std::process::exit(1);
                });
            workflow_output_format = Some(format.to_owned());
            i += 1;
        } else if args[i] == "--help" || args[i] == "-h" {
            print_help();
            std::process::exit(0);
        } else {
            op_args.push(args[i].clone());
            i += 1;
        }
    }

    if check_regression && baseline.is_none() {
        eprintln!("--check-regression requires --baseline <path>");
        std::process::exit(1);
    }

    if multi_sizes && scenario_set.is_some() {
        eprintln!("--sizes and --scenario-set are mutually exclusive");
        std::process::exit(1);
    }

    if profile_stages && op != "thumbnail" {
        eprintln!("--profile-stages is only supported for the thumbnail operation");
        std::process::exit(1);
    }

    if profile_stages && e2e {
        eprintln!("--profile-stages requires --no-e2e so stage timings exclude codec decode");
        std::process::exit(1);
    }

    if ai_output && !profile_stages {
        eprintln!("--ai is only supported together with --profile-stages");
        std::process::exit(1);
    }

    if thumbnail_requires_explicit_width(&op, parameter_matrix, &op_args) {
        eprintln!("thumbnail requires an explicit width argument (or use --matrix)");
        std::process::exit(1);
    }

    if profile_only {
        e2e = false;
    }

    if let Some(target_format) = workflow_output_format {
        let mut normalized_op_args = vec![target_format];
        normalized_op_args.extend(op_args.into_iter().filter(|arg| arg.parse::<u32>().is_ok()));
        op_args = normalized_op_args;
    }

    BenchArgs {
        input,
        op,
        op_args,
        iterations,
        threads,
        multi_sizes,
        scenario_set,
        json_output,
        check_regression,
        baseline,
        e2e,
        profile_stages,
        ai_output,
        profile_only,
        parameter_matrix,
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_args, thumbnail_requires_explicit_width};

    #[test]
    fn parse_args_normalizes_load_tiff_alias() {
        let args = vec![
            "tests/fixtures/images".to_owned(),
            "load_tiff".to_owned(),
            "--sizes".to_owned(),
        ];

        let parsed = parse_args(&args);

        assert_eq!(parsed.op, "load-tiff");
        assert!(parsed.multi_sizes);
    }

    #[test]
    fn parse_args_normalizes_load_exr_alias() {
        let args = vec![
            "tests/fixtures/images".to_owned(),
            "load_exr".to_owned(),
            "--sizes".to_owned(),
        ];

        let parsed = parse_args(&args);

        assert_eq!(parsed.op, "load-exr");
        assert!(parsed.multi_sizes);
    }

    #[test]
    fn parse_args_normalizes_new_decode_aliases() {
        for (alias, canonical) in [
            ("load_heif", "load-heif"),
            ("load_avif", "load-avif"),
            ("load_jpeg", "load-jpeg"),
            ("load_svg", "load-svg"),
            ("load_pdf", "load-pdf"),
        ] {
            let args = vec![
                "tests/fixtures/images".to_owned(),
                alias.to_owned(),
                "--sizes".to_owned(),
            ];
            let parsed = parse_args(&args);
            assert_eq!(parsed.op, canonical);
            assert!(parsed.multi_sizes);
        }
    }

    #[test]
    fn parse_args_enables_ai_output_flag() {
        let args = vec![
            "input.jpg".to_owned(),
            "thumbnail".to_owned(),
            "800".to_owned(),
            "--profile-stages".to_owned(),
            "--no-e2e".to_owned(),
            "--ai".to_owned(),
        ];

        let parsed = parse_args(&args);

        assert!(parsed.profile_stages);
        assert!(parsed.ai_output);
    }

    #[test]
    fn thumbnail_requires_explicit_width_without_matrix() {
        assert!(thumbnail_requires_explicit_width("thumbnail", false, &[]));
        assert!(thumbnail_requires_explicit_width(
            "thumbnail",
            false,
            &["webp".to_owned()]
        ));
        assert!(!thumbnail_requires_explicit_width(
            "thumbnail",
            false,
            &["400".to_owned()]
        ));
        assert!(!thumbnail_requires_explicit_width("thumbnail", true, &[]));
        assert!(!thumbnail_requires_explicit_width("workflow", false, &[]));
    }

    #[test]
    fn parse_args_accepts_thumbnail_width() {
        let args = vec![
            "input.webp".to_owned(),
            "thumbnail".to_owned(),
            "400".to_owned(),
            "--iterations".to_owned(),
            "5".to_owned(),
        ];

        let parsed = parse_args(&args);

        assert_eq!(parsed.op, "thumbnail");
        assert_eq!(parsed.op_args, vec!["400".to_owned()]);
    }

    #[test]
    fn parse_args_accepts_thumbnail_matrix_without_width() {
        let args = vec![
            "input.webp".to_owned(),
            "thumbnail".to_owned(),
            "--matrix".to_owned(),
        ];

        let parsed = parse_args(&args);

        assert_eq!(parsed.op, "thumbnail");
        assert!(parsed.parameter_matrix);
        assert!(parsed.op_args.is_empty());
    }

    #[test]
    fn parse_args_normalizes_perceptual_enhance_output_format_flag() {
        let args = vec![
            "input.jpg".to_owned(),
            "perceptual_enhance".to_owned(),
            "640".to_owned(),
            "--output-format".to_owned(),
            "webp".to_owned(),
        ];

        let parsed = parse_args(&args);

        assert_eq!(parsed.op_args, vec!["webp".to_owned(), "640".to_owned()]);
    }
}
