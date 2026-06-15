use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use viprs::adapters::codecs::TiffCodec;
use viprs::domain::codec_options::{LoadOptions, SaveOptions, TiffCompression};
use viprs::domain::format::{U8, U16};
use viprs::domain::image::Image;
use viprs::ports::codec::ImageEncoder;

use super::types::{
    BenchFixtureSpec, BenchImage, BenchResult, Comparison, ScenarioSpec, SummaryRow, TiffSaveInput,
    TrendRecord,
};
pub const WARMUP_ITERATIONS: usize = 3;
pub const DEFAULT_ITERATIONS: usize = 50;
// 32 MiB: covers one full 2048×2048 RGB image in 128×128 tiles (~12 MB) with
// 2.5× slack. Keeps the LRU working set small so stale tiles from prior
// benchmark iterations are evicted promptly instead of accumulating to
// the old 256-tile-count limit that caused the 9.85× RSS explosion (B-239).
pub const DEFAULT_VIPRS_CACHE_BYTES: usize = 32 * 1024 * 1024;
pub const DEFAULT_JP2K_LOAD_THREADS: usize = 8;
pub const DEBUG_BENCH_ERROR: &str = "ERROR: cargo xtask bench must be run via 'cargo xtask bench', not './target/debug/xtask'. Compile with --release or use 'cargo xtask'.";

/// Default `cargo xtask bench --sizes` fixtures.
pub const STANDARD_BENCH_FIXTURES: &[BenchFixtureSpec] = &[
    BenchFixtureSpec {
        size: 512,
        width: 512,
        height: 512,
        input: "tests/fixtures/images/bench_512x512.jpg",
    },
    BenchFixtureSpec {
        size: 640,
        width: 640,
        height: 480,
        input: "tests/fixtures/images/bench_640x480.jpg",
    },
    BenchFixtureSpec {
        size: 1920,
        width: 1920,
        height: 1080,
        input: "tests/fixtures/images/bench_1920x1080.jpg",
    },
    BenchFixtureSpec {
        size: 2048,
        width: 2048,
        height: 2048,
        input: "tests/fixtures/images/bench_2048x2048.jpg",
    },
    BenchFixtureSpec {
        size: 8192,
        width: 8192,
        height: 8192,
        input: "tests/fixtures/images/bench_8192x8192.jpg",
    },
];
pub const FLOAT_BENCH_FIXTURES: &[BenchFixtureSpec] = &[
    BenchFixtureSpec {
        size: 512,
        width: 512,
        height: 512,
        input: "tests/fixtures/images/bench_512x512.exr",
    },
    BenchFixtureSpec {
        size: 2048,
        width: 2048,
        height: 2048,
        input: "tests/fixtures/images/bench_2048x2048.exr",
    },
    BenchFixtureSpec {
        size: 8192,
        width: 8192,
        height: 8192,
        input: "tests/fixtures/images/bench_8192x8192.exr",
    },
];
pub const COMPOSITE_BENCH_FIXTURES: &[BenchFixtureSpec] = &[
    BenchFixtureSpec {
        size: 512,
        width: 512,
        height: 512,
        input: "tests/fixtures/images/bench_512x512_rgba.png",
    },
    BenchFixtureSpec {
        size: 2048,
        width: 2048,
        height: 2048,
        input: "tests/fixtures/images/bench_2048x2048_rgba.png",
    },
    BenchFixtureSpec {
        size: 8192,
        width: 8192,
        height: 8192,
        input: "tests/fixtures/images/bench_8192x8192_rgba.png",
    },
];
pub const ZOOM_BENCH_FIXTURES: &[BenchFixtureSpec] = &[
    BenchFixtureSpec {
        size: 512,
        width: 512,
        height: 512,
        input: "tests/fixtures/images/bench_512x512.jpg",
    },
    BenchFixtureSpec {
        size: 2048,
        width: 2048,
        height: 2048,
        input: "tests/fixtures/images/bench_2048x2048.jpg",
    },
    BenchFixtureSpec {
        size: 8192,
        width: 8192,
        height: 8192,
        input: "tests/fixtures/images/bench_8192x8192.jpg",
    },
];
pub const EXR_BENCH_FIXTURES: &[BenchFixtureSpec] = &[
    BenchFixtureSpec {
        size: 512,
        width: 512,
        height: 512,
        input: "tests/fixtures/images/bench_512x512.exr",
    },
    BenchFixtureSpec {
        size: 2048,
        width: 2048,
        height: 2048,
        input: "tests/fixtures/images/bench_2048x2048.exr",
    },
    BenchFixtureSpec {
        size: 8192,
        width: 8192,
        height: 8192,
        input: "tests/fixtures/images/bench_8192x8192.exr",
    },
];
pub const TIFF_BENCH_FIXTURES: &[BenchFixtureSpec] = &[
    BenchFixtureSpec {
        size: 512,
        width: 512,
        height: 512,
        input: "tests/fixtures/images/bench_512x512.tif",
    },
    BenchFixtureSpec {
        size: 2048,
        width: 2048,
        height: 2048,
        input: "tests/fixtures/images/bench_2048x2048.tif",
    },
    BenchFixtureSpec {
        size: 8192,
        width: 8192,
        height: 8192,
        input: "tests/fixtures/images/bench_8192x8192.tif",
    },
];
pub const HEIF_BENCH_FIXTURES: &[BenchFixtureSpec] = &[
    BenchFixtureSpec {
        size: 512,
        width: 512,
        height: 512,
        input: "tests/fixtures/images/bench_512x512.heic",
    },
    BenchFixtureSpec {
        size: 2048,
        width: 2048,
        height: 2048,
        input: "tests/fixtures/images/bench_2048x2048.heic",
    },
    BenchFixtureSpec {
        size: 8192,
        width: 8192,
        height: 8192,
        input: "tests/fixtures/images/bench_8192x8192.heic",
    },
];
pub const AVIF_BENCH_FIXTURES: &[BenchFixtureSpec] = &[
    BenchFixtureSpec {
        size: 512,
        width: 512,
        height: 512,
        input: "tests/fixtures/images/bench_512x512.avif",
    },
    BenchFixtureSpec {
        size: 2048,
        width: 2048,
        height: 2048,
        input: "tests/fixtures/images/bench_2048x2048.avif",
    },
    BenchFixtureSpec {
        size: 8192,
        width: 8192,
        height: 8192,
        input: "tests/fixtures/images/bench_8192x8192.avif",
    },
];
pub const SVG_BENCH_FIXTURES: &[BenchFixtureSpec] = &[
    BenchFixtureSpec {
        size: 512,
        width: 512,
        height: 512,
        input: "tests/fixtures/images/bench_512x512.svg",
    },
    BenchFixtureSpec {
        size: 2048,
        width: 2048,
        height: 2048,
        input: "tests/fixtures/images/bench_2048x2048.svg",
    },
    BenchFixtureSpec {
        size: 8192,
        width: 8192,
        height: 8192,
        input: "tests/fixtures/images/bench_8192x8192.svg",
    },
];
pub const PDF_BENCH_FIXTURES: &[BenchFixtureSpec] = &[
    BenchFixtureSpec {
        size: 512,
        width: 512,
        height: 512,
        input: "tests/fixtures/images/bench_512x512.pdf",
    },
    BenchFixtureSpec {
        size: 2048,
        width: 2048,
        height: 2048,
        input: "tests/fixtures/images/bench_2048x2048.pdf",
    },
    BenchFixtureSpec {
        size: 8192,
        width: 8192,
        height: 8192,
        input: "tests/fixtures/images/bench_8192x8192.pdf",
    },
];
pub const STANDARD_SIZES: &[u32] = &[512, 2048, 8192];
pub const INPUT_DIVERSITY_SUPPORTED_OPS: &[&str] = &[
    "load",
    "invert",
    "bandmean",
    "add",
    "multiply",
    "and",
    "equal",
    "linear",
    "colourspace",
    "resize",
    "zoom",
    "shrink",
    "shrinkh",
    "shrinkv",
    "thumbnail",
    "gauss_blur",
    "workflow",
    "abs",
    "sign",
    "round",
    "floor",
    "ceil",
    "freqfilt",
];
pub const COMPUTE_BASELINE_SCENARIOS: &[ScenarioSpec] = &[
    ScenarioSpec {
        key: "gray-u8-512",
        input: "tests/fixtures/images/bench_512x512_gray.png",
        description: "1-band grayscale compute baseline (512px)",
    },
    ScenarioSpec {
        key: "gray-u8-2048",
        input: "tests/fixtures/images/bench_2048x2048_gray.png",
        description: "1-band grayscale compute baseline (2048px)",
    },
    ScenarioSpec {
        key: "gray-u8-8192",
        input: "tests/fixtures/images/bench_8192x8192_gray.png",
        description: "1-band grayscale compute baseline (8192px)",
    },
    ScenarioSpec {
        key: "rgb-u8-512",
        input: "tests/fixtures/images/bench_512x512.jpg",
        description: "3-band RGB compute baseline (512px)",
    },
    ScenarioSpec {
        key: "rgb-u8-2048",
        input: "tests/fixtures/images/bench_2048x2048.jpg",
        description: "3-band RGB compute baseline (2048px)",
    },
    ScenarioSpec {
        key: "rgb-u8-8192",
        input: "tests/fixtures/images/bench_8192x8192.jpg",
        description: "3-band RGB compute baseline (8192px)",
    },
    ScenarioSpec {
        key: "rgba-u8-512",
        input: "tests/fixtures/images/bench_512x512_rgba.png",
        description: "4-band RGBA compute baseline (512px)",
    },
    ScenarioSpec {
        key: "rgba-u8-2048",
        input: "tests/fixtures/images/bench_2048x2048_rgba.png",
        description: "4-band RGBA compute baseline (2048px)",
    },
    ScenarioSpec {
        key: "rgba-u8-8192",
        input: "tests/fixtures/images/bench_8192x8192_rgba.png",
        description: "4-band RGBA compute baseline (8192px)",
    },
    ScenarioSpec {
        key: "rgb-u16-512",
        input: "tests/fixtures/images/bench_512x512_u16.tif",
        description: "16-bit RGB compute baseline (512px)",
    },
    ScenarioSpec {
        key: "rgb-u16-2048",
        input: "tests/fixtures/images/bench_2048x2048_u16.tif",
        description: "16-bit RGB compute baseline (2048px)",
    },
    ScenarioSpec {
        key: "rgb-u16-8192",
        input: "tests/fixtures/images/bench_8192x8192_u16.tif",
        description: "16-bit RGB compute baseline (8192px)",
    },
    ScenarioSpec {
        key: "rgb-f32-512",
        input: "tests/fixtures/images/bench_512x512.exr",
        description: "float RGB compute baseline (512px)",
    },
    ScenarioSpec {
        key: "rgb-f32-2048",
        input: "tests/fixtures/images/bench_2048x2048.exr",
        description: "float RGB compute baseline (2048px)",
    },
    ScenarioSpec {
        key: "rgb-f32-8192",
        input: "tests/fixtures/images/bench_8192x8192.exr",
        description: "float RGB compute baseline (8192px)",
    },
];
pub const THUMBNAIL_TARGET_WIDTHS: &[u32] = &[100, 200, 400, 800];
pub const GAUSS_BLUR_SIGMAS: &[&str] = &["0.5", "1.0", "2.0", "5.0"];
pub const COLOURSPACE_ROUTES: &[&[&str]] = &[
    &["lab"],
    &["lab", "srgb"],
    &["xyz"],
    &["xyz", "srgb"],
    &["cmyk"],
    &["cmyk", "srgb"],
    &["hsv"],
    &["hsv", "srgb"],
    &["scrgb"],
    &["scrgb", "srgb"],
];

/// Production workflow scenarios: each represents a format→format conversion
/// through a realistic pipeline (decode → thumbnail → sharpen → encode).
///
/// The `key` doubles as the target format arg for the `workflow` operation.
pub const PRODUCTION_WORKFLOW_SCENARIOS: &[ScenarioSpec] = &[
    ScenarioSpec {
        key: "512-jpg-to-webp",
        input: "tests/fixtures/images/bench_512x512.jpg",
        description: "JPEG→WebP CDN serving (512px)",
    },
    ScenarioSpec {
        key: "2048-jpg-to-webp",
        input: "tests/fixtures/images/bench_2048x2048.jpg",
        description: "JPEG→WebP CDN serving (2048px)",
    },
    ScenarioSpec {
        key: "8192-jpg-to-webp",
        input: "tests/fixtures/images/bench_8192x8192.jpg",
        description: "JPEG→WebP CDN serving (8192px)",
    },
    ScenarioSpec {
        key: "512-jpg-to-avif",
        input: "tests/fixtures/images/bench_512x512.jpg",
        description: "JPEG→AVIF modern optimization (512px)",
    },
    ScenarioSpec {
        key: "2048-jpg-to-avif",
        input: "tests/fixtures/images/bench_2048x2048.jpg",
        description: "JPEG→AVIF modern optimization (2048px)",
    },
    ScenarioSpec {
        key: "8192-jpg-to-avif",
        input: "tests/fixtures/images/bench_8192x8192.jpg",
        description: "JPEG→AVIF modern optimization (8192px)",
    },
    ScenarioSpec {
        key: "512-png-to-webp",
        input: "tests/fixtures/images/bench_512x512.png",
        description: "PNG→WebP e-commerce catalog (512px)",
    },
    ScenarioSpec {
        key: "2048-png-to-webp",
        input: "tests/fixtures/images/bench_2048x2048.png",
        description: "PNG→WebP e-commerce catalog (2048px)",
    },
    ScenarioSpec {
        key: "8192-png-to-webp",
        input: "tests/fixtures/images/bench_8192x8192.png",
        description: "PNG→WebP e-commerce catalog (8192px)",
    },
    ScenarioSpec {
        key: "512-webp-to-jpg",
        input: "tests/fixtures/images/bench_512x512.webp",
        description: "WebP→JPEG fallback compat (512px)",
    },
    ScenarioSpec {
        key: "2048-webp-to-jpg",
        input: "tests/fixtures/images/bench_2048x2048.webp",
        description: "WebP→JPEG fallback compat (2048px)",
    },
    ScenarioSpec {
        key: "8192-webp-to-jpg",
        input: "tests/fixtures/images/bench_8192x8192.webp",
        description: "WebP→JPEG fallback compat (8192px)",
    },
    ScenarioSpec {
        key: "512-jpg-to-jpg",
        input: "tests/fixtures/images/bench_512x512.jpg",
        description: "JPEG→JPEG re-encode/optimize (512px)",
    },
    ScenarioSpec {
        key: "2048-jpg-to-jpg",
        input: "tests/fixtures/images/bench_2048x2048.jpg",
        description: "JPEG→JPEG re-encode/optimize (2048px)",
    },
    ScenarioSpec {
        key: "8192-jpg-to-jpg",
        input: "tests/fixtures/images/bench_8192x8192.jpg",
        description: "JPEG→JPEG re-encode/optimize (8192px)",
    },
    ScenarioSpec {
        key: "512-png-to-png",
        input: "tests/fixtures/images/bench_512x512.png",
        description: "PNG→PNG lossless pipeline (512px)",
    },
    ScenarioSpec {
        key: "2048-png-to-png",
        input: "tests/fixtures/images/bench_2048x2048.png",
        description: "PNG→PNG lossless pipeline (2048px)",
    },
    ScenarioSpec {
        key: "8192-png-to-png",
        input: "tests/fixtures/images/bench_8192x8192.png",
        description: "PNG→PNG lossless pipeline (8192px)",
    },
];

const DEFAULT_WORKFLOW_WIDTH: &str = "400";

fn workflow_width_arg(op_args: &[String]) -> String {
    op_args
        .get(1)
        .or_else(|| op_args.first().filter(|arg| arg.parse::<u32>().is_ok()))
        .cloned()
        .unwrap_or_else(|| DEFAULT_WORKFLOW_WIDTH.to_owned())
}

pub fn canonical_op_arg_matrix(op: &str) -> Option<Vec<Vec<String>>> {
    match op {
        "thumbnail" => Some(
            THUMBNAIL_TARGET_WIDTHS
                .iter()
                .map(|width| vec![width.to_string()])
                .collect(),
        ),
        "gauss_blur" => Some(
            GAUSS_BLUR_SIGMAS
                .iter()
                .map(|sigma| vec![(*sigma).to_owned()])
                .collect(),
        ),
        "colourspace" => Some(
            COLOURSPACE_ROUTES
                .iter()
                .map(|route| route.iter().map(|step| (*step).to_owned()).collect())
                .collect(),
        ),
        "dilate" | "erode" | "median_blur" => {
            Some(vec![vec!["3".to_owned()], vec!["5".to_owned()]])
        }
        _ => None,
    }
}

pub fn scenario_display_label(op: &str, op_args: &[String]) -> String {
    match op {
        "thumbnail" => format!(
            "thumbnail width={}",
            op_args.first().map(String::as_str).unwrap_or("800")
        ),
        "zoom" => format!(
            "zoom xfac={} yfac={}",
            op_args.first().map(String::as_str).unwrap_or("2"),
            op_args
                .get(1)
                .map(String::as_str)
                .unwrap_or_else(|| op_args.first().map(String::as_str).unwrap_or("2"))
        ),
        "gauss_blur" => format!(
            "gauss_blur sigma={}",
            op_args.first().map(String::as_str).unwrap_or("1.5")
        ),
        "median_blur" => format!(
            "median_blur size={}",
            op_args.first().map(String::as_str).unwrap_or("3")
        ),
        "unsharp_mask" => format!(
            "unsharp_mask sigma={} strength={}",
            op_args.first().map(String::as_str).unwrap_or("0.5"),
            op_args.get(1).map(String::as_str).unwrap_or("3.0")
        ),
        "affine" => format!(
            "affine matrix=[{},{},{},{}]",
            op_args.first().map(String::as_str).unwrap_or("1.0"),
            op_args.get(1).map(String::as_str).unwrap_or("0.2"),
            op_args.get(2).map(String::as_str).unwrap_or("-0.1"),
            op_args.get(3).map(String::as_str).unwrap_or("0.95")
        ),
        "similarity" => format!(
            "similarity scale={} angle={}",
            op_args.first().map(String::as_str).unwrap_or("0.9"),
            op_args.get(1).map(String::as_str).unwrap_or("15")
        ),
        "mapim" => format!(
            "mapim dx={} dy={}",
            op_args.first().map(String::as_str).unwrap_or("0.25"),
            op_args.get(1).map(String::as_str).unwrap_or("0.25")
        ),
        "composite" => format!(
            "composite mode={}",
            op_args.first().map(String::as_str).unwrap_or("over")
        ),
        "workflow" => format!(
            "workflow target={} width={}",
            op_args.first().map(String::as_str).unwrap_or("webp"),
            op_args.get(1).map(String::as_str).unwrap_or("400")
        ),
        "invert_invert" => "invert → invert".to_owned(),
        "thumbnail_sharpen" => "thumbnail width=400 → sharpen sigma=0.5 strength=3.0".to_owned(),
        "thumbnail_colourspace_cast" => {
            "thumbnail width=400 → colourspace lab → cast u8".to_owned()
        }
        "thumbnail_gauss_blur" => "thumbnail width=400 → gauss_blur sigma=2.0".to_owned(),
        "thumbnail_linear" => "thumbnail width=400 → linear scale=1.2 offset=0.0".to_owned(),
        "resize_colourspace" => "resize scale=0.5 → colourspace lab".to_owned(),
        "embed" => "embed offset=64,48 canvas=src+256x192 extend=copy".to_owned(),
        "extract-area" => "extract-area x=32 y=24 width=src-64 height=src-48".to_owned(),
        "embed_extract" => "embed min=2048x2048 → extract_area 100,100 800x600".to_owned(),
        "three_op_chain" => {
            "thumbnail width=400 → sharpen sigma=0.5 strength=3.0 → gauss_blur sigma=1.0".to_owned()
        }
        "colourspace" if op_args.is_empty() => "colourspace auto".to_owned(),
        "colourspace" => format!("colourspace {}", op_args.join("→")),
        _ if op_args.is_empty() => op.to_owned(),
        _ => format!("{op} {}", op_args.join(" ")),
    }
}

pub fn workflow_op_args_for_scenario(scenario_key: &str, op_args: &[String]) -> Vec<String> {
    let target_format = scenario_key
        .rsplit_once("-to-")
        .map(|(_, fmt)| fmt)
        .or_else(|| {
            op_args
                .first()
                .filter(|arg| arg.parse::<u32>().is_err())
                .map(String::as_str)
        })
        .unwrap_or("webp");
    let width = workflow_width_arg(op_args);

    vec![target_format.to_owned(), width]
}

pub fn scenario_set_display_label(scenario_set_name: &str, op: &str, op_args: &[String]) -> String {
    if scenario_set_name == "production-workflows" && op == "workflow" {
        format!(
            "workflow target=<scenario-derived> width={}",
            workflow_width_arg(op_args)
        )
    } else {
        scenario_display_label(op, op_args)
    }
}

pub fn scenario_slug(op: &str, op_args: &[String]) -> String {
    if op_args.is_empty() {
        return op.to_owned();
    }

    let args_slug = op_args
        .iter()
        .flat_map(|arg| arg.chars())
        .map(|ch| match ch {
            '0'..='9' | 'a'..='z' | 'A'..='Z' => ch,
            '.' => 'p',
            _ => '_',
        })
        .collect::<String>();
    format!("{op}_{args_slug}")
}

pub fn parse_save_tiff_compression(op_args: &[String]) -> TiffCompression {
    match op_args.first().map(String::as_str).unwrap_or("none") {
        "none" => TiffCompression::None,
        "lzw" => TiffCompression::Lzw,
        "deflate" => TiffCompression::Deflate,
        "packbits" => TiffCompression::PackBits,
        "jpeg" => TiffCompression::Jpeg,
        other => {
            eprintln!(
                "save-tiff only accepts optional compression arg 'none', 'lzw', 'deflate', \
                 'packbits', or 'jpeg', got '{other}'"
            );
            std::process::exit(1);
        }
    }
}

pub fn load_tiff_save_input(input: &Path) -> TiffSaveInput {
    if let Ok(image) = Image::<U8>::load(input) {
        TiffSaveInput::U8(image)
    } else if let Ok(image) = Image::<U16>::load(input) {
        TiffSaveInput::U16(image)
    } else {
        eprintln!("save-tiff expects an integer input image (U8/U16)");
        std::process::exit(1);
    }
}

pub fn load_bench_image(input: &Path) -> BenchImage {
    load_bench_image_with_options(input, &LoadOptions::default())
}

pub fn default_bench_threads(requested_threads: usize, op: &str, input: &Path) -> usize {
    let is_jp2k_load = op == "load"
        && input
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                matches!(
                    ext.to_ascii_lowercase().as_str(),
                    "jp2" | "j2k" | "jpf" | "jpx" | "j2c" | "jpc"
                )
            });
    if !is_jp2k_load {
        return requested_threads;
    }

    let default_threads = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(4);
    if requested_threads == default_threads {
        requested_threads.min(DEFAULT_JP2K_LOAD_THREADS)
    } else {
        requested_threads
    }
}

pub fn load_bench_image_with_options(input: &Path, opts: &LoadOptions) -> BenchImage {
    use viprs::domain::format::F32;
    if let Ok(image) = Image::<U8>::load_with_options(input, opts) {
        BenchImage::U8(image)
    } else if let Ok(image) = Image::<U16>::load_with_options(input, opts) {
        BenchImage::U16(image)
    } else if let Ok(image) = Image::<F32>::load_with_options(input, opts) {
        BenchImage::F32(image)
    } else {
        eprintln!("failed to load benchmark input as U8, U16, or F32 image");
        std::process::exit(1);
    }
}

pub fn encode_tiff_with_input(
    codec: &TiffCodec,
    input: &TiffSaveInput,
    opts: &SaveOptions,
) -> Vec<u8> {
    match input {
        TiffSaveInput::U8(image) => codec
            .encode_with_options(image, opts)
            .expect("Failed to encode TIFF"),
        TiffSaveInput::U16(image) => codec
            .encode_with_options(image, opts)
            .expect("Failed to encode 16-bit TIFF"),
    }
}

pub fn debug_build_error() -> Option<&'static str> {
    if cfg!(debug_assertions) {
        Some(DEBUG_BENCH_ERROR)
    } else {
        None
    }
}

pub fn getrusage() -> libc::rusage {
    // SAFETY: `rusage` is a POD struct; zeroing it is valid. `getrusage(RUSAGE_SELF, …)` is
    // always safe for the calling process and writes a fully-initialized `rusage` on success.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        ru
    }
}

pub fn viprs_backend_label() -> String {
    "viprs".to_owned()
}

pub fn libvips_backend_label() -> String {
    "libvips".to_owned()
}

pub fn percentile(sorted: &[u64], p: f64) -> u64 {
    let idx = ((sorted.len() as f64) * p) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn infer_square_size_from_input(input_path: &Path) -> Option<u32> {
    let stem = input_path.file_stem()?.to_str()?;
    let dims = stem.strip_prefix("bench_")?;
    let (width, height) = dims.split_once('x')?;
    let width = width.parse::<u32>().ok()?;
    let height = height.parse::<u32>().ok()?;
    (width == height).then_some(width)
}

pub fn bench_result_percentiles(result: &BenchResult) -> (f64, f64) {
    let mut sorted = result.wall_ns.clone();
    sorted.sort_unstable();
    (
        percentile(&sorted, 0.50) as f64 / 1e6,
        percentile(&sorted, 0.95) as f64 / 1e6,
    )
}

pub fn build_summary_row(
    op: &str,
    op_args: &[String],
    input_path: &Path,
    scenario: Option<&str>,
    size: Option<u32>,
    cmp: &Comparison,
) -> SummaryRow {
    let (viprs_p50_ms, viprs_p95_ms) = cmp
        .viprs
        .as_ref()
        .map(bench_result_percentiles)
        .unwrap_or((0.0, 0.0));
    let (libvips_p50_ms, libvips_p95_ms) = cmp
        .libvips
        .as_ref()
        .map(bench_result_percentiles)
        .map(|(p50, p95)| (Some(p50), Some(p95)))
        .unwrap_or((None, None));

    SummaryRow {
        op: op.to_owned(),
        op_args: op_args.to_vec(),
        input: input_path.display().to_string(),
        scenario: scenario.map(str::to_owned),
        size,
        viprs_p50_ms,
        viprs_p95_ms,
        libvips_p50_ms,
        libvips_p95_ms,
        ratio: cmp.ratios.as_ref().map(|r| r.latency_p50),
        ratio_p95: cmp.ratios.as_ref().map(|r| r.latency_p95),
    }
}

pub fn summary_key(row: &SummaryRow) -> String {
    let identity = row
        .scenario
        .clone()
        .or_else(|| row.size.map(|size| size.to_string()))
        .unwrap_or_else(|| row.input.clone());
    let op_args = if row.op_args.is_empty() {
        String::new()
    } else {
        row.op_args.join(",")
    };
    format!("{}|{}|{}", row.op, op_args, identity)
}

pub fn compute_baseline_scenarios() -> &'static [ScenarioSpec] {
    COMPUTE_BASELINE_SCENARIOS
}

pub fn scenario_set(name: &str) -> Option<&'static [ScenarioSpec]> {
    match name {
        "compute-baselines" | "input-diversity" => Some(compute_baseline_scenarios()),
        "production-workflows" => Some(PRODUCTION_WORKFLOW_SCENARIOS),
        _ => None,
    }
}

pub fn scenario_set_supports_op(op: &str) -> bool {
    INPUT_DIVERSITY_SUPPORTED_OPS.contains(&op)
}

pub fn bench_fixtures_for_op(op: &str) -> &'static [BenchFixtureSpec] {
    match op {
        "composite" => COMPOSITE_BENCH_FIXTURES,
        "zoom" => ZOOM_BENCH_FIXTURES,
        "load-heif" => HEIF_BENCH_FIXTURES,
        "load-avif" => AVIF_BENCH_FIXTURES,
        "load-svg" => SVG_BENCH_FIXTURES,
        "load-pdf" => PDF_BENCH_FIXTURES,
        "load-tiff" => TIFF_BENCH_FIXTURES,
        "load-exr" | "save-exr" => EXR_BENCH_FIXTURES,
        "abs" | "sign" | "round" | "floor" | "ceil" => FLOAT_BENCH_FIXTURES,
        _ => STANDARD_BENCH_FIXTURES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::repo_root;
    use viprs::domain::{format::U8, image::Image};

    #[test]
    fn compute_baseline_scenarios_cover_all_standard_sizes_per_fixture_family() {
        let keys = COMPUTE_BASELINE_SCENARIOS
            .iter()
            .map(|scenario| scenario.key)
            .collect::<Vec<_>>();

        for family in ["gray-u8", "rgb-u8", "rgba-u8", "rgb-u16", "rgb-f32"] {
            for size in STANDARD_SIZES {
                assert!(
                    keys.contains(&format!("{family}-{size}").as_str()),
                    "missing compute-baseline scenario for {family}-{size}"
                );
            }
        }
    }

    #[test]
    fn standard_bench_fixtures_cover_realistic_sizes_and_existing_paths() {
        let repo_root = repo_root();

        for expected_size in [512, 640, 1920, 2048, 8192] {
            let fixture = find_bench_fixture(expected_size)
                .unwrap_or_else(|| panic!("missing multi-size fixture for {expected_size}"));
            let input = repo_root.join(fixture.input);
            assert!(
                input.exists(),
                "missing multi-size fixture: {}",
                input.display()
            );
        }
    }

    #[test]
    fn composite_bench_fixtures_cover_rgba_standard_sizes() {
        let repo_root = repo_root();

        for expected_size in STANDARD_SIZES {
            let fixture = COMPOSITE_BENCH_FIXTURES
                .iter()
                .find(|fixture| fixture.size == *expected_size)
                .unwrap_or_else(|| panic!("missing composite fixture for {expected_size}"));
            let input = repo_root.join(fixture.input);
            assert!(
                input.exists(),
                "missing composite multi-size fixture: {}",
                input.display()
            );
        }
    }

    #[test]
    fn exr_bench_fixtures_cover_float_standard_sizes() {
        let repo_root = repo_root();

        for expected_size in STANDARD_SIZES {
            let fixture = EXR_BENCH_FIXTURES
                .iter()
                .find(|fixture| fixture.size == *expected_size)
                .unwrap_or_else(|| panic!("missing EXR fixture for {expected_size}"));
            let input = repo_root.join(fixture.input);
            assert!(
                input.exists(),
                "missing EXR multi-size fixture: {}",
                input.display()
            );
        }
    }

    #[test]
    fn zoom_bench_fixtures_cover_standard_sizes() {
        let repo_root = repo_root();

        for expected_size in [512, 2048, 8192] {
            let fixture = ZOOM_BENCH_FIXTURES
                .iter()
                .find(|fixture| fixture.size == expected_size)
                .unwrap_or_else(|| panic!("missing zoom fixture for {expected_size}"));
            let input = repo_root.join(fixture.input);
            assert!(
                input.exists(),
                "missing zoom multi-size fixture: {}",
                input.display()
            );
        }
    }

    #[test]
    fn exr_ops_use_exr_fixture_matrix() {
        assert_eq!(bench_fixtures_for_op("load-exr"), EXR_BENCH_FIXTURES);
        assert_eq!(bench_fixtures_for_op("save-exr"), EXR_BENCH_FIXTURES);
    }

    #[test]
    fn heif_and_avif_decode_ops_use_codec_specific_fixture_matrices() {
        assert_eq!(bench_fixtures_for_op("load-heif"), HEIF_BENCH_FIXTURES);
        assert_eq!(bench_fixtures_for_op("load-avif"), AVIF_BENCH_FIXTURES);
    }

    #[test]
    fn vector_decode_ops_use_vector_fixture_matrices() {
        assert_eq!(bench_fixtures_for_op("load-svg"), SVG_BENCH_FIXTURES);
        assert_eq!(bench_fixtures_for_op("load-pdf"), PDF_BENCH_FIXTURES);
    }

    #[test]
    fn bench_fixtures_switch_to_rgba_matrix_for_composite() {
        assert_eq!(bench_fixtures_for_op("composite"), COMPOSITE_BENCH_FIXTURES);
        assert_eq!(bench_fixtures_for_op("zoom"), ZOOM_BENCH_FIXTURES);
        assert_eq!(bench_fixtures_for_op("load-tiff"), TIFF_BENCH_FIXTURES);
        assert_eq!(bench_fixtures_for_op("load-exr"), EXR_BENCH_FIXTURES);
        assert_eq!(bench_fixtures_for_op("save-exr"), EXR_BENCH_FIXTURES);
        assert_eq!(bench_fixtures_for_op("invert"), STANDARD_BENCH_FIXTURES);
        assert_eq!(bench_fixtures_for_op("abs"), FLOAT_BENCH_FIXTURES);
        assert_eq!(bench_fixtures_for_op("round"), FLOAT_BENCH_FIXTURES);
    }

    #[test]
    fn compute_baseline_scenarios_have_existing_fixtures() {
        let repo_root = repo_root();

        for scenario in COMPUTE_BASELINE_SCENARIOS {
            let input = repo_root.join(scenario.input);
            assert!(
                input.exists(),
                "missing compute baseline fixture: {}",
                input.display()
            );
        }
    }

    #[test]
    fn heif_bench_fixtures_cover_standard_sizes() {
        let repo_root = repo_root();

        for expected_size in STANDARD_SIZES {
            let fixture = HEIF_BENCH_FIXTURES
                .iter()
                .find(|fixture| fixture.size == *expected_size)
                .unwrap_or_else(|| panic!("missing HEIF fixture for {expected_size}"));
            let input = repo_root.join(fixture.input);
            assert!(
                input.exists(),
                "missing HEIF multi-size fixture: {}",
                input.display()
            );
        }
    }

    #[test]
    fn avif_bench_fixtures_cover_standard_sizes() {
        let repo_root = repo_root();

        for expected_size in STANDARD_SIZES {
            let fixture = AVIF_BENCH_FIXTURES
                .iter()
                .find(|fixture| fixture.size == *expected_size)
                .unwrap_or_else(|| panic!("missing AVIF fixture for {expected_size}"));
            let input = repo_root.join(fixture.input);
            assert!(
                input.exists(),
                "missing AVIF multi-size fixture: {}",
                input.display()
            );
        }
    }

    #[test]
    fn svg_bench_fixtures_cover_standard_sizes() {
        let repo_root = repo_root();

        for expected_size in STANDARD_SIZES {
            let fixture = SVG_BENCH_FIXTURES
                .iter()
                .find(|fixture| fixture.size == *expected_size)
                .unwrap_or_else(|| panic!("missing SVG fixture for {expected_size}"));
            let input = repo_root.join(fixture.input);
            assert!(
                input.exists(),
                "missing SVG multi-size fixture: {}",
                input.display()
            );
        }
    }

    #[test]
    fn pdf_bench_fixtures_cover_standard_sizes() {
        let repo_root = repo_root();

        for expected_size in STANDARD_SIZES {
            let fixture = PDF_BENCH_FIXTURES
                .iter()
                .find(|fixture| fixture.size == *expected_size)
                .unwrap_or_else(|| panic!("missing PDF fixture for {expected_size}"));
            let input = repo_root.join(fixture.input);
            assert!(
                input.exists(),
                "missing PDF multi-size fixture: {}",
                input.display()
            );
        }
    }

    #[test]
    fn svg_bench_fixture_decodes_in_xtask() {
        let repo_root = repo_root();
        let fixture = SVG_BENCH_FIXTURES
            .iter()
            .find(|fixture| fixture.size == 512)
            .expect("missing SVG 512 fixture");
        let input = repo_root.join(fixture.input);

        let image = Image::<U8>::load(&input)
            .unwrap_or_else(|err| panic!("failed to decode SVG bench fixture: {err}"));

        assert_eq!(image.width(), fixture.width);
        assert_eq!(image.height(), fixture.height);
        assert_eq!(image.bands(), 4);
    }

    #[test]
    fn pdf_bench_fixture_decodes_in_xtask() {
        let repo_root = repo_root();
        let fixture = PDF_BENCH_FIXTURES
            .iter()
            .find(|fixture| fixture.size == 512)
            .expect("missing PDF 512 fixture");
        let input = repo_root.join(fixture.input);

        let image = Image::<U8>::load(&input)
            .unwrap_or_else(|err| panic!("failed to decode PDF bench fixture: {err}"));

        assert_eq!(image.width(), fixture.width);
        assert_eq!(image.height(), fixture.height);
        assert_eq!(image.bands(), 4);
    }

    #[test]
    fn production_workflow_scenario_set_exposes_static_matrix() {
        assert_eq!(
            scenario_set("production-workflows"),
            Some(PRODUCTION_WORKFLOW_SCENARIOS)
        );
    }

    #[test]
    fn production_workflow_scenarios_cover_all_standard_sizes_per_workflow() {
        let keys = PRODUCTION_WORKFLOW_SCENARIOS
            .iter()
            .map(|scenario| scenario.key)
            .collect::<Vec<_>>();

        for workflow in [
            "jpg-to-webp",
            "jpg-to-avif",
            "png-to-webp",
            "webp-to-jpg",
            "jpg-to-jpg",
            "png-to-png",
        ] {
            for size in STANDARD_SIZES {
                assert!(
                    keys.contains(&format!("{size}-{workflow}").as_str()),
                    "missing production-workflow scenario for {size}-{workflow}"
                );
            }
        }
    }

    #[test]
    fn production_workflow_scenarios_have_existing_supported_fixtures() {
        let repo_root = repo_root();

        for scenario in PRODUCTION_WORKFLOW_SCENARIOS {
            let target = scenario
                .key
                .rsplit_once("-to-")
                .map(|(_, fmt)| fmt)
                .expect("production workflow key must contain -to-");
            assert!(matches!(
                target,
                "jpg" | "jpeg" | "webp" | "png" | "avif" | "tif" | "tiff"
            ));

            let input = repo_root.join(scenario.input);
            assert!(
                input.exists(),
                "missing production workflow fixture: {}",
                input.display()
            );
            let ext = input
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .expect("fixture must have an extension");
            assert!(
                matches!(ext, "jpg" | "jpeg" | "png" | "webp"),
                "unexpected production workflow fixture format: {ext}"
            );
        }
    }

    #[test]
    fn workflow_op_args_follow_scenario_target_and_explicit_width() {
        assert_eq!(
            workflow_op_args_for_scenario("png-to-webp", &["ignored".into(), "512".into()]),
            vec!["webp".to_owned(), "512".to_owned()]
        );
        assert_eq!(
            workflow_op_args_for_scenario("jpg-to-avif", &[]),
            vec!["avif".to_owned(), DEFAULT_WORKFLOW_WIDTH.to_owned()]
        );
        assert_eq!(
            workflow_op_args_for_scenario("png-to-png", &["640".into()]),
            vec!["png".to_owned(), "640".to_owned()]
        );
    }

    #[test]
    fn production_workflow_scenario_set_label_reports_derived_target_and_width() {
        assert_eq!(
            scenario_set_display_label("production-workflows", "workflow", &[]),
            "workflow target=<scenario-derived> width=400"
        );
        assert_eq!(
            scenario_set_display_label(
                "production-workflows",
                "workflow",
                &["ignored".into(), "512".into()]
            ),
            "workflow target=<scenario-derived> width=512"
        );
        assert_eq!(
            scenario_set_display_label("compute-baselines", "workflow", &["webp".into()]),
            "workflow target=webp width=400"
        );
    }
}

pub fn parse_summary_rows(content: &str) -> Result<Vec<SummaryRow>, String> {
    serde_json::from_str::<Vec<SummaryRow>>(content)
        .or_else(|_| serde_json::from_str::<SummaryRow>(content).map(|row| vec![row]))
        .map_err(|error| format!("failed to parse baseline JSON: {error}"))
}

pub fn load_baseline_rows(path: &Path) -> Result<Vec<SummaryRow>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read baseline {}: {error}", path.display()))?;
    parse_summary_rows(&content)
}

pub fn regression_messages(
    current_rows: &[SummaryRow],
    baseline_rows: &[SummaryRow],
) -> Vec<String> {
    let baseline_by_key: BTreeMap<String, &SummaryRow> = baseline_rows
        .iter()
        .map(|row| (summary_key(row), row))
        .collect();
    let mut failures = Vec::new();

    for row in current_rows {
        let key = summary_key(row);
        let Some(baseline) = baseline_by_key.get(&key) else {
            failures.push(format!("missing baseline entry for {key}"));
            continue;
        };

        match (baseline.ratio, row.ratio) {
            (Some(baseline_ratio), Some(current_ratio)) => {
                let allowed_ratio = baseline_ratio * 1.10;
                if current_ratio > allowed_ratio {
                    failures.push(format!(
                        "{key} regressed: current ratio {:.3}x > allowed {:.3}x (baseline {:.3}x)",
                        current_ratio, allowed_ratio, baseline_ratio
                    ));
                }
            }
            (Some(_), None) => failures.push(format!("missing current ratio for {key}")),
            (None, Some(_)) => failures.push(format!("baseline ratio missing for {key}")),
            (None, None) => failures.push(format!("baseline and current ratios missing for {key}")),
        }
    }

    failures
}

pub fn find_bench_fixture(size: u32) -> Option<&'static BenchFixtureSpec> {
    STANDARD_BENCH_FIXTURES
        .iter()
        .find(|fixture| fixture.size == size)
}

/// Find the bench image for a given multi-size benchmark entry.
/// Return the short git SHA of HEAD, or an empty string on failure.
pub fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(o.stdout)
            } else {
                None
            }
        })
        .and_then(|b| String::from_utf8(b).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_default()
}

/// Return an ISO-8601 UTC timestamp string for the current moment.
///
/// Avoids pulling in `chrono` by formatting the Unix timestamp manually.
pub fn iso_timestamp() -> String {
    // Seconds since epoch; good enough for trend file sorting.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as YYYY-MM-DDThh:mm:ssZ using integer arithmetic.
    let s = secs;
    let days = s / 86400;
    let rem = s % 86400;
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let sec = rem % 60;

    // Julian Day Number → calendar date (civil date from epoch).
    // Algorithm: Richards (2013) — no external deps, no unsafe.
    let jd = days + 2440588; // 2440588 = JDN of 1970-01-01
    let f = jd + 1401 + (((4 * jd + 274277) / 146097) * 3) / 4 - 38;
    let e = 4 * f + 3;
    let g = (e % 1461) / 4;
    let dg = 5 * g + 2;
    let dd = (dg % 153) / 5 + 1;
    let dm = (dg / 153 + 2) % 12 + 1;
    let dy = e / 1461 - 4716 + (14 - dm) / 12;

    format!("{dy:04}-{dm:02}-{dd:02}T{h:02}:{m:02}:{sec:02}Z")
}

/// Append `record` to `<results_dir>/trend.jsonl`.
///
/// The file is created on first use. Each line is a self-contained JSON object.
pub fn append_trend(results_dir: &Path, record: &TrendRecord) {
    let path = results_dir.join("trend.jsonl");
    let line = match serde_json::to_string(record) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("WARNING: failed to serialize trend record: {e}");
            return;
        }
    };
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("WARNING: failed to open {}: {e}", path.display());
            return;
        }
    };
    if let Err(e) = writeln!(file, "{line}") {
        eprintln!("WARNING: failed to write trend record: {e}");
    }
}
