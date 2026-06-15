/// dhat heap profiler — produces allocation profiles with full call stacks.
///
/// Unlike a simple counting allocator, dhat records WHERE each allocation happens,
/// how long it lives, how many bytes it uses, and produces a JSON that can be
/// visualized at https://nnethercote.github.io/dh_view/dh_view.html
///
/// Usage:
///   cargo run --example dhat_profile --release -- <input> <op> [op_args...]
///
/// Examples:
///   cargo run --example dhat_profile --release -- tests/fixtures/images/bench_2048x2048.jpg thumbnail 400
///   cargo run --example dhat_profile --release -- tests/fixtures/images/bench_2048x2048.jpg resize 0.5
///   cargo run --example dhat_profile --release -- tests/fixtures/images/bench_2048x2048.jpg sharpen 0.5 3.0
///   cargo run --example dhat_profile --release -- invert 2048
///   cargo run --example dhat_profile --release -- add 8192
///   cargo run --example dhat_profile --release -- gauss_blur 2048
///
/// Output: `dhat-heap.json` in the current directory.
/// Open it at: https://nnethercote.github.io/dh_view/dh_view.html
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use std::path::{Path, PathBuf};

use viprs::adapters::pipeline::PipelineBuilder;
use viprs::adapters::scheduler::rayon_scheduler::RayonScheduler;
use viprs::adapters::sinks::memory::MemorySink;
use viprs::adapters::sources::memory::MemorySource;
use viprs::domain::format::U8;
use viprs::domain::image::{DemandHint, ImageMetadata, Interpretation};
use viprs::domain::kernel::InterpolationKernel;
use viprs::domain::op::OperationBridge;
use viprs::domain::ops::arithmetic::add::Add;
use viprs::domain::ops::resample::resize::Resize;
use viprs::domain::ops::resample::thumbnail::{Thumbnail, ThumbnailTarget};
use viprs::ports::scheduler::TileScheduler;

type ExampleResult<T> = Result<T, String>;

fn tile_samples(width: u32, height: u32) -> usize {
    let tw = DemandHint::ThinStrip.tile_width(width) as usize;
    let th = DemandHint::ThinStrip.tile_height(width, height) as usize;
    tw * th
}

fn infer_square_size(path: &Path) -> Option<u32> {
    let stem = path.file_stem()?.to_str()?;
    let dims = stem.strip_prefix("bench_")?;
    let (width, height) = dims.split_once('x')?;
    let width = width.parse::<u32>().ok()?;
    let height = height.parse::<u32>().ok()?;
    (width == height).then_some(width)
}

fn render_error(context: &str, err: impl std::fmt::Display) -> String {
    format!("{context}: {err}")
}

fn run_u8_profile(
    width: u32,
    height: u32,
    bands: u32,
    op: &str,
    op_args: &[String],
) -> ExampleResult<()> {
    let pixel_count = (width as usize) * (height as usize) * (bands as usize);
    let pixels = vec![128u8; pixel_count];
    let mut metadata = ImageMetadata::default();
    if bands == 3 {
        metadata.interpretation = Some(Interpretation::Srgb);
    }
    let source = MemorySource::<U8>::new(width, height, bands, pixels)
        .map_err(|err| render_error("failed to create memory source", err))?
        .with_metadata(metadata);
    let builder = PipelineBuilder::from_source(source);

    let builder = match op {
        "invert" => builder
            .invert()
            .map_err(|err| render_error("failed to add invert op", err))?,
        "add" => {
            let rhs = vec![1u8; tile_samples(width, height)];
            let dyn_op = Box::new(OperationBridge::new(Add::<U8>::new(rhs), 1u32));
            builder
                .then(dyn_op)
                .map_err(|err| render_error("failed to add add op", err))?
        }
        "resize" => {
            let scale = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(0.5);
            let resize = Resize::new(scale, scale, InterpolationKernel::Lanczos3);
            builder
                .resize(resize)
                .map_err(|err| render_error("failed to add resize op", err))?
        }
        "thumbnail" => {
            let target_width = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(800);
            let target = ThumbnailTarget::Width(target_width);
            let thumbnail = Thumbnail::new(target, InterpolationKernel::Lanczos3);
            builder
                .thumbnail(thumbnail)
                .map_err(|err| render_error("failed to add thumbnail op", err))?
        }
        "sharpen" => {
            let sigma = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(0.5);
            let strength = op_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3.0);
            builder
                .sharpen(sigma, 2.0, 10.0, 20.0, 0.0, strength)
                .map_err(|err| render_error("failed to add sharpen op", err))?
        }
        other => {
            eprintln!("Unknown op: {other}. Available: invert, add, resize, thumbnail, sharpen");
            std::process::exit(1);
        }
    };

    let pipeline = builder
        .build()
        .map_err(|err| render_error("failed to build pipeline", err))?;
    let mut sink = MemorySink::for_pipeline(&pipeline);
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads())
        .map_err(|err| render_error("failed to create scheduler", err))?;
    for _ in 0..RayonScheduler::default_threads() {
        scheduler
            .run(&pipeline, &mut sink)
            .map_err(|err| render_error("failed to warm up scheduler", err))?;
    }
    let _profiler = dhat::Profiler::new_heap();
    scheduler
        .run(&pipeline, &mut sink)
        .map_err(|err| render_error("failed to execute profiled run", err))?;
    Ok(())
}

fn run_invert(size: u32) -> ExampleResult<()> {
    run_u8_profile(size, size, 1, "invert", &[])
}

fn run_add(size: u32) -> ExampleResult<()> {
    run_u8_profile(size, size, 1, "add", &[])
}

enum ProfileMode {
    InputFile {
        path: PathBuf,
        op: String,
        op_args: Vec<String>,
    },
    Synthetic {
        op: String,
        size: u32,
    },
}

fn is_supported_op(op: &str) -> bool {
    matches!(
        op,
        "invert" | "add" | "resize" | "thumbnail" | "sharpen" | "gauss_blur"
    )
}

fn parse_mode(args: &[String]) -> ProfileMode {
    match args {
        [op, rest @ ..] if is_supported_op(op) => ProfileMode::Synthetic {
            op: op.clone(),
            size: rest.first().and_then(|s| s.parse().ok()).unwrap_or(2048),
        },
        [input, op, rest @ ..] if is_supported_op(op) => ProfileMode::InputFile {
            path: PathBuf::from(input),
            op: op.clone(),
            op_args: rest.to_vec(),
        },
        _ => {
            eprintln!("Usage:");
            eprintln!("  cargo run --example dhat_profile --release -- <input> <op> [op_args...]");
            eprintln!("  cargo run --example dhat_profile --release -- invert 2048");
            eprintln!("  cargo run --example dhat_profile --release -- gauss_blur 2048");
            std::process::exit(1);
        }
    }
}

fn run_gauss_blur(size: u32) -> ExampleResult<()> {
    let pixel_count = (size as usize) * (size as usize) * 3;
    let pixels = vec![128u8; pixel_count];

    let source = MemorySource::<U8>::new(size, size, 3, pixels)
        .map_err(|err| render_error("failed to create gauss blur source", err))?;
    let pipeline = PipelineBuilder::from_source(source)
        .gauss_blur(1.5)
        .map_err(|err| render_error("failed to add gauss_blur op", err))?
        .build()
        .map_err(|err| render_error("failed to build gauss_blur pipeline", err))?;
    let mut sink = MemorySink::for_pipeline(&pipeline);
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads())
        .map_err(|err| render_error("failed to create scheduler", err))?;
    for _ in 0..RayonScheduler::default_threads() {
        scheduler
            .run(&pipeline, &mut sink)
            .map_err(|err| render_error("failed to warm up gauss_blur scheduler", err))?;
    }
    let _profiler = dhat::Profiler::new_heap();
    scheduler
        .run(&pipeline, &mut sink)
        .map_err(|err| render_error("failed to execute gauss_blur run", err))?;
    Ok(())
}

fn run() -> ExampleResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_mode(&args) {
        ProfileMode::Synthetic { op, size } => {
            println!("Running {op} at {size}x{size}...");
            match op.as_str() {
                "invert" => run_invert(size)?,
                "add" => run_add(size)?,
                "gauss_blur" => run_gauss_blur(size)?,
                "resize" | "thumbnail" | "sharpen" => {
                    eprintln!(
                        "Synthetic mode only supports invert/add/gauss_blur. Use <input> <op> [op_args...] for {op}."
                    );
                    std::process::exit(1);
                }
                _ => unreachable!(),
            }
        }
        ProfileMode::InputFile { path, op, op_args } => {
            if op == "gauss_blur" {
                eprintln!(
                    "gauss_blur only supports synthetic mode. Use: cargo run --example dhat_profile --release -- gauss_blur 2048"
                );
                std::process::exit(1);
            }

            let size = infer_square_size(&path).unwrap_or(2048);
            println!(
                "Running {op} on synthetic {size}x{size} RGB input for {} with args {:?}...",
                path.display(),
                op_args
            );
            run_u8_profile(size, size, 3, &op, &op_args)?;
        }
    }

    println!("Done. Profile written to dhat-heap.json");
    println!("View at: https://nnethercote.github.io/dh_view/dh_view.html");
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
