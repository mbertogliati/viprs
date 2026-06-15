use std::{
    process::Command,
    time::{Duration, Instant},
};

use libc::{RUSAGE_SELF, getrusage, rusage};
use viprs::{
    BandFormatId,
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{ColorspaceId, Lab},
        format::U8,
        kernel::InterpolationKernel,
        ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
    },
    ports::scheduler::TileScheduler,
};

const SUMMARY_SAMPLES: usize = 21;
const WARMUP_SAMPLES: usize = 3;
const TARGET_WIDTH: u32 = 400;
const STANDARD_SIZES: [u32; 3] = [512, 2048, 8192];

type PipelineFactory = fn(u32) -> viprs::adapters::pipeline::CompiledPipeline;

fn current_rusage() -> rusage {
    // SAFETY: `rusage` is plain old data and `getrusage(RUSAGE_SELF, ...)` fully initializes it.
    unsafe {
        let mut usage: rusage = std::mem::zeroed();
        getrusage(RUSAGE_SELF, &mut usage);
        usage
    }
}

fn percentile(sorted: &[Duration], percentile: f64) -> Duration {
    let index = ((sorted.len().saturating_sub(1)) as f64 * percentile).round() as usize;
    sorted[index.min(sorted.len().saturating_sub(1))]
}

fn rgb_pixels(size: u32) -> Vec<u8> {
    let samples = size as usize * size as usize * 3;
    (0..samples)
        .map(|index| ((index * 31 + 17) % 255) as u8)
        .collect()
}

fn grayscale_pixels(size: u32) -> Vec<u8> {
    let samples = size as usize * size as usize;
    (0..samples)
        .map(|index| ((index * 13 + 29) % 255) as u8)
        .collect()
}

fn build_invert_invert(size: u32) -> viprs::adapters::pipeline::CompiledPipeline {
    let source = MemorySource::<U8>::new(size, size, 1, grayscale_pixels(size)).unwrap();
    PipelineBuilder::from_source(source)
        .invert()
        .unwrap()
        .invert()
        .unwrap()
        .build()
        .unwrap()
}

fn build_thumbnail_sharpen(size: u32) -> viprs::adapters::pipeline::CompiledPipeline {
    let source = MemorySource::<U8>::new(size, size, 3, rgb_pixels(size)).unwrap();
    PipelineBuilder::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(TARGET_WIDTH),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .sharpen(0.5, 2.0, 10.0, 20.0, 0.0, 3.0)
        .unwrap()
        .build()
        .unwrap()
}

fn build_thumbnail_colourspace_cast(size: u32) -> viprs::adapters::pipeline::CompiledPipeline {
    let source = MemorySource::<U8>::new(size, size, 3, rgb_pixels(size)).unwrap();
    PipelineBuilder::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(TARGET_WIDTH),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .colourspace::<Lab>()
        .unwrap()
        .cast(BandFormatId::U8)
        .unwrap()
        .build()
        .unwrap()
}

fn measure_summary(name: &str, size: u32, build: PipelineFactory) {
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
    let before = current_rusage();
    let mut samples = Vec::with_capacity(SUMMARY_SAMPLES);

    for _ in 0..WARMUP_SAMPLES {
        let pipeline = build(size);
        let mut sink = MemorySink::for_pipeline(&pipeline);
        scheduler.run(&pipeline, &mut sink).unwrap();
        let _ = sink.into_buffer();
    }

    for _ in 0..SUMMARY_SAMPLES {
        let pipeline = build(size);
        let mut sink = MemorySink::for_pipeline(&pipeline);
        let started = Instant::now();
        scheduler.run(&pipeline, &mut sink).unwrap();
        let buffer = sink.into_buffer();
        std::hint::black_box(buffer);
        samples.push(started.elapsed());
    }

    samples.sort_unstable();
    let after = current_rusage();
    println!(
        "multi-op summary {name} size={size}: p50={:.2}ms p95={:.2}ms peak_rss={}KB",
        percentile(&samples, 0.50).as_secs_f64() * 1_000.0,
        percentile(&samples, 0.95).as_secs_f64() * 1_000.0,
        (after.ru_maxrss.max(before.ru_maxrss) / 1024) as u64
    );
}

fn scenario_factory(name: &str) -> Option<PipelineFactory> {
    match name {
        "invert_invert" => Some(build_invert_invert),
        "thumbnail_sharpen" => Some(build_thumbnail_sharpen),
        "thumbnail_colourspace_cast" => Some(build_thumbnail_colourspace_cast),
        _ => None,
    }
}

fn run_single_summary(name: &str, size: u32) {
    let build = scenario_factory(name).unwrap_or_else(|| panic!("unknown scenario '{name}'"));
    measure_summary(name, size, build);
}

fn run_in_subprocess(name: &str, size: u32) {
    let current_exe =
        std::env::current_exe().unwrap_or_else(|error| panic!("current_exe failed: {error}"));
    let status = Command::new(current_exe)
        .args(["--scenario", name, "--size", &size.to_string()])
        .status()
        .unwrap_or_else(|error| {
            panic!("failed to spawn benchmark child for {name}/{size}: {error}")
        });
    assert!(status.success(), "benchmark child failed for {name}/{size}");
}

fn main() {
    let mut args = std::env::args().skip(1);
    if matches!(args.next().as_deref(), Some("--scenario")) {
        let scenario = args.next().expect("missing scenario name");
        assert_eq!(args.next().as_deref(), Some("--size"), "expected --size");
        let size: u32 = args
            .next()
            .expect("missing scenario size")
            .parse()
            .expect("invalid scenario size");
        run_single_summary(&scenario, size);
        return;
    }

    let scenarios: &[(&str, PipelineFactory)] = &[
        ("invert_invert", build_invert_invert),
        ("thumbnail_sharpen", build_thumbnail_sharpen),
        (
            "thumbnail_colourspace_cast",
            build_thumbnail_colourspace_cast,
        ),
    ];

    for (name, build) in scenarios {
        let _ = build;
        for &size in &STANDARD_SIZES {
            run_in_subprocess(name, size);
        }
    }
}
