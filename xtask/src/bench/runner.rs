use std::hint::black_box;
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::{Command, Output};
use std::thread;
use std::time::Instant;

use bytemuck::{Pod, cast_slice};
use serde::Serialize;
use viprs::ImageCodecExt;
use viprs::adapters::codecs::{
    AvifCodec, ExrCodec, GifCodec, HeifCodec, Jp2kCodec, JpegCodec, PngCodec, TiffCodec, WebpCodec,
};
use viprs::adapters::scheduler::rayon_scheduler::RayonScheduler;
use viprs::adapters::sinks::{discard::DiscardSink, memory::MemorySink};
use viprs::domain::codec_options::{LoadOptions, SaveOptions};
use viprs::domain::draw::DrawOp;
use viprs::domain::format::{BandFormat, BandFormatId, F32, U8, U16};
use viprs::domain::image::{ImageMetadata, InMemoryImage, Region};
use viprs::domain::kernel::InterpolationKernel;
use viprs::domain::op::DynOperation;
use viprs::domain::ops::conversion::BlendMode;
use viprs::domain::ops::draw::DrawLineOp;
use viprs::domain::ops::histogram::HistFindOp;
use viprs::domain::ops::resample::mapim::{MapImExtend, MapImOp};
use viprs::domain::ops::resample::thumbnail::{Thumbnail, ThumbnailNode, ThumbnailTarget};
use viprs::ports::codec::ImageEncoder;
use viprs::ports::scheduler::{ReducingScheduler, TileScheduler};

use super::helpers::{
    INPUT_DIVERSITY_SUPPORTED_OPS, WARMUP_ITERATIONS, append_trend, bench_fixtures_for_op,
    bench_result_percentiles, build_summary_row, encode_tiff_with_input, getrusage, git_sha,
    is_workflow_like_op, iso_timestamp, libvips_backend_label, load_bench_image,
    load_bench_image_with_options, load_tiff_save_input, parse_save_tiff_compression, percentile,
    scenario_display_label, scenario_set, scenario_set_display_label, scenario_set_supports_op,
    scenario_slug, viprs_backend_label, workflow_op_args_for_scenario,
};
use super::pipeline::{
    build_viprs_composite_pipeline, build_viprs_composite_pipeline_from_preloaded,
    build_viprs_e2e_pipeline, build_viprs_pipeline, build_viprs_pipeline_from_preloaded,
    preload_bench_source,
};
use super::types::{BenchImage, BenchResult, Comparison, Ratios, SummaryRow, TrendRecord};
use crate::common::repo_root;

const DEFAULT_MAPIM_DX: f32 = 0.25;
const DEFAULT_MAPIM_DY: f32 = 0.25;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BaselineBackend {
    Libvips,
    OpenExr,
}

impl BaselineBackend {
    pub(crate) fn for_op(op: &str) -> Self {
        if op == "save-exr" {
            Self::OpenExr
        } else {
            Self::Libvips
        }
    }

    fn runner_name(self) -> &'static str {
        match self {
            Self::Libvips => "libvips-runner",
            Self::OpenExr => "openexr-runner",
        }
    }

    fn runner_path(self) -> std::path::PathBuf {
        repo_root()
            .join("tools/bench-vs-libvips")
            .join(self.runner_name())
    }

    fn display_label(self) -> String {
        match self {
            Self::Libvips => libvips_backend_label(),
            Self::OpenExr => "openexr (encode baseline)".to_owned(),
        }
    }
}

fn is_load_benchmark(op: &str) -> bool {
    matches!(
        op,
        "load"
            | "load-avif"
            | "load-exr"
            | "load-heif"
            | "load-jpeg"
            | "load-pdf"
            | "load-svg"
            | "load-tiff"
    )
}

// REASON: Draw-line benchmark scaffolding, not yet wired into the CLI.
#[allow(dead_code)]
fn run_draw_line_once_u8(width: u32, height: u32, bands: u32) {
    let mut pixels = vec![0u8; width as usize * height as usize * bands as usize];
    let mut tile =
        viprs::domain::image::TileMut::new(Region::new(0, 0, width, height), bands, &mut pixels);
    let op = DrawLineOp::<U8>::new(
        0,
        (height / 2) as i32,
        width.saturating_sub(1) as i32,
        (height / 2) as i32,
        vec![0u8; bands as usize],
    )
    .expect("draw_line op");
    op.draw(&mut tile);
    black_box(pixels);
}

// REASON: Draw-line benchmark scaffolding, not yet wired into the CLI.
#[allow(dead_code)]
fn run_draw_line_once_u16(width: u32, height: u32, bands: u32) {
    let mut pixels = vec![0u16; width as usize * height as usize * bands as usize];
    let mut tile =
        viprs::domain::image::TileMut::new(Region::new(0, 0, width, height), bands, &mut pixels);
    let op = DrawLineOp::<U16>::new(
        0,
        (height / 2) as i32,
        width.saturating_sub(1) as i32,
        (height / 2) as i32,
        vec![0u16; bands as usize],
    )
    .expect("draw_line op");
    op.draw(&mut tile);
    black_box(pixels);
}

// REASON: Draw-line benchmark scaffolding, not yet wired into the CLI.
#[allow(dead_code)]
fn run_draw_line_once(format: BandFormatId, width: u32, height: u32, bands: u32) {
    match format {
        BandFormatId::U8 => run_draw_line_once_u8(width, height, bands),
        BandFormatId::U16 => run_draw_line_once_u16(width, height, bands),
        other => {
            eprintln!("draw_line benchmark only supports u8/u16 fixtures, got {other:?}");
            std::process::exit(1);
        }
    }
}

// REASON: Draw-line benchmark scaffolding, not yet wired into the CLI.
#[allow(dead_code)]
fn run_viprs_draw_line_bench(input: &Path, iterations: usize, e2e: bool) -> BenchResult {
    let input_str = input.to_string_lossy().to_string();

    let warmup = || {
        if e2e {
            match load_bench_image(input) {
                BenchImage::U8(image) => run_draw_line_once(
                    BandFormatId::U8,
                    image.width(),
                    image.height(),
                    image.bands(),
                ),
                BenchImage::U16(image) => run_draw_line_once(
                    BandFormatId::U16,
                    image.width(),
                    image.height(),
                    image.bands(),
                ),
                BenchImage::F32(image) => run_draw_line_once(
                    BandFormatId::F32,
                    image.width(),
                    image.height(),
                    image.bands(),
                ),
            }
        }
    };

    if e2e {
        for _ in 0..WARMUP_ITERATIONS {
            warmup();
        }

        let ru_before = getrusage();
        let mut wall_ns = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            match load_bench_image(input) {
                BenchImage::U8(image) => run_draw_line_once(
                    BandFormatId::U8,
                    image.width(),
                    image.height(),
                    image.bands(),
                ),
                BenchImage::U16(image) => run_draw_line_once(
                    BandFormatId::U16,
                    image.width(),
                    image.height(),
                    image.bands(),
                ),
                BenchImage::F32(image) => run_draw_line_once(
                    BandFormatId::F32,
                    image.width(),
                    image.height(),
                    image.bands(),
                ),
            }
            wall_ns.push(start.elapsed().as_nanos() as u64);
        }
        let ru_after = getrusage();

        return BenchResult {
            backend: viprs_backend_label(),
            input: input_str,
            operation: "draw_line".into(),
            iterations,
            wall_ns,
            peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
            minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
            major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
            vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
            invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
        };
    }

    let image = load_bench_image(input);
    let (format, width, height, bands) = match image {
        BenchImage::U8(image) => (
            BandFormatId::U8,
            image.width(),
            image.height(),
            image.bands(),
        ),
        BenchImage::U16(image) => (
            BandFormatId::U16,
            image.width(),
            image.height(),
            image.bands(),
        ),
        BenchImage::F32(image) => (
            BandFormatId::F32,
            image.width(),
            image.height(),
            image.bands(),
        ),
    };

    for _ in 0..WARMUP_ITERATIONS {
        run_draw_line_once(format, width, height, bands);
    }

    let ru_before = getrusage();
    let mut wall_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        run_draw_line_once(format, width, height, bands);
        wall_ns.push(start.elapsed().as_nanos() as u64);
    }
    let ru_after = getrusage();

    BenchResult {
        backend: viprs_backend_label(),
        input: input_str,
        operation: "draw_line".into(),
        iterations,
        wall_ns,
        peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
        minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
        major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
        vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
        invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
    }
}

fn run_histogram_once(
    scheduler: &RayonScheduler,
    pipeline: &viprs::adapters::pipeline::CompiledPipeline,
    image: &BenchImage,
    e2e: bool,
) {
    match image {
        BenchImage::U8(image) => {
            let reducer = HistFindOp::for_format(image.bands(), None, u8::MAX as u32);
            let hist = if e2e {
                let sink = MemorySink::for_pipeline(pipeline)
                    .expect("MemorySink allocation failed: dimensions too large for bench input");
                scheduler
                    .run_with_reducer::<U8, HistFindOp>(pipeline, &sink, &reducer)
                    .expect("histogram run")
            } else {
                let sink = DiscardSink::new();
                scheduler
                    .run_with_reducer::<U8, HistFindOp>(pipeline, &sink, &reducer)
                    .expect("histogram run")
            };
            black_box(hist);
        }
        BenchImage::U16(image) => {
            let reducer = HistFindOp::for_format(image.bands(), None, u16::MAX as u32);
            let hist = if e2e {
                let sink = MemorySink::for_pipeline(pipeline)
                    .expect("MemorySink allocation failed: dimensions too large for bench input");
                scheduler
                    .run_with_reducer::<U16, HistFindOp>(pipeline, &sink, &reducer)
                    .expect("histogram run")
            } else {
                let sink = DiscardSink::new();
                scheduler
                    .run_with_reducer::<U16, HistFindOp>(pipeline, &sink, &reducer)
                    .expect("histogram run")
            };
            black_box(hist);
        }
        BenchImage::F32(image) => {
            let reducer = HistFindOp::for_format(image.bands(), None, u8::MAX as u32);
            let hist = if e2e {
                let sink = MemorySink::for_pipeline(pipeline)
                    .expect("MemorySink allocation failed: dimensions too large for bench input");
                scheduler
                    .run_with_reducer::<F32, HistFindOp>(pipeline, &sink, &reducer)
                    .expect("histogram run")
            } else {
                let sink = DiscardSink::new();
                scheduler
                    .run_with_reducer::<F32, HistFindOp>(pipeline, &sink, &reducer)
                    .expect("histogram run")
            };
            black_box(hist);
        }
    }
}

fn run_viprs_histogram_bench(
    input: &Path,
    iterations: usize,
    threads: usize,
    e2e: bool,
) -> BenchResult {
    let scheduler = RayonScheduler::new(threads).expect("scheduler creation");
    let input_str = input.to_string_lossy().to_string();

    if e2e {
        for _ in 0..WARMUP_ITERATIONS {
            let image = load_bench_image(input);
            let pipeline = build_viprs_e2e_pipeline(input, "histogram", &[]);
            run_histogram_once(&scheduler, &pipeline, &image, true);
        }

        let ru_before = getrusage();
        let mut wall_ns = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let image = load_bench_image(input);
            let pipeline = build_viprs_e2e_pipeline(input, "histogram", &[]);
            run_histogram_once(&scheduler, &pipeline, &image, true);
            wall_ns.push(start.elapsed().as_nanos() as u64);
        }
        let ru_after = getrusage();

        return BenchResult {
            backend: viprs_backend_label(),
            input: input_str,
            operation: "histogram".into(),
            iterations,
            wall_ns,
            peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
            minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
            major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
            vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
            invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
        };
    }

    let source = preload_bench_source(input);
    let preloaded_image = load_bench_image(input);
    for _ in 0..WARMUP_ITERATIONS {
        let pipeline = build_viprs_pipeline_from_preloaded(&source, "histogram", &[]);
        run_histogram_once(&scheduler, &pipeline, &preloaded_image, false);
    }

    let ru_before = getrusage();
    let mut wall_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let pipeline = build_viprs_pipeline_from_preloaded(&source, "histogram", &[]);
        run_histogram_once(&scheduler, &pipeline, &preloaded_image, false);
        wall_ns.push(start.elapsed().as_nanos() as u64);
    }
    let ru_after = getrusage();

    BenchResult {
        backend: viprs_backend_label(),
        input: input_str,
        operation: "histogram".into(),
        iterations,
        wall_ns,
        peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
        minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
        major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
        vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
        invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
    }
}

fn baseline_op_name(op: &str) -> &str {
    if matches!(
        op,
        "load-avif" | "load-exr" | "load-heif" | "load-pdf" | "load-svg"
    ) {
        "load"
    } else {
        op
    }
}

#[cfg(test)]
mod tests {
    use super::{baseline_op_name, is_load_benchmark};

    #[test]
    fn load_jpeg_is_treated_as_load_benchmark() {
        assert!(is_load_benchmark("load-jpeg"));
    }

    #[test]
    fn load_jpeg_uses_own_baseline_op_name() {
        assert_eq!(baseline_op_name("load-jpeg"), "load-jpeg");
    }
}

pub(crate) fn parse_baseline_output(
    backend: BaselineBackend,
    output: &Output,
) -> Result<BenchResult, String> {
    if !output.status.success() {
        return Err(format!(
            "{} failed: {}",
            backend.runner_name(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let json_str = String::from_utf8_lossy(&output.stdout);
    let mut result: BenchResult = serde_json::from_str(&json_str)
        .map_err(|error| format!("{} returned invalid JSON: {error}", backend.runner_name()))?;
    result.backend = backend.display_label();
    Ok(result)
}

fn percentile_u128(sorted: &[u128], p: f64) -> u128 {
    let idx = ((sorted.len() as f64) * p) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn percentiles_ns(samples: &[u128]) -> (f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    (
        percentile_u128(&sorted, 0.50) as f64 / 1e6,
        percentile_u128(&sorted, 0.95) as f64 / 1e6,
    )
}

fn average_u64(samples: impl Iterator<Item = u64>) -> f64 {
    let (sum, count) = samples.fold((0_u64, 0_u64), |(sum, count), sample| {
        (sum + sample, count + 1)
    });
    if count == 0 {
        0.0
    } else {
        sum as f64 / count as f64
    }
}

fn parse_composite_mode(op_args: &[String]) -> BlendMode {
    match op_args.first().map(String::as_str).unwrap_or("over") {
        "over" => BlendMode::Over,
        "atop" => BlendMode::Atop,
        other => {
            eprintln!("composite only accepts optional mode arg 'over' or 'atop', got '{other}'");
            std::process::exit(1);
        }
    }
}

fn thumbnail_stage_label(node: &ThumbnailNode) -> String {
    match node {
        ThumbnailNode::Premultiply => "Premultiply".to_owned(),
        ThumbnailNode::ShrinkH { factor } => format!("ShrinkH x{factor}"),
        ThumbnailNode::ShrinkV { factor } => format!("ShrinkV x{factor}"),
        ThumbnailNode::ReduceH { factor, .. } => format!("ReduceH x{factor:.3}"),
        ThumbnailNode::ReduceV { factor, .. } => format!("ReduceV x{factor:.3}"),
        ThumbnailNode::Affine {
            output_width,
            output_height,
            ..
        } => format!("Affine -> {output_width}x{output_height}"),
        ThumbnailNode::ExtractArea { width, height, .. } => {
            format!("ExtractArea {width}x{height}")
        }
        ThumbnailNode::Unpremultiply => "Unpremultiply".to_owned(),
        ThumbnailNode::Flatten { .. } => "Flatten".to_owned(),
    }
}

fn thumbnail_stage_name(node: &ThumbnailNode) -> &'static str {
    match node {
        ThumbnailNode::Premultiply => "premultiply",
        ThumbnailNode::ShrinkH { .. } => "shrink_h",
        ThumbnailNode::ShrinkV { .. } => "shrink_v",
        ThumbnailNode::ReduceH { .. } => "reduce_h",
        ThumbnailNode::ReduceV { .. } => "reduce_v",
        ThumbnailNode::Affine { .. } => "affine",
        ThumbnailNode::ExtractArea { .. } => "extract_area",
        ThumbnailNode::Unpremultiply => "unpremultiply",
        ThumbnailNode::Flatten { .. } => "flatten",
    }
}

fn thumbnail_stage_nodes(input: &Path, op_args: &[String]) -> Vec<ThumbnailNode> {
    let target_width = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(800);
    let thumbnail = Thumbnail::new(
        ThumbnailTarget::Width(target_width),
        InterpolationKernel::Lanczos3,
    );

    match load_bench_image(input) {
        BenchImage::U8(image) => thumbnail.into_pipeline_nodes_without_shrink_hint(
            image.width(),
            image.height(),
            image.bands(),
        ),
        BenchImage::U16(image) => thumbnail.into_pipeline_nodes_without_shrink_hint(
            image.width(),
            image.height(),
            image.bands(),
        ),
        BenchImage::F32(image) => thumbnail.into_pipeline_nodes_without_shrink_hint(
            image.width(),
            image.height(),
            image.bands(),
        ),
    }
    .nodes
}

#[derive(Serialize)]
struct ThumbnailStageJson {
    name: String,
    p50_ms: f64,
    p95_ms: f64,
    exec_per_run: f64,
    cache_hits_per_run: f64,
}

#[derive(Serialize)]
struct SourceReadJson {
    p50_ms: f64,
    p95_ms: f64,
    reads_per_run: f64,
}

#[derive(Serialize)]
struct SinkWriteJson {
    p50_ms: f64,
    p95_ms: f64,
}

#[derive(Serialize)]
struct SchedulingJson {
    p50_ms: f64,
    p95_ms: f64,
}

#[derive(Serialize)]
struct ThumbnailStageProfileJson {
    #[serde(rename = "type")]
    profile_type: &'static str,
    operation: &'static str,
    total_p50_ms: f64,
    total_p95_ms: f64,
    tiles_per_run: f64,
    recomputation_detected: bool,
    stages: Vec<ThumbnailStageJson>,
    source_read: SourceReadJson,
    sink_write: SinkWriteJson,
    scheduling: SchedulingJson,
}

pub fn print_thumbnail_stage_profile(
    input: &Path,
    op_args: &[String],
    iterations: usize,
    threads: usize,
    ai_output: bool,
) {
    let scheduler = RayonScheduler::new(threads).expect("scheduler creation");
    let thumbnail_nodes = thumbnail_stage_nodes(input, op_args);
    let mut stage_labels: Vec<String> = thumbnail_nodes.iter().map(thumbnail_stage_label).collect();
    let mut stage_names: Vec<String> = thumbnail_nodes
        .iter()
        .map(thumbnail_stage_name)
        .map(str::to_owned)
        .collect();
    let node_count = build_viprs_pipeline(input, "thumbnail", op_args)
        .nodes
        .len();
    if stage_labels.len() != node_count {
        stage_labels = (0..node_count).map(|idx| format!("node[{idx}]")).collect();
        stage_names = stage_labels.clone();
    }

    for _ in 0..WARMUP_ITERATIONS {
        let pipeline = build_viprs_pipeline(input, "thumbnail", op_args);
        let mut sink = MemorySink::for_pipeline(&pipeline)
            .expect("MemorySink allocation failed: dimensions too large for bench input");
        let _ = scheduler
            .run_with_profile(&pipeline, &mut sink)
            .expect("thumbnail stage profile warmup");
    }

    let mut runs = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let pipeline = build_viprs_pipeline(input, "thumbnail", op_args);
        let mut sink = MemorySink::for_pipeline(&pipeline)
            .expect("MemorySink allocation failed: dimensions too large for bench input");
        runs.push(
            scheduler
                .run_with_profile(&pipeline, &mut sink)
                .expect("thumbnail stage profile run"),
        );
    }

    let total_samples: Vec<u128> = runs.iter().map(|run| run.total_ns).collect();
    let source_samples: Vec<u128> = runs.iter().map(|run| run.source_read_ns).collect();
    let sink_samples: Vec<u128> = runs.iter().map(|run| run.sink_write_ns).collect();
    let scheduling_samples: Vec<u128> = runs
        .iter()
        .map(|run| {
            let node_total = run
                .nodes
                .iter()
                .fold(0_u128, |sum, node| sum + node.process_ns);
            run.total_ns
                .saturating_sub(node_total)
                .saturating_sub(run.source_read_ns)
                .saturating_sub(run.sink_write_ns)
        })
        .collect();

    let (total_p50, total_p95) = percentiles_ns(&total_samples);
    let (source_p50, source_p95) = percentiles_ns(&source_samples);
    let (sink_p50, sink_p95) = percentiles_ns(&sink_samples);
    let (sched_p50, sched_p95) = percentiles_ns(&scheduling_samples);
    let tiles_per_run = average_u64(runs.iter().map(|run| run.tile_count));
    let source_reads_per_run = average_u64(runs.iter().map(|run| run.source_read_count));
    let recomputation_detected = runs.iter().any(|run| {
        run.source_read_count > run.tile_count
            || run
                .nodes
                .iter()
                .any(|node| node.exec_count > run.tile_count)
    });

    if ai_output {
        let stages = stage_names
            .iter()
            .enumerate()
            .map(|(idx, name)| {
                let samples: Vec<u128> = runs.iter().map(|run| run.nodes[idx].process_ns).collect();
                let (p50, p95) = percentiles_ns(&samples);
                ThumbnailStageJson {
                    name: name.clone(),
                    p50_ms: p50,
                    p95_ms: p95,
                    exec_per_run: average_u64(runs.iter().map(|run| run.nodes[idx].exec_count)),
                    cache_hits_per_run: average_u64(
                        runs.iter().map(|run| run.nodes[idx].cache_hits),
                    ),
                }
            })
            .collect();
        let profile = ThumbnailStageProfileJson {
            profile_type: "stage_profile",
            operation: "thumbnail",
            total_p50_ms: total_p50,
            total_p95_ms: total_p95,
            tiles_per_run,
            recomputation_detected,
            stages,
            source_read: SourceReadJson {
                p50_ms: source_p50,
                p95_ms: source_p95,
                reads_per_run: source_reads_per_run,
            },
            sink_write: SinkWriteJson {
                p50_ms: sink_p50,
                p95_ms: sink_p95,
            },
            scheduling: SchedulingJson {
                p50_ms: sched_p50,
                p95_ms: sched_p95,
            },
        };

        match serde_json::to_string_pretty(&profile) {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("Failed to serialize thumbnail stage-profile JSON: {error}");
                std::process::exit(1);
            }
        }
        return;
    }

    println!("--- viprs thumbnail stage profile ---");
    println!(
        "  total           p50: {:.2} ms  p95: {:.2} ms",
        total_p50, total_p95
    );
    for (idx, label) in stage_labels.iter().enumerate() {
        let samples: Vec<u128> = runs.iter().map(|run| run.nodes[idx].process_ns).collect();
        let (p50, p95) = percentiles_ns(&samples);
        let execs = average_u64(runs.iter().map(|run| run.nodes[idx].exec_count));
        let cache_hits = average_u64(runs.iter().map(|run| run.nodes[idx].cache_hits));
        println!(
            "  {:<14} p50: {:>6.2} ms  p95: {:>6.2} ms  exec/run: {:.1}  cache hits/run: {:.1}",
            label, p50, p95, execs, cache_hits
        );
    }
    println!(
        "  source-read     p50: {:>6.2} ms  p95: {:>6.2} ms  reads/run: {:.1}",
        source_p50, source_p95, source_reads_per_run
    );
    println!(
        "  sink-write      p50: {:>6.2} ms  p95: {:>6.2} ms",
        sink_p50, sink_p95
    );
    println!(
        "  scheduling      p50: {:>6.2} ms  p95: {:>6.2} ms  tiles/run: {:.1}",
        sched_p50, sched_p95, tiles_per_run
    );
    println!(
        "  recomputation:  {}",
        if recomputation_detected {
            "possible upstream recomputation detected"
        } else {
            "none detected (source reads and node executions stay at one per tile)"
        }
    );
    println!();
}

pub fn run_viprs_profile_only(
    input: &Path,
    op: &str,
    op_args: &[String],
    iterations: usize,
    threads: usize,
) {
    if op == "mapim" {
        let (dx, dy) = parse_mapim_offsets(op_args);
        let image = load_bench_image(input);
        let (width, height) = match &image {
            BenchImage::U8(image) => (image.width(), image.height()),
            BenchImage::U16(image) => (image.width(), image.height()),
            BenchImage::F32(image) => (image.width(), image.height()),
        };
        let index = build_mapim_index(width, height, dx, dy);
        for _ in 0..WARMUP_ITERATIONS {
            run_mapim_for_bench_image(&image, &index, threads);
        }
        for _ in 0..iterations {
            run_mapim_for_bench_image(&image, &index, threads);
        }
        return;
    }

    if op == "composite" {
        let mode = parse_composite_mode(op_args);
        let scheduler = RayonScheduler::new(threads).expect("scheduler creation");
        let source = preload_bench_source(input);
        for _ in 0..WARMUP_ITERATIONS {
            let pipeline = build_viprs_composite_pipeline_from_preloaded(&source, mode);
            let mut sink = DiscardSink::new();
            scheduler
                .run(&pipeline, &mut sink)
                .expect("composite warmup");
        }
        for _ in 0..iterations {
            let pipeline = build_viprs_composite_pipeline_from_preloaded(&source, mode);
            let mut sink = DiscardSink::new();
            scheduler.run(&pipeline, &mut sink).expect("composite run");
        }
        return;
    }

    if is_workflow_like_op(op) {
        let scheduler = RayonScheduler::new(threads).expect("scheduler creation");
        let source = preload_bench_source(input);

        for _ in 0..WARMUP_ITERATIONS {
            let pipeline = build_viprs_pipeline_from_preloaded(&source, op, op_args);
            let mut sink = DiscardSink::new();
            scheduler
                .run(&pipeline, &mut sink)
                .expect("workflow warmup");
        }
        for _ in 0..iterations {
            let pipeline = build_viprs_pipeline_from_preloaded(&source, op, op_args);
            let mut sink = DiscardSink::new();
            scheduler.run(&pipeline, &mut sink).expect("workflow run");
        }
        return;
    }

    if matches!(
        op,
        "load"
            | "load-avif"
            | "load-tiff"
            | "load-exr"
            | "load-heif"
            | "load-pdf"
            | "load-svg"
            | "save-avif"
            | "save-exr"
            | "save-gif"
            | "save-heif"
            | "save-jp2k"
            | "save-tiff"
    ) {
        run_viprs_special_profile_only(input, op, op_args, iterations);
        return;
    }

    let scheduler = RayonScheduler::new(threads).expect("scheduler creation");
    let source = preload_bench_source(input);
    for _ in 0..WARMUP_ITERATIONS {
        let pipeline = build_viprs_pipeline_from_preloaded(&source, op, op_args);
        let mut sink = DiscardSink::new();
        scheduler.run(&pipeline, &mut sink).expect("warmup");
    }
    for _ in 0..iterations {
        let pipeline = build_viprs_pipeline_from_preloaded(&source, op, op_args);
        let mut sink = DiscardSink::new();
        scheduler.run(&pipeline, &mut sink).expect("run");
    }
}

fn run_viprs_special_profile_only(input: &Path, op: &str, op_args: &[String], iterations: usize) {
    let is_exr = input
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("exr"));
    let is_save_avif = op == "save-avif";
    let is_save_exr = op == "save-exr";
    let is_save_gif = op == "save-gif";
    let is_save_heif = op == "save-heif";
    let is_save_jp2k = op == "save-jp2k";
    let is_save_tiff = op == "save-tiff";

    let save_avif_format = if is_save_avif {
        match op_args.first().map(String::as_str).unwrap_or("u8") {
            "u8" => Some("u8"),
            "u16" => Some("u16"),
            other => {
                eprintln!(
                    "save-avif only accepts optional bit depth arg 'u8' or 'u16', got '{other}'"
                );
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    let save_tiff_compression = is_save_tiff.then(|| parse_save_tiff_compression(op_args));

    if save_avif_format == Some("u16") && is_exr {
        eprintln!("save-avif u16 expects an integer input image, not EXR");
        std::process::exit(1);
    }
    if is_save_gif && is_exr {
        eprintln!("save-gif expects an integer input image, not EXR");
        std::process::exit(1);
    }
    if is_save_heif && is_exr {
        eprintln!("save-heif expects an integer input image, not EXR");
        std::process::exit(1);
    }
    if is_save_jp2k && is_exr {
        eprintln!("save-jp2k expects an integer input image, not EXR");
        std::process::exit(1);
    }
    if is_save_tiff && is_exr {
        eprintln!("save-tiff expects an integer input image, not EXR");
        std::process::exit(1);
    }

    let avif_codec = if is_save_avif { Some(AvifCodec) } else { None };
    let exr_codec = if is_save_exr { Some(ExrCodec) } else { None };
    let gif_codec = if is_save_gif {
        Some(GifCodec::default())
    } else {
        None
    };
    let heif_codec = if is_save_heif { Some(HeifCodec) } else { None };
    let jp2k_codec = if is_save_jp2k { Some(Jp2kCodec) } else { None };
    let tiff_codec = if is_save_tiff {
        Some(TiffCodec::default())
    } else {
        None
    };
    let save_tiff_options = save_tiff_compression
        .map(|compression| SaveOptions::default().with_tiff_compression(compression));
    let preloaded_u8 =
        if (save_avif_format == Some("u8") || is_save_gif || is_save_heif || is_save_jp2k)
            && !is_load_benchmark(op)
        {
            Some(
                InMemoryImage::<U8>::load(input)
                    .expect("Failed to pre-load integer image for encode benchmark"),
            )
        } else {
            None
        };
    let preloaded_u16 = if save_avif_format == Some("u16") {
        Some(
            InMemoryImage::<U16>::load(input)
                .expect("Failed to pre-load image for 16-bit AVIF encode"),
        )
    } else {
        None
    };
    let preloaded_f32 = if is_save_exr {
        Some(InMemoryImage::<F32>::load(input).expect("Failed to pre-load image for EXR encode"))
    } else {
        None
    };
    let preloaded_tiff = if is_save_tiff {
        Some(load_tiff_save_input(input))
    } else {
        None
    };

    let run_iter = || {
        if let Some(codec) = avif_codec.as_ref() {
            if let Some(image) = preloaded_u8.as_ref() {
                black_box(codec.encode(image).expect("Failed to encode AVIF"));
            } else if let Some(image) = preloaded_u16.as_ref() {
                black_box(codec.encode(image).expect("Failed to encode 16-bit AVIF"));
            }
        } else if let Some(codec) = gif_codec.as_ref() {
            black_box(
                codec
                    .encode(preloaded_u8.as_ref().expect("missing preloaded GIF input"))
                    .expect("Failed to encode GIF"),
            );
        } else if let Some(codec) = heif_codec.as_ref() {
            black_box(
                codec
                    .encode(preloaded_u8.as_ref().expect("missing preloaded HEIF input"))
                    .expect("Failed to encode HEIF"),
            );
        } else if let Some(codec) = exr_codec.as_ref() {
            black_box(
                codec
                    .encode(preloaded_f32.as_ref().expect("missing preloaded EXR input"))
                    .expect("Failed to encode EXR"),
            );
        } else if let Some(codec) = jp2k_codec.as_ref() {
            black_box(
                codec
                    .encode(preloaded_u8.as_ref().expect("missing preloaded JP2K input"))
                    .expect("Failed to encode JPEG 2000"),
            );
        } else if let Some(codec) = tiff_codec.as_ref() {
            let opts = save_tiff_options
                .as_ref()
                .expect("missing TIFF save options");
            let encoded = encode_tiff_with_input(
                codec,
                preloaded_tiff
                    .as_ref()
                    .expect("missing preloaded TIFF input"),
                opts,
            );
            black_box(encoded);
        } else {
            match load_bench_image(input) {
                BenchImage::U8(image) => {
                    black_box(image);
                }
                BenchImage::U16(image) => {
                    black_box(image);
                }
                BenchImage::F32(image) => {
                    black_box(image);
                }
            };
        }
    };

    for _ in 0..WARMUP_ITERATIONS {
        run_iter();
    }
    for _ in 0..iterations {
        run_iter();
    }
}

pub fn run_viprs_bench(
    input: &Path,
    op: &str,
    op_args: &[String],
    iterations: usize,
    threads: usize,
    e2e: bool,
) -> BenchResult {
    if op == "histogram" {
        return run_viprs_histogram_bench(input, iterations, threads, e2e);
    }
    if op == "mapim" {
        return run_viprs_mapim_bench(input, op_args, iterations, threads, e2e);
    }
    if op == "composite" {
        return run_viprs_composite_bench(input, op_args, iterations, threads, e2e);
    }

    let input_str = input.to_string_lossy().to_string();

    if matches!(
        op,
        "load"
            | "load-avif"
            | "load-tiff"
            | "load-exr"
            | "load-heif"
            | "load-pdf"
            | "load-svg"
            | "save-avif"
            | "save-exr"
            | "save-gif"
            | "save-heif"
            | "save-jpeg"
            | "save-jp2k"
            | "save-tiff"
    ) {
        let is_exr = input
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("exr"));
        let is_save_avif = op == "save-avif";
        let is_save_exr = op == "save-exr";
        let is_save_gif = op == "save-gif";
        let is_save_heif = op == "save-heif";
        let is_save_jpeg = op == "save-jpeg";
        let is_save_jp2k = op == "save-jp2k";
        let is_save_tiff = op == "save-tiff";
        let save_jpeg_options = is_save_jpeg.then(|| SaveOptions::default().with_quality(85));

        let save_avif_format = if is_save_avif {
            match op_args.first().map(String::as_str).unwrap_or("u8") {
                "u8" => Some("u8"),
                "u16" => Some("u16"),
                other => {
                    eprintln!(
                        "save-avif only accepts optional bit depth arg 'u8' or 'u16', got '{other}'"
                    );
                    std::process::exit(1);
                }
            }
        } else {
            None
        };
        let save_tiff_compression = is_save_tiff.then(|| parse_save_tiff_compression(op_args));

        if save_avif_format == Some("u16") && is_exr {
            eprintln!("save-avif u16 expects an integer input image, not EXR");
            std::process::exit(1);
        }

        if is_save_gif && is_exr {
            eprintln!("save-gif expects an integer input image, not EXR");
            std::process::exit(1);
        }

        if is_save_heif && is_exr {
            eprintln!("save-heif expects an integer input image, not EXR");
            std::process::exit(1);
        }

        if is_save_jp2k && is_exr {
            eprintln!("save-jp2k expects an integer input image, not EXR");
            std::process::exit(1);
        }

        if is_save_tiff && is_exr {
            eprintln!("save-tiff expects an integer input image, not EXR");
            std::process::exit(1);
        }

        let avif_codec = if is_save_avif { Some(AvifCodec) } else { None };
        let exr_codec = if is_save_exr { Some(ExrCodec) } else { None };
        let gif_codec = if is_save_gif {
            Some(GifCodec::default())
        } else {
            None
        };
        let heif_codec = if is_save_heif { Some(HeifCodec) } else { None };
        let jpeg_codec = if is_save_jpeg { Some(JpegCodec) } else { None };
        let jp2k_codec = if is_save_jp2k { Some(Jp2kCodec) } else { None };
        let tiff_codec = if is_save_tiff {
            Some(TiffCodec::default())
        } else {
            None
        };
        let save_tiff_options = save_tiff_compression
            .map(|compression| SaveOptions::default().with_tiff_compression(compression));
        let load_options = LoadOptions::default()
            .with_decoder_threads(NonZeroUsize::new(threads).expect("threads must be non-zero"));
        let preloaded_u8 = if (save_avif_format == Some("u8")
            || is_save_gif
            || is_save_heif
            || is_save_jpeg
            || is_save_jp2k)
            && !e2e
        {
            Some(
                InMemoryImage::<U8>::load(input)
                    .expect("Failed to pre-load integer image for encode benchmark"),
            )
        } else {
            None
        };
        let preloaded_u16 = if save_avif_format == Some("u16") && !e2e {
            Some(
                InMemoryImage::<U16>::load(input)
                    .expect("Failed to pre-load image for 16-bit AVIF encode"),
            )
        } else {
            None
        };
        let preloaded_f32 = if is_save_exr && !e2e {
            Some(
                InMemoryImage::<F32>::load(input).expect("Failed to pre-load image for EXR encode"),
            )
        } else {
            None
        };
        let preloaded_tiff = if is_save_tiff && !e2e {
            Some(load_tiff_save_input(input))
        } else {
            None
        };

        for _ in 0..WARMUP_ITERATIONS {
            if let Some(codec) = avif_codec.as_ref() {
                if let Some(image) = preloaded_u8.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode AVIF");
                    black_box(encoded);
                } else if let Some(image) = preloaded_u16.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode 16-bit AVIF");
                    black_box(encoded);
                } else {
                    match save_avif_format {
                        Some("u16") => {
                            let image =
                                InMemoryImage::<U16>::load(input).expect("Failed to load image");
                            let encoded =
                                codec.encode(&image).expect("Failed to encode 16-bit AVIF");
                            black_box(encoded);
                        }
                        _ => {
                            let image =
                                InMemoryImage::<U8>::load(input).expect("Failed to load image");
                            let encoded = codec.encode(&image).expect("Failed to encode AVIF");
                            black_box(encoded);
                        }
                    }
                }
            } else if let Some(codec) = gif_codec.as_ref() {
                if let Some(image) = preloaded_u8.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode GIF");
                    black_box(encoded);
                } else {
                    let image = InMemoryImage::<U8>::load(input).expect("Failed to load image");
                    let encoded = codec.encode(&image).expect("Failed to encode GIF");
                    black_box(encoded);
                }
            } else if let Some(codec) = heif_codec.as_ref() {
                if let Some(image) = preloaded_u8.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode HEIF");
                    black_box(encoded);
                } else {
                    let image = InMemoryImage::<U8>::load(input).expect("Failed to load image");
                    let encoded = codec.encode(&image).expect("Failed to encode HEIF");
                    black_box(encoded);
                }
            } else if let Some(codec) = jpeg_codec.as_ref() {
                let opts = save_jpeg_options
                    .as_ref()
                    .expect("missing JPEG save options");
                if let Some(image) = preloaded_u8.as_ref() {
                    let encoded = codec
                        .encode_with_options(image, opts)
                        .expect("Failed to encode JPEG");
                    black_box(encoded);
                } else {
                    let image = InMemoryImage::<U8>::load(input).expect("Failed to load image");
                    let encoded = codec
                        .encode_with_options(&image, opts)
                        .expect("Failed to encode JPEG");
                    black_box(encoded);
                }
            } else if let Some(codec) = exr_codec.as_ref() {
                if let Some(image) = preloaded_f32.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode EXR");
                    black_box(encoded);
                } else {
                    let image = InMemoryImage::<F32>::load(input).expect("Failed to load image");
                    let encoded = codec.encode(&image).expect("Failed to encode EXR");
                    black_box(encoded);
                }
            } else if let Some(codec) = jp2k_codec.as_ref() {
                if let Some(image) = preloaded_u8.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode JPEG 2000");
                    black_box(encoded);
                } else {
                    let image = InMemoryImage::<U8>::load(input).expect("Failed to load image");
                    let encoded = codec.encode(&image).expect("Failed to encode JPEG 2000");
                    black_box(encoded);
                }
            } else if let Some(codec) = tiff_codec.as_ref() {
                let opts = save_tiff_options
                    .as_ref()
                    .expect("missing TIFF save options");
                if let Some(image) = preloaded_tiff.as_ref() {
                    let encoded = encode_tiff_with_input(codec, image, opts);
                    black_box(encoded);
                } else {
                    let image = load_tiff_save_input(input);
                    let encoded = encode_tiff_with_input(codec, &image, opts);
                    black_box(encoded);
                }
            } else {
                match load_bench_image_with_options(input, &load_options) {
                    BenchImage::U8(image) => {
                        black_box(image);
                    }
                    BenchImage::U16(image) => {
                        black_box(image);
                    }
                    BenchImage::F32(image) => {
                        black_box(image);
                    }
                };
            }
        }

        #[cfg(feature = "count-alloc")]
        crate::counting_alloc::reset();

        let ru_before = getrusage();
        let mut wall_ns = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            if let Some(codec) = avif_codec.as_ref() {
                if let Some(image) = preloaded_u8.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode AVIF");
                    black_box(encoded);
                } else if let Some(image) = preloaded_u16.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode 16-bit AVIF");
                    black_box(encoded);
                } else {
                    match save_avif_format {
                        Some("u16") => {
                            let image =
                                InMemoryImage::<U16>::load(input).expect("Failed to load image");
                            let encoded =
                                codec.encode(&image).expect("Failed to encode 16-bit AVIF");
                            black_box(encoded);
                        }
                        _ => {
                            let image =
                                InMemoryImage::<U8>::load(input).expect("Failed to load image");
                            let encoded = codec.encode(&image).expect("Failed to encode AVIF");
                            black_box(encoded);
                        }
                    }
                }
            } else if let Some(codec) = gif_codec.as_ref() {
                if let Some(image) = preloaded_u8.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode GIF");
                    black_box(encoded);
                } else {
                    let image = InMemoryImage::<U8>::load(input).expect("Failed to load image");
                    let encoded = codec.encode(&image).expect("Failed to encode GIF");
                    black_box(encoded);
                }
            } else if let Some(codec) = heif_codec.as_ref() {
                if let Some(image) = preloaded_u8.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode HEIF");
                    black_box(encoded);
                } else {
                    let image = InMemoryImage::<U8>::load(input).expect("Failed to load image");
                    let encoded = codec.encode(&image).expect("Failed to encode HEIF");
                    black_box(encoded);
                }
            } else if let Some(codec) = jpeg_codec.as_ref() {
                let opts = save_jpeg_options
                    .as_ref()
                    .expect("missing JPEG save options");
                if let Some(image) = preloaded_u8.as_ref() {
                    let encoded = codec
                        .encode_with_options(image, opts)
                        .expect("Failed to encode JPEG");
                    black_box(encoded);
                } else {
                    let image = InMemoryImage::<U8>::load(input).expect("Failed to load image");
                    let encoded = codec
                        .encode_with_options(&image, opts)
                        .expect("Failed to encode JPEG");
                    black_box(encoded);
                }
            } else if let Some(codec) = exr_codec.as_ref() {
                if let Some(image) = preloaded_f32.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode EXR");
                    black_box(encoded);
                } else {
                    let image = InMemoryImage::<F32>::load(input).expect("Failed to load image");
                    let encoded = codec.encode(&image).expect("Failed to encode EXR");
                    black_box(encoded);
                }
            } else if let Some(codec) = jp2k_codec.as_ref() {
                if let Some(image) = preloaded_u8.as_ref() {
                    let encoded = codec.encode(image).expect("Failed to encode JPEG 2000");
                    black_box(encoded);
                } else {
                    let image = InMemoryImage::<U8>::load(input).expect("Failed to load image");
                    let encoded = codec.encode(&image).expect("Failed to encode JPEG 2000");
                    black_box(encoded);
                }
            } else if let Some(codec) = tiff_codec.as_ref() {
                let opts = save_tiff_options
                    .as_ref()
                    .expect("missing TIFF save options");
                if let Some(image) = preloaded_tiff.as_ref() {
                    let encoded = encode_tiff_with_input(codec, image, opts);
                    black_box(encoded);
                } else {
                    let image = load_tiff_save_input(input);
                    let encoded = encode_tiff_with_input(codec, &image, opts);
                    black_box(encoded);
                }
            } else {
                match load_bench_image_with_options(input, &load_options) {
                    BenchImage::U8(image) => {
                        black_box(image);
                    }
                    BenchImage::U16(image) => {
                        black_box(image);
                    }
                    BenchImage::F32(image) => {
                        black_box(image);
                    }
                };
            }
            wall_ns.push(start.elapsed().as_nanos() as u64);
        }
        let ru_after = getrusage();

        #[cfg(feature = "count-alloc")]
        {
            let stats = crate::counting_alloc::snapshot();
            let per_iter_allocs = stats.alloc_count / iterations as u64;
            let per_iter_bytes = stats.alloc_bytes / iterations as u64;
            println!(
                "ALLOC_STATS:{{\"alloc_count\":{},\"alloc_bytes\":{},\"per_iter_allocs\":{},\"per_iter_bytes\":{},\"peak_live_bytes\":{}}}",
                stats.alloc_count,
                stats.alloc_bytes,
                per_iter_allocs,
                per_iter_bytes,
                stats.peak_live_bytes
            );
        }

        return BenchResult {
            backend: viprs_backend_label(),
            input: input_str,
            operation: op.into(),
            iterations,
            wall_ns,
            peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
            minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
            major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
            vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
            invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
        };
    }

    // Create scheduler once — equivalent to libvips' global thread pool.
    let scheduler = RayonScheduler::new(threads).expect("scheduler creation");
    let backend = viprs_backend_label();
    // E2E mode: decode from disk on every iteration so the measurement
    // includes full codec decode cost (the productive pipeline scenario).
    //
    // The tile cache is NOT shared across iterations here. Each iteration
    // builds a fresh pipeline with a cold cache, matching libvips which
    // also computes from scratch on every call. Sharing a warm tile cache
    // across iterations would make subsequent runs near-free (cache hits)
    // while libvips still does real work — an unfair comparison.
    if e2e {
        let run_iter = |wall_ns: &mut Vec<u64>| {
            let start = Instant::now();
            let pipeline = build_viprs_e2e_pipeline(input, op, op_args);
            let mut sink = MemorySink::for_pipeline(&pipeline)
                .expect("MemorySink allocation failed: dimensions too large for bench input");
            scheduler.run(&pipeline, &mut sink).expect("run");
            black_box(sink);
            wall_ns.push(start.elapsed().as_nanos() as u64);
        };

        let mut warmup_ns = Vec::new();
        for _ in 0..WARMUP_ITERATIONS {
            run_iter(&mut warmup_ns);
        }
        drop(warmup_ns);

        #[cfg(feature = "count-alloc")]
        crate::counting_alloc::reset();

        let ru_before = getrusage();
        let mut wall_ns = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            run_iter(&mut wall_ns);
        }
        let ru_after = getrusage();

        return BenchResult {
            backend,
            input: input.to_string_lossy().to_string(),
            operation: op.into(),
            iterations,
            wall_ns,
            peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
            minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
            major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
            vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
            invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
        };
    }

    let source = preload_bench_source(input);
    for _ in 0..WARMUP_ITERATIONS {
        let pipeline = build_viprs_pipeline_from_preloaded(&source, op, op_args);
        let mut sink = DiscardSink::new();
        scheduler.run(&pipeline, &mut sink).expect("warmup");
    }

    #[cfg(feature = "count-alloc")]
    crate::counting_alloc::reset();

    let ru_before = getrusage();
    let mut wall_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let pipeline = build_viprs_pipeline_from_preloaded(&source, op, op_args);
        let mut sink = DiscardSink::new();
        scheduler.run(&pipeline, &mut sink).expect("run");
        wall_ns.push(start.elapsed().as_nanos() as u64);
    }
    let ru_after = getrusage();

    // Report allocation stats if counting allocator is active
    #[cfg(feature = "count-alloc")]
    {
        let stats = crate::counting_alloc::snapshot();
        let per_iter_allocs = stats.alloc_count / iterations as u64;
        let per_iter_bytes = stats.alloc_bytes / iterations as u64;
        println!(
            "ALLOC_STATS:{{\"alloc_count\":{},\"alloc_bytes\":{},\"per_iter_allocs\":{},\"per_iter_bytes\":{},\"peak_live_bytes\":{}}}",
            stats.alloc_count,
            stats.alloc_bytes,
            per_iter_allocs,
            per_iter_bytes,
            stats.peak_live_bytes
        );
    }

    BenchResult {
        backend,
        input: input_str,
        operation: op.into(),
        iterations,
        wall_ns,
        peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
        minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
        major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
        vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
        invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
    }
}

fn run_viprs_composite_bench(
    input: &Path,
    op_args: &[String],
    iterations: usize,
    threads: usize,
    e2e: bool,
) -> BenchResult {
    let input_str = input.to_string_lossy().to_string();
    let backend = viprs_backend_label();
    let mode = parse_composite_mode(op_args);
    let scheduler = RayonScheduler::new(threads).expect("scheduler creation");

    if e2e {
        let run_iter = |wall_ns: &mut Vec<u64>| {
            let start = Instant::now();
            let pipeline = build_viprs_composite_pipeline(input, mode);
            let mut sink = MemorySink::for_pipeline(&pipeline)
                .expect("MemorySink allocation failed: dimensions too large for bench input");
            scheduler.run(&pipeline, &mut sink).expect("composite run");
            black_box(sink);
            wall_ns.push(start.elapsed().as_nanos() as u64);
        };

        let mut warmup_ns = Vec::new();
        for _ in 0..WARMUP_ITERATIONS {
            run_iter(&mut warmup_ns);
        }
        drop(warmup_ns);

        let ru_before = getrusage();
        let mut wall_ns = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            run_iter(&mut wall_ns);
        }
        let ru_after = getrusage();

        return BenchResult {
            backend,
            input: input_str,
            operation: "composite".into(),
            iterations,
            wall_ns,
            peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
            minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
            major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
            vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
            invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
        };
    }

    let source = preload_bench_source(input);

    for _ in 0..WARMUP_ITERATIONS {
        let pipeline = build_viprs_composite_pipeline_from_preloaded(&source, mode);
        let mut sink = DiscardSink::new();
        scheduler
            .run(&pipeline, &mut sink)
            .expect("composite warmup");
    }

    let ru_before = getrusage();
    let mut wall_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let pipeline = build_viprs_composite_pipeline_from_preloaded(&source, mode);
        let mut sink = DiscardSink::new();
        scheduler.run(&pipeline, &mut sink).expect("composite run");
        wall_ns.push(start.elapsed().as_nanos() as u64);
    }
    let ru_after = getrusage();

    BenchResult {
        backend,
        input: input_str,
        operation: "composite".into(),
        iterations,
        wall_ns,
        peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
        minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
        major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
        vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
        invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
    }
}

fn parse_mapim_offsets(op_args: &[String]) -> (f32, f32) {
    let dx = op_args
        .first()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(DEFAULT_MAPIM_DX);
    let dy = op_args
        .get(1)
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(DEFAULT_MAPIM_DY);
    (dx, dy)
}

fn build_mapim_index(width: u32, height: u32, dx: f32, dy: f32) -> Vec<f32> {
    let mut index = Vec::with_capacity(width as usize * height as usize * 2);
    for y in 0..height {
        for x in 0..width {
            index.push(x as f32 + dx);
            index.push(y as f32 + dy);
        }
    }
    index
}

fn run_mapim_kernel<F>(image: &InMemoryImage<F>, index: &[f32], threads: usize)
where
    F: BandFormat + Send + Sync + 'static,
    F::Sample: Pod + Send + Sync,
    MapImOp<F>: DynOperation,
{
    let width = image.width();
    let height = image.height();
    let bands = image.bands();
    let op = MapImOp::<F>::new(width, height, bands, width, height, BandFormatId::F32)
        .with_extend(MapImExtend::Copy);
    let op = &op;
    let source_region = Region::new(0, 0, width, height);
    let source_bytes = cast_slice(image.pixels());
    let thread_count = threads.max(1).min(height.max(1) as usize);
    let rows_per_chunk = ((height as usize) + thread_count - 1) / thread_count;
    let mut output =
        vec![
            0u8;
            width as usize * height as usize * bands as usize * std::mem::size_of::<F::Sample>()
        ];

    thread::scope(|scope| {
        let mut remaining_output = output.as_mut_slice();
        for chunk_idx in 0..thread_count {
            let start_row = chunk_idx * rows_per_chunk;
            if start_row >= height as usize {
                break;
            }

            let chunk_rows = ((height as usize - start_row).min(rows_per_chunk)) as u32;
            let chunk_bytes = width as usize
                * chunk_rows as usize
                * bands as usize
                * std::mem::size_of::<F::Sample>();
            let (chunk_output, tail) = remaining_output.split_at_mut(chunk_bytes);
            remaining_output = tail;

            let index_start = start_row * width as usize * 2;
            let index_end = index_start + width as usize * chunk_rows as usize * 2;
            let index_chunk = &index[index_start..index_end];

            scope.spawn(move || {
                let output_region = Region::new(0, start_row as i32, width, chunk_rows);
                let input_regions = [source_region, output_region];
                let inputs = [source_bytes, cast_slice(index_chunk)];
                let mut state = op.dyn_start();
                op.dyn_process_region_multi(
                    state.as_mut(),
                    &inputs,
                    chunk_output,
                    &input_regions,
                    output_region,
                );
            });
        }
    });

    black_box(output);
}

fn run_mapim_for_bench_image(image: &BenchImage, index: &[f32], threads: usize) {
    match image {
        BenchImage::U8(image) => run_mapim_kernel(image, index, threads),
        BenchImage::U16(image) => run_mapim_kernel(image, index, threads),
        BenchImage::F32(image) => run_mapim_kernel(image, index, threads),
    }
}

fn run_viprs_mapim_bench(
    input: &Path,
    op_args: &[String],
    iterations: usize,
    threads: usize,
    e2e: bool,
) -> BenchResult {
    let input_str = input.to_string_lossy().to_string();
    let backend = viprs_backend_label();
    let (dx, dy) = parse_mapim_offsets(op_args);

    if e2e {
        let run_iter = |wall_ns: &mut Vec<u64>| {
            let start = Instant::now();
            let image = load_bench_image(input);
            let (width, height) = match &image {
                BenchImage::U8(image) => (image.width(), image.height()),
                BenchImage::U16(image) => (image.width(), image.height()),
                BenchImage::F32(image) => (image.width(), image.height()),
            };
            let index = build_mapim_index(width, height, dx, dy);
            run_mapim_for_bench_image(&image, &index, threads);
            wall_ns.push(start.elapsed().as_nanos() as u64);
        };

        let mut warmup_ns = Vec::new();
        for _ in 0..WARMUP_ITERATIONS {
            run_iter(&mut warmup_ns);
        }
        drop(warmup_ns);

        let ru_before = getrusage();
        let mut wall_ns = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            run_iter(&mut wall_ns);
        }
        let ru_after = getrusage();

        return BenchResult {
            backend,
            input: input_str,
            operation: "mapim".into(),
            iterations,
            wall_ns,
            peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
            minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
            major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
            vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
            invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
        };
    }

    let image = load_bench_image(input);
    let (width, height) = match &image {
        BenchImage::U8(image) => (image.width(), image.height()),
        BenchImage::U16(image) => (image.width(), image.height()),
        BenchImage::F32(image) => (image.width(), image.height()),
    };
    let index = build_mapim_index(width, height, dx, dy);

    for _ in 0..WARMUP_ITERATIONS {
        run_mapim_for_bench_image(&image, &index, threads);
    }

    let ru_before = getrusage();
    let mut wall_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        run_mapim_for_bench_image(&image, &index, threads);
        wall_ns.push(start.elapsed().as_nanos() as u64);
    }
    let ru_after = getrusage();

    BenchResult {
        backend,
        input: input_str,
        operation: "mapim".into(),
        iterations,
        wall_ns,
        peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
        minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
        major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
        vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
        invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
    }
}

/// Encode a `MemorySink` output to the requested target format.
///
/// Returns the encoded bytes. The pipeline output is always U8 (thumbnail + sharpen
/// produce U8) so this helper only handles that case.
fn encode_sink_to_format(
    sink: MemorySink,
    pipeline_width: u32,
    pipeline_height: u32,
    pipeline_bands: u32,
    pipeline_format: BandFormatId,
    target_format: &str,
) -> Vec<u8> {
    match pipeline_format {
        BandFormatId::U8 => {
            let image = sink
                .into_image::<U8>(
                    pipeline_width,
                    pipeline_height,
                    pipeline_bands,
                    ImageMetadata::default(),
                )
                .expect("workflow: failed to create Image from sink");
            encode_image_u8(&image, target_format)
        }
        other => {
            panic!(
                "workflow: unexpected pipeline output format {other:?} (thumbnail+sharpen should produce U8)"
            );
        }
    }
}

fn encode_image_u8(image: &InMemoryImage<U8>, target_format: &str) -> Vec<u8> {
    match target_format {
        "jpg" | "jpeg" => {
            let codec = JpegCodec;
            codec.encode(image).expect("workflow: JPEG encode failed")
        }
        "webp" => {
            let codec = WebpCodec;
            codec.encode(image).expect("workflow: WebP encode failed")
        }
        "png" => {
            let codec = PngCodec::default();
            codec.encode(image).expect("workflow: PNG encode failed")
        }
        "avif" => {
            let codec = AvifCodec;
            codec.encode(image).expect("workflow: AVIF encode failed")
        }
        "tif" | "tiff" => {
            let codec = TiffCodec::default();
            codec.encode(image).expect("workflow: TIFF encode failed")
        }
        _ => {
            panic!(
                "workflow: unsupported target format '{target_format}' (supported: jpg, webp, png, avif, tif)"
            );
        }
    }
}

/// Run the production workflow benchmark: decode → thumbnail → sharpen → encode.
///
/// This is always E2E (decode from disk) since that's the production scenario.
/// Unlike `run_viprs_bench`, the timing loop includes encoding the output.
pub fn run_viprs_workflow_bench(
    input: &Path,
    op: &str,
    op_args: &[String],
    iterations: usize,
    threads: usize,
) -> BenchResult {
    let input_str = input.to_string_lossy().to_string();
    let target_format = op_args.first().map(String::as_str).unwrap_or("webp");
    let scheduler = RayonScheduler::new(threads).expect("scheduler creation");
    let backend = viprs_backend_label();

    let run_iter = |wall_ns: &mut Vec<u64>| {
        let start = Instant::now();
        let pipeline = build_viprs_e2e_pipeline(input, op, op_args);
        let width = pipeline.width;
        let height = pipeline.height;
        let bands = pipeline.output_bands;
        let format = pipeline.output_format;
        let mut sink = MemorySink::for_pipeline(&pipeline)
            .expect("MemorySink allocation failed: dimensions too large for bench input");
        scheduler.run(&pipeline, &mut sink).expect("run");
        let encoded = encode_sink_to_format(sink, width, height, bands, format, target_format);
        black_box(encoded);
        wall_ns.push(start.elapsed().as_nanos() as u64);
    };

    // Warmup
    let mut warmup_ns = Vec::new();
    for _ in 0..WARMUP_ITERATIONS {
        run_iter(&mut warmup_ns);
    }
    drop(warmup_ns);

    let ru_before = getrusage();
    let mut wall_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        run_iter(&mut wall_ns);
    }
    let ru_after = getrusage();

    BenchResult {
        backend,
        input: input_str,
        operation: format!(
            "{}({}→{},w={})",
            op,
            input.extension().and_then(|e| e.to_str()).unwrap_or("?"),
            target_format,
            op_args.get(1).map(String::as_str).unwrap_or("800")
        ),
        iterations,
        wall_ns,
        peak_rss_kb: (ru_after.ru_maxrss / 1024) as u64,
        minor_faults: (ru_after.ru_minflt - ru_before.ru_minflt) as u64,
        major_faults: (ru_after.ru_majflt - ru_before.ru_majflt) as u64,
        vol_ctx_switches: (ru_after.ru_nvcsw - ru_before.ru_nvcsw) as u64,
        invol_ctx_switches: (ru_after.ru_nivcsw - ru_before.ru_nivcsw) as u64,
    }
}

pub fn run_baseline_bench(
    input: &Path,
    op: &str,
    op_args: &[String],
    iterations: usize,
    threads: usize,
    e2e: bool,
) -> Result<Option<BenchResult>, String> {
    let backend = BaselineBackend::for_op(op);
    let runner = backend.runner_path();
    if !runner.exists() {
        eprintln!(
            "{} not found. Build with: make -C tools/bench-vs-libvips",
            backend.runner_name()
        );
        return Ok(None);
    }

    let mut cmd = Command::new(&runner);
    cmd.arg(input.to_string_lossy().as_ref())
        .arg(baseline_op_name(op));
    for a in op_args {
        cmd.arg(a);
    }
    cmd.arg("--iterations").arg(iterations.to_string());
    cmd.arg("--threads").arg(threads.to_string());
    // Never enable libvips operation cache in benchmarks. The cache is an
    // application-level memoization layer (like Redis) — on iterations 2+ it
    // serves pre-computed results, while viprs honestly recomputes every time.
    // This made ALL E2E ratios dishonest (e.g. PNG invert showed 6.4x gap when
    // the real ratio without cache is 0.75x — viprs wins).
    if e2e {
        cmd.arg("--e2e");
    }

    let output = cmd
        .output()
        .unwrap_or_else(|error| panic!("Failed to run {}: {error}", backend.runner_name()));
    parse_baseline_output(backend, &output).map(Some)
}

/// Run a single input file and return the printed summary plus the Comparison value.
pub fn run_single(
    input_path: &Path,
    op: &str,
    op_args: &[String],
    iterations: usize,
    threads: usize,
    e2e: bool,
    quiet: bool,
) -> Comparison {
    // Normalize perceptual_enhance op_args: both viprs and libvips must receive the same
    // target format. Viprs defaults to "webp" internally; ensure the C runner gets it too.
    let normalized_args: Vec<String>;
    let op_args = if is_workflow_like_op(op) && op != "workflow" && op_args.is_empty() {
        normalized_args = vec!["webp".to_string()];
        normalized_args.as_slice()
    } else {
        op_args
    };

    if !quiet {
        println!("Input: {}", input_path.display());
        println!("Operation: {}", scenario_display_label(op, op_args));
        println!("Iterations: {}", iterations);
        if e2e {
            println!("Mode: e2e (decode-from-disk included in every iteration)");
        }
        println!();
    }

    // Run libvips side
    if !quiet {
        println!("--- {} ---", libvips_backend_label());
    }
    let libvips_result = run_baseline_bench(input_path, op, op_args, iterations, threads, e2e)
        .unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(1);
        });
    if let Some(ref r) = libvips_result {
        let mut sorted = r.wall_ns.clone();
        sorted.sort_unstable();
        let p50 = percentile(&sorted, 0.50);
        let p95 = percentile(&sorted, 0.95);
        if !quiet {
            println!(
                "  {}  p50: {:.2} ms  p95: {:.2} ms  RSS: {} KB  faults: {}  ctx_sw: {}",
                r.backend,
                p50 as f64 / 1e6,
                p95 as f64 / 1e6,
                r.peak_rss_kb,
                r.minor_faults,
                r.vol_ctx_switches + r.invol_ctx_switches
            );
        }
    } else if !quiet {
        println!("  (skipped — runner not available)");
    }
    if !quiet {
        println!();
    }

    // Run viprs side
    if !quiet {
        println!("--- {} ---", viprs_backend_label());
    }
    let viprs_result = if !e2e {
        run_viprs_bench(input_path, op, op_args, iterations, threads, false)
    } else if is_workflow_like_op(op) {
        run_viprs_workflow_bench(input_path, op, op_args, iterations, threads)
    } else {
        run_viprs_bench(input_path, op, op_args, iterations, threads, true)
    };
    let mut viprs_sorted = viprs_result.wall_ns.clone();
    viprs_sorted.sort_unstable();
    let vp50 = percentile(&viprs_sorted, 0.50);
    let vp95 = percentile(&viprs_sorted, 0.95);
    if !quiet {
        println!(
            "  {}  p50: {:.2} ms  p95: {:.2} ms  RSS: {} KB  faults: {}  ctx_sw: {}",
            viprs_result.backend,
            vp50 as f64 / 1e6,
            vp95 as f64 / 1e6,
            viprs_result.peak_rss_kb,
            viprs_result.minor_faults,
            viprs_result.vol_ctx_switches + viprs_result.invol_ctx_switches
        );
        println!();
    }

    // Compute ratios
    let ratios = libvips_result.as_ref().map(|lv| {
        let mut lv_sorted = lv.wall_ns.clone();
        lv_sorted.sort_unstable();
        let lp50 = percentile(&lv_sorted, 0.50);
        let lp95 = percentile(&lv_sorted, 0.95);

        let r = Ratios {
            latency_p50: vp50 as f64 / lp50.max(1) as f64,
            latency_p95: vp95 as f64 / lp95.max(1) as f64,
            rss: viprs_result.peak_rss_kb as f64 / lv.peak_rss_kb.max(1) as f64,
        };

        if !quiet {
            println!("--- comparison (ratio = viprs/libvips, <1.0 means viprs wins) ---");
            println!("  latency p50: {:.3}x", r.latency_p50);
            println!("  latency p95: {:.3}x", r.latency_p95);
            println!("  RSS:         {:.3}x", r.rss);
            if r.latency_p50 < 1.0 {
                println!("  >>> viprs is {:.2}x FASTER <<<", 1.0 / r.latency_p50);
            } else {
                println!("  >>> viprs is {:.2}x slower <<<", r.latency_p50);
            }
        }

        r
    });

    Comparison {
        libvips: libvips_result,
        viprs: Some(viprs_result),
        ratios,
    }
}

/// Run the benchmark across the default size matrix and print a summary table.
pub fn run_multi_size(
    op: &str,
    op_args: &[String],
    iterations: usize,
    threads: usize,
    e2e: bool,
    json_output: bool,
) -> Vec<SummaryRow> {
    let fixtures = bench_fixtures_for_op(op);
    let dimensions = fixtures
        .iter()
        .map(|fixture| format!("{}x{}", fixture.width, fixture.height))
        .collect::<Vec<_>>()
        .join(" / ");

    if !json_output {
        println!("=== xtask bench (multi-size: {dimensions}) ===");
        println!("Operation:  {}", scenario_display_label(op, op_args));
        println!("Iterations: {} per size", iterations);
        println!(
            "Mode:       {}",
            if e2e {
                "e2e (decode-from-disk per iteration)"
            } else {
                "kernel-only (pre-loaded pixels)"
            }
        );
        println!();
    }

    struct SizeRow {
        dimensions: String,
        lv_p50_ms: f64,
        lv_p95_ms: f64,
        vp_p50_ms: f64,
        vp_p95_ms: f64,
        ratio_p50: Option<f64>,
        ratio_p95: Option<f64>,
    }

    let results_dir = repo_root().join("tools/bench-vs-libvips/results");
    std::fs::create_dir_all(&results_dir).ok();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut rows: Vec<SizeRow> = Vec::new();
    let mut summary_rows: Vec<SummaryRow> = Vec::new();
    let mut any_missing = false;

    for fixture in fixtures {
        let path = repo_root().join(fixture.input);
        if !path.exists() {
            eprintln!(
                "WARNING: bench image for {}x{} not found at {}",
                fixture.width,
                fixture.height,
                path.display()
            );
            eprintln!("  Generate it with: tools/gen-fixtures.sh");
            any_missing = true;
            continue;
        }

        let size = fixture.size;
        if !json_output {
            println!("--- size {}x{} ---", fixture.width, fixture.height);
        }
        let cmp = run_single(&path, op, op_args, iterations, threads, e2e, true);

        let (lv_p50_ms, lv_p95_ms) = cmp
            .libvips
            .as_ref()
            .map(bench_result_percentiles)
            .unwrap_or((0.0, 0.0));
        let (vp_p50_ms, vp_p95_ms) = cmp
            .viprs
            .as_ref()
            .map(bench_result_percentiles)
            .unwrap_or((0.0, 0.0));
        let ratio_p50 = cmp.ratios.as_ref().map(|r| r.latency_p50);
        let ratio_p95 = cmp.ratios.as_ref().map(|r| r.latency_p95);

        let result_file = results_dir.join(format!(
            "{}_{}x{}_{}.json",
            scenario_slug(op, op_args),
            fixture.width,
            fixture.height,
            timestamp
        ));
        let json = serde_json::to_string_pretty(&cmp).unwrap_or_default();
        std::fs::write(&result_file, &json).ok();

        let lv_p50_opt = cmp.libvips.as_ref().map(|_| lv_p50_ms);
        let lv_p95_opt = cmp.libvips.as_ref().map(|_| lv_p95_ms);
        append_trend(
            &results_dir,
            &TrendRecord {
                date: iso_timestamp(),
                git_sha: git_sha(),
                op: op.to_owned(),
                op_args: op_args.to_vec(),
                size,
                viprs_p50_ms: vp_p50_ms,
                viprs_p95_ms: vp_p95_ms,
                libvips_p50_ms: lv_p50_opt,
                libvips_p95_ms: lv_p95_opt,
                ratio_p50,
            },
        );

        summary_rows.push(build_summary_row(
            op,
            op_args,
            &path,
            None,
            Some(size),
            &cmp,
        ));
        rows.push(SizeRow {
            dimensions: format!("{}x{}", fixture.width, fixture.height),
            lv_p50_ms,
            lv_p95_ms,
            vp_p50_ms,
            vp_p95_ms,
            ratio_p50,
            ratio_p95,
        });
        if !json_output {
            println!();
        }
    }

    if any_missing && !json_output {
        println!("WARNING: some sizes were skipped. See messages above.");
        println!();
    }

    if rows.is_empty() {
        eprintln!("No benchmark images found. Run:");
        eprintln!("  tools/gen-fixtures.sh");
        return summary_rows;
    }

    if !json_output {
        println!(
            "=== SUMMARY: {} (ratio = viprs/libvips, <1.0 means viprs wins) ===",
            scenario_display_label(op, op_args)
        );
        println!(
            "{:<10}  {:>12}  {:>12}  {:>12}  {:>12}  {:>10}  {:>10}",
            "size", "lv p50 ms", "lv p95 ms", "vp p50 ms", "vp p95 ms", "ratio p50", "ratio p95"
        );
        println!("{}", "-".repeat(90));
        for row in &rows {
            let r_p50 = row
                .ratio_p50
                .map(|r| format!("{:.3}x", r))
                .unwrap_or_else(|| "N/A".into());
            let r_p95 = row
                .ratio_p95
                .map(|r| format!("{:.3}x", r))
                .unwrap_or_else(|| "N/A".into());
            println!(
                "{:<10}  {:>12.2}  {:>12.2}  {:>12.2}  {:>12.2}  {:>10}  {:>10}",
                row.dimensions,
                row.lv_p50_ms,
                row.lv_p95_ms,
                row.vp_p50_ms,
                row.vp_p95_ms,
                r_p50,
                r_p95,
            );
        }
        println!();
    }
    summary_rows
}

pub fn run_scenario_set(
    scenario_set_name: &str,
    op: &str,
    op_args: &[String],
    iterations: usize,
    threads: usize,
    e2e: bool,
    json_output: bool,
) -> Vec<SummaryRow> {
    let scenarios = match scenario_set(scenario_set_name) {
        Some(scenarios) => scenarios,
        None => {
            eprintln!(
                "Unknown --scenario-set '{scenario_set_name}'. Supported: compute-baselines, input-diversity, production-workflows"
            );
            std::process::exit(1);
        }
    };

    if !scenario_set_supports_op(op) {
        eprintln!(
            "--scenario-set only supports ops that run across the full input-diversity matrix: {}",
            INPUT_DIVERSITY_SUPPORTED_OPS.join(", ")
        );
        std::process::exit(1);
    }

    if !json_output {
        println!("=== xtask bench (scenario-set: {scenario_set_name}) ===");
        println!(
            "Operation:  {}",
            scenario_set_display_label(scenario_set_name, op, op_args)
        );
        println!("Iterations: {} per scenario", iterations);
        println!(
            "Mode:       {}",
            if e2e {
                "e2e (decode-from-disk per iteration)"
            } else {
                "kernel-only (pre-loaded pixels)"
            }
        );
        println!();
        println!("Covered input matrix:");
        for scenario in scenarios {
            println!("  {:<10} {}", scenario.key, scenario.description);
        }
        println!();
    }

    struct ScenarioRow {
        key: String,
        lv_p50_ms: f64,
        lv_p95_ms: f64,
        vp_p50_ms: f64,
        vp_p95_ms: f64,
        ratio_p50: Option<f64>,
        ratio_p95: Option<f64>,
    }

    let results_dir = repo_root().join("tools/bench-vs-libvips/results");
    std::fs::create_dir_all(&results_dir).ok();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut rows: Vec<ScenarioRow> = Vec::new();
    let mut summary_rows: Vec<SummaryRow> = Vec::new();

    for scenario in scenarios {
        let path = repo_root().join(scenario.input);
        if !path.exists() {
            eprintln!(
                "WARNING: scenario '{}' skipped because fixture is missing: {}",
                scenario.key,
                path.display()
            );
            continue;
        }
        if !json_output {
            println!(
                "--- scenario {} ({}) ---",
                scenario.key, scenario.description
            );
        }
        let effective_op_args: Vec<String> = if op == "workflow" {
            workflow_op_args_for_scenario(scenario.key, op_args)
        } else {
            op_args.to_vec()
        };

        let cmp = run_single(
            &path,
            op,
            &effective_op_args,
            iterations,
            threads,
            e2e,
            true,
        );

        let (lv_p50_ms, lv_p95_ms) = cmp
            .libvips
            .as_ref()
            .map(bench_result_percentiles)
            .unwrap_or((0.0, 0.0));
        let (vp_p50_ms, vp_p95_ms) = cmp
            .viprs
            .as_ref()
            .map(bench_result_percentiles)
            .unwrap_or((0.0, 0.0));
        let ratio_p50 = cmp.ratios.as_ref().map(|r| r.latency_p50);
        let ratio_p95 = cmp.ratios.as_ref().map(|r| r.latency_p95);

        let result_file = results_dir.join(format!(
            "{}_{}_{}.json",
            scenario_slug(op, &effective_op_args),
            scenario.key,
            timestamp
        ));
        let json = serde_json::to_string_pretty(&cmp).unwrap_or_default();
        std::fs::write(&result_file, &json).ok();

        summary_rows.push(build_summary_row(
            op,
            &effective_op_args,
            &path,
            Some(scenario.key),
            None,
            &cmp,
        ));
        rows.push(ScenarioRow {
            key: scenario.key.to_owned(),
            lv_p50_ms,
            lv_p95_ms,
            vp_p50_ms,
            vp_p95_ms,
            ratio_p50,
            ratio_p95,
        });
        if !json_output {
            println!();
        }
    }

    if !json_output {
        println!(
            "=== SUMMARY: {} [{}] (ratio = viprs/libvips, <1.0 means viprs wins) ===",
            scenario_set_display_label(scenario_set_name, op, op_args),
            scenario_set_name
        );
        println!(
            "{:<10}  {:>12}  {:>12}  {:>12}  {:>12}  {:>10}  {:>10}",
            "scenario",
            "lv p50 ms",
            "lv p95 ms",
            "vp p50 ms",
            "vp p95 ms",
            "ratio p50",
            "ratio p95"
        );
        println!("{}", "-".repeat(92));
        for row in &rows {
            let r_p50 = row
                .ratio_p50
                .map(|r| format!("{:.3}x", r))
                .unwrap_or_else(|| "N/A".into());
            let r_p95 = row
                .ratio_p95
                .map(|r| format!("{:.3}x", r))
                .unwrap_or_else(|| "N/A".into());
            println!(
                "{:<10}  {:>12.2}  {:>12.2}  {:>12.2}  {:>12.2}  {:>10}  {:>10}",
                row.key, row.lv_p50_ms, row.lv_p95_ms, row.vp_p50_ms, row.vp_p95_ms, r_p50, r_p95,
            );
        }
        println!();
    }

    summary_rows
}
