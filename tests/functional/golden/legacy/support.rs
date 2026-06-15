use super::super::support as golden;

pub(crate) use viprs::{
    BuildError, PipelineBuilder, TileScheduler,
    adapters::{
        pipeline::CompiledPipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::U8,
        image::{Region, Tile, TileMut},
        kernel::InterpolationKernel,
        op::Op,
        ops::resample::{
            Resize, Thumbnail, reduceh::ReduceH, reducev::ReduceV, resize::ResizeNode,
            shrinkh::ShrinkH, shrinkv::ShrinkV,
        },
    },
};

pub(crate) use golden::{ImageSpec, VipsBandFormat};
pub(crate) use std::num::NonZeroUsize;
pub(crate) use viprs::domain::ops::resample::thumbnail::ThumbnailTarget;
pub(crate) use viprs::ports::source::ImageSource;

pub(crate) fn ensure_vips() {
    golden::require_vips();
}

pub(crate) fn rgb_source(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 29 + y * 7 + 3) % 256) as u8);
            pixels.push(((x * 11 + y * 23 + 17) % 256) as u8);
            pixels.push(((x * 5 + y * 13 + 29) % 256) as u8);
        }
    }
    pixels
}

pub(crate) fn grayscale_source(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 17 + y * 13 + 5) % 256) as u8);
        }
    }
    pixels
}

pub(crate) fn smooth_grayscale_source(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| ((x * 4 + y * 3) % 256) as u8))
        .collect()
}

pub(crate) fn build_pipeline_u8(
    source_pixels: Vec<u8>,
    width: u32,
    height: u32,
    bands: u32,
    configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder, BuildError>,
) -> CompiledPipeline {
    let source =
        MemorySource::<U8>::new(width, height, bands, source_pixels).expect("MemorySource");
    configure(PipelineBuilder::from_source(source))
        .expect("pipeline step")
        .build()
        .expect("pipeline build")
}

pub(crate) fn run_pipeline_with_scheduler(
    pipeline: &CompiledPipeline,
    scheduler: &RayonScheduler,
) -> Vec<u8> {
    let mut sink = MemorySink::for_pipeline(pipeline).unwrap();
    scheduler.run(pipeline, &mut sink).expect("pipeline run");
    sink.into_buffer()
}

pub(crate) fn run_pipeline_u8(
    source_pixels: Vec<u8>,
    width: u32,
    height: u32,
    bands: u32,
    configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder, BuildError>,
) -> (u32, u32, Vec<u8>) {
    let pipeline = build_pipeline_u8(source_pixels, width, height, bands, configure);
    let scheduler = RayonScheduler::new(1).expect("scheduler");
    let output = run_pipeline_with_scheduler(&pipeline, &scheduler);
    (pipeline.width, pipeline.height, output)
}

pub(crate) fn run_cached_pipeline_u8_twice(
    source_pixels: Vec<u8>,
    width: u32,
    height: u32,
    bands: u32,
    configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder, BuildError>,
) -> (u32, u32, Vec<u8>, Vec<u8>) {
    let pipeline = build_pipeline_u8(source_pixels, width, height, bands, configure);
    let scheduler = RayonScheduler::new(1).expect("scheduler");
    let first = run_pipeline_with_scheduler(&pipeline, &scheduler);
    let second = run_pipeline_with_scheduler(&pipeline, &scheduler);
    assert_eq!(
        first, second,
        "cache must return identical pixels on second run"
    );
    (pipeline.width, pipeline.height, first, second)
}

pub(crate) fn write_u8_input_spec(
    op: &str,
    case: &str,
    name: &str,
    pixels: &[u8],
    spec: ImageSpec,
) -> String {
    golden::write_vips_input(op, case, name, pixels, spec)
        .display()
        .to_string()
}

pub(crate) fn round_vips(value: f64) -> i32 {
    if value > 0.0 {
        (value + 0.5) as i32
    } else {
        (value - 0.5) as i32
    }
}

pub(crate) fn affine_rotate_scale_auto_canvas(
    scale: f64,
    angle_degrees: f64,
    input_width: u32,
    input_height: u32,
) -> ([f64; 4], f64, f64, u32, u32) {
    let radians = angle_degrees.to_radians();
    let cos = radians.cos();
    let sin = radians.sin();
    let forward = [scale * cos, scale * -sin, scale * sin, scale * cos];
    let inverse = [cos / scale, sin / scale, -sin / scale, cos / scale];

    let width = f64::from(input_width);
    let height = f64::from(input_height);
    let corners = [
        (0.0_f64, 0.0_f64),
        (forward[0] * width, forward[2] * width),
        (forward[1] * height, forward[3] * height),
        (
            forward[0] * width + forward[1] * height,
            forward[2] * width + forward[3] * height,
        ),
    ];

    let left = round_vips(
        corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::INFINITY, f64::min),
    );
    let right = round_vips(
        corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let top = round_vips(
        corners
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::INFINITY, f64::min),
    );
    let bottom = round_vips(
        corners
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::NEG_INFINITY, f64::max),
    );

    let output_width = (right - left).max(0) as u32;
    let output_height = (bottom - top).max(0) as u32;
    let tx = inverse[0] * f64::from(left) + inverse[1] * f64::from(top);
    let ty = inverse[2] * f64::from(left) + inverse[3] * f64::from(top);

    (inverse, tx, ty, output_width, output_height)
}

pub(crate) fn resize_kernel_name(kernel: InterpolationKernel) -> &'static str {
    match kernel {
        InterpolationKernel::Nearest => "nearest",
        InterpolationKernel::Bilinear => "linear",
        InterpolationKernel::Bicubic => "cubic",
        InterpolationKernel::Lanczos3 => "lanczos3",
        _ => panic!("unexpected kernel for this golden test: {kernel:?}"),
    }
}

pub(crate) fn interpolate_name(kernel: InterpolationKernel) -> &'static str {
    match kernel {
        InterpolationKernel::Nearest => "nearest",
        InterpolationKernel::Bilinear => "bilinear",
        InterpolationKernel::Bicubic => "bicubic",
        _ => panic!("unexpected interpolator for this golden test: {kernel:?}"),
    }
}
