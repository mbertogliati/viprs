use super::*;
use crate::{
    domain::{
        format::{F32, U8},
        image::{Region, Tile, TileMut},
        op::{Op, OperationBridge, PixelLocalOp},
        ops::resample::{Resize, Thumbnail, mapim::MapImOp, thumbnail::ThumbnailTarget},
    },
    ports::scheduler::TileScheduler,
    scheduler::rayon_scheduler::RayonScheduler,
    sinks::memory::MemorySink,
    sources::memory::MemorySource,
};

fn run_resize_pipeline_with_pixels(
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    hscale: f64,
    vscale: f64,
    kernel: InterpolationKernel,
) -> (u32, u32, Vec<u8>) {
    let source = MemorySource::<U8>::new(width, height, 1, pixels).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .resize(Resize::new(hscale, vscale, kernel))
        .unwrap()
        .build()
        .unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    (pipeline.width, pipeline.height, sink.into_buffer())
}

fn run_thumbnail_pipeline_with_pixels(
    input_width: u32,
    input_height: u32,
    bands: u32,
    pixels: Vec<u8>,
    target_width: u32,
    target_height: u32,
    kernel: InterpolationKernel,
) -> (u32, u32, Vec<u8>) {
    let source = MemorySource::<U8>::new(input_width, input_height, bands, pixels).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::FitBox {
                width: target_width,
                height: target_height,
            },
            kernel,
        ))
        .unwrap()
        .build()
        .unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(RayonScheduler::default_threads())
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    (pipeline.width, pipeline.height, sink.into_buffer())
}

fn stepped_gradient_pixels(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let ramp = x * 5 + y * 3;
                let plateau = if x >= width / 3 && x < (width * 2) / 3 {
                    72
                } else {
                    0
                };
                let stripes = if (x / 2 + y / 3) % 2 == 0 { 24 } else { 0 };
                (ramp + plateau + stripes).min(u32::from(u8::MAX)) as u8
            })
        })
        .collect()
}

struct IdentityF32;

impl PixelLocalOp for IdentityF32 {}

impl Op for IdentityF32 {
    type Input = F32;
    type Output = F32;
    type State = ();

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) -> Self::State {}

    fn process_region(
        &self,
        _state: &mut Self::State,
        input: &Tile<Self::Input>,
        output: &mut TileMut<Self::Output>,
    ) {
        output.data.copy_from_slice(input.data);
    }
}

#[test]
fn resize_identity_scale_returns_same_pixels() {
    let pixels: Vec<u8> = (0..100)
        .flat_map(|y| (0..100).map(move |x| ((x * 9 + y * 5).min(u32::from(u8::MAX))) as u8))
        .collect();

    let (width, height, buffer) = run_resize_pipeline_with_pixels(
        100,
        100,
        pixels.clone(),
        1.0,
        1.0,
        InterpolationKernel::Lanczos3,
    );

    assert_eq!((width, height), (100, 100));
    assert_eq!(buffer, pixels);
}

#[test]
fn resize_downscale_keeps_uniform_pixels_within_rounding_error() {
    let (width, height, output) = run_resize_pipeline_with_pixels(
        8,
        8,
        vec![200_u8; 8 * 8],
        0.5,
        0.5,
        InterpolationKernel::Lanczos3,
    );

    assert_eq!((width, height), (4, 4));
    assert!(output.iter().all(|&pixel| pixel.abs_diff(200) <= 1));
}

#[test]
fn thumbnail_pipeline_width_and_height_limits_match_target_box() {
    let (width, height, pixels) = run_thumbnail_pipeline_with_pixels(
        1000,
        500,
        3,
        vec![128u8; 1000 * 500 * 3],
        100,
        100,
        InterpolationKernel::Lanczos3,
    );
    assert_eq!((width, height), (100, 50));
    assert!(pixels.iter().all(|&sample| sample == 128));

    let (width, height, pixels) = run_thumbnail_pipeline_with_pixels(
        500,
        1000,
        3,
        vec![128u8; 500 * 1000 * 3],
        100,
        100,
        InterpolationKernel::Lanczos3,
    );
    assert_eq!((width, height), (50, 100));
    assert!(pixels.iter().all(|&sample| sample == 128));
}

#[test]
fn thumbnail_non_constant_input_exposes_lanczos3_reduce_path() {
    let width = 34;
    let height = 34;
    let target = 14;
    let (output_width, output_height, pixels) = run_thumbnail_pipeline_with_pixels(
        width,
        height,
        1,
        stepped_gradient_pixels(width, height),
        target,
        target,
        InterpolationKernel::Lanczos3,
    );
    let (_, _, bilinear_pixels) = run_thumbnail_pipeline_with_pixels(
        width,
        height,
        1,
        stepped_gradient_pixels(width, height),
        target,
        target,
        InterpolationKernel::Bilinear,
    );

    assert_eq!((output_width, output_height), (target, target));
    assert_ne!(pixels, bilinear_pixels);
}

#[test]
fn reduce_fractional_pipeline_matches_declared_dimensions() {
    let image = InMemoryImage::<U8>::from_buffer(
        777,
        333,
        3,
        (0..777 * 333 * 3)
            .map(|index| ((index * 31 + 11) % 256) as u8)
            .collect(),
    )
    .unwrap();
    let pipeline = ImagePipeline::from_source(
        MemorySource::<U8>::new(
            image.width(),
            image.height(),
            image.bands(),
            image.pixels().to_vec(),
        )
        .unwrap(),
    )
    .reduce(1.5, 2.5, InterpolationKernel::Lanczos3)
    .unwrap()
    .build()
    .unwrap();

    let expected_len =
        pipeline.width as usize * pipeline.height as usize * pipeline.output_bands as usize;
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    assert_eq!(sink.into_buffer().len(), expected_len);
}

#[test]
fn mapim_pipeline_connect_to_slot_runs_coordinate_map() {
    let source_pixels = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let source = MemorySource::<F32>::new(2, 2, 2, source_pixels.clone()).unwrap();

    let mut arena = PipelineArena::with_source(Box::new(source));
    let upstream = arena.add_node(Box::new(OperationBridge::new_pixel_local(IdentityF32, 2)));
    let node = arena.add_node(Box::new(
        MapImOp::<F32>::new(2, 2, 2, 2, 2, BandFormatId::F32).with_premultiplied(true),
    ));
    arena.connect(upstream, node).unwrap();
    arena.connect_to_slot(upstream, node, 1).unwrap();
    let pipeline = arena.compile().unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(RayonScheduler::default_threads())
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    assert_eq!(
        bytemuck::cast_slice::<u8, f32>(&sink.into_buffer()),
        source_pixels.as_slice()
    );
}

#[test]
fn subsample_builder_zero_x_factor_returns_source_hint_error() {
    let source = MemorySource::<U8>::new(4, 4, 1, (0u8..16).collect()).unwrap();
    let result = ImagePipeline::from_source(source).subsample(0, 1);

    assert!(matches!(
        result,
        Err(BuildError::SourceHint {
            context: "subsample",
            ..
        })
    ));
}
