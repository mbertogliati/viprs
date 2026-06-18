use super::flows_support::*;
use bytemuck::{cast_slice, cast_slice_mut};
use viprs::{
    BuildError, F32, F64, Image, ImageMetadata, Interpretation, OperationBridge, Region, U8,
    adapters::{
        pipeline::PipelineArena, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        kernel::InterpolationKernel,
        op::{DynOperation, Op},
        ops::{
            conversion::{
                composite::{BlendMode, CompositeOp},
                smartcrop::{Interesting, SmartcropOp},
            },
            convolution::{Canny, ConvaOp, ConvolutionMask2d, EdgeOp},
            histogram::ClaheOp,
            morphology::{Dilate, Erode},
            resample::{MapImOp, mapim::MapImExtend},
        },
    },
    ports::scheduler::TileScheduler,
};

#[cfg(feature = "jpeg")]
use viprs::adapters::codecs::JpegCodec;
#[cfg(feature = "jpeg")]
use viprs::ports::codec::{ImageDecoder, ImageEncoder};

#[cfg(feature = "icc")]
use viprs::domain::ops::colour::{IccTransformOptions, icc_transform, profile_load};
#[cfg(feature = "fft")]
use viprs::domain::ops::freqfilt::{COMPLEX_BANDS, FreqMultOp, PhasecorOp};
#[cfg(feature = "fft")]
use viprs::{fwfft, invfft};

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |acc, byte| {
        (acc ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn to_rgba_u8(image: &Image<U8>) -> Image<U8> {
    match image.bands() {
        4 => image.clone(),
        3 => {
            let mut pixels = Vec::with_capacity((image.width() * image.height() * 4) as usize);
            for rgb in image.pixels().chunks_exact(3) {
                pixels.extend_from_slice(rgb);
                pixels.push(u8::MAX);
            }
            Image::<U8>::from_buffer(image.width(), image.height(), 4, pixels)
                .expect("failed to materialize rgba image")
                .with_metadata(image.metadata().clone())
        }
        bands => panic!("unsupported band count for rgba conversion: {bands}"),
    }
}

fn place_overlay_on_canvas(
    base_width: u32,
    base_height: u32,
    overlay: &Image<U8>,
    x: u32,
    y: u32,
) -> Image<U8> {
    let mut pixels = vec![0u8; (base_width * base_height * overlay.bands()) as usize];
    let bands = overlay.bands() as usize;
    for row in 0..overlay.height() {
        let dst_y = y + row;
        if dst_y >= base_height {
            break;
        }
        for col in 0..overlay.width() {
            let dst_x = x + col;
            if dst_x >= base_width {
                break;
            }
            let src_idx = ((row * overlay.width() + col) * overlay.bands()) as usize;
            let dst_idx = ((dst_y * base_width + dst_x) * overlay.bands()) as usize;
            pixels[dst_idx..dst_idx + bands]
                .copy_from_slice(&overlay.pixels()[src_idx..src_idx + bands]);
        }
    }
    Image::<U8>::from_buffer(base_width, base_height, overlay.bands(), pixels)
        .expect("failed to materialize overlay canvas")
        .with_metadata(ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        })
}

fn run_u8_to_u8_op<T>(image: &Image<U8>, op: T, output_bands: u32) -> Image<U8>
where
    T: Op<Input = U8, Output = U8> + 'static,
{
    let source = memory_source_from_image(image);
    let mut arena = PipelineArena::with_source(Box::new(source));
    let _node = arena.add_node(Box::new(OperationBridge::new(op, image.bands())));
    let pipeline = arena.compile().expect("pipeline build failed");
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .expect("scheduler construction failed")
        .run(&pipeline, &mut sink)
        .expect("pipeline execution failed");
    let buffer = sink.into_buffer();
    Image::<U8>::from_buffer(pipeline.width, pipeline.height, output_bands, buffer)
        .expect("failed to materialize u8 output")
        .with_metadata(image.metadata().clone())
}

fn run_u8_to_f32_op<T>(image: &Image<U8>, op: T, output_bands: u32) -> Image<F32>
where
    T: Op<Input = U8, Output = F32> + 'static,
{
    let source = memory_source_from_image(image);
    let mut arena = PipelineArena::with_source(Box::new(source));
    let _node = arena.add_node(Box::new(OperationBridge::new(op, image.bands())));
    let pipeline = arena.compile().expect("pipeline build failed");
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .expect("scheduler construction failed")
        .run(&pipeline, &mut sink)
        .expect("pipeline execution failed");
    let raw = sink.into_buffer();
    let pixels = cast_slice::<u8, f32>(&raw).to_vec();
    Image::<F32>::from_buffer(pipeline.width, pipeline.height, output_bands, pixels)
        .expect("failed to materialize f32 output")
}

fn run_f32_to_f32_op<T>(image: &Image<F32>, op: T, output_bands: u32) -> Image<F32>
where
    T: Op<Input = F32, Output = F32> + 'static,
{
    let source = MemorySource::<F32>::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .expect("failed to create f32 source")
    .with_metadata(image.metadata().clone());
    let mut arena = PipelineArena::with_source(Box::new(source));
    let _node = arena.add_node(Box::new(OperationBridge::new(op, image.bands())));
    let pipeline = arena.compile().expect("pipeline build failed");
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .expect("scheduler construction failed")
        .run(&pipeline, &mut sink)
        .expect("pipeline execution failed");
    let raw = sink.into_buffer();
    let pixels = cast_slice::<u8, f32>(&raw).to_vec();
    Image::<F32>::from_buffer(pipeline.width, pipeline.height, output_bands, pixels)
        .expect("failed to materialize f32 output")
}

fn apply_smartcrop(
    image: &Image<U8>,
    target_width: u32,
    target_height: u32,
    interesting: Interesting,
) -> Image<U8> {
    let op = SmartcropOp::analyze_with_interesting(image, target_width, target_height, interesting);
    let source = memory_source_from_image(image);
    let mut arena = PipelineArena::with_source(Box::new(source));
    let _node = arena.add_view_node(Box::new(op.into_bridge(image.bands())));
    let pipeline = arena.compile().expect("smartcrop compile failed");
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .expect("scheduler construction failed")
        .run(&pipeline, &mut sink)
        .expect("smartcrop execution failed");
    Image::<U8>::from_buffer(
        pipeline.width,
        pipeline.height,
        image.bands(),
        sink.into_buffer(),
    )
    .expect("failed to materialize smartcrop output")
    .with_metadata(image.metadata().clone())
}

fn mean_abs_diff(lhs: &[u8], rhs: &[u8]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .sum::<f64>()
        / lhs.len() as f64
}

fn histogram_spread(image: &Image<U8>) -> usize {
    let mut seen = [false; 256];
    for sample in image.pixels() {
        seen[*sample as usize] = true;
    }
    seen.into_iter().filter(|present| *present).count()
}

fn mean_rect_diff(
    lhs: &Image<U8>,
    rhs: &Image<U8>,
    left: u32,
    top: u32,
    width: u32,
    height: u32,
) -> f64 {
    assert_eq!(
        (lhs.width(), lhs.height()),
        (rhs.width(), rhs.height()),
        "images must share dimensions for region diff"
    );
    let bands = lhs.bands().min(rhs.bands()) as usize;
    let mut total = 0.0;
    let mut count = 0usize;
    for y in top..top + height {
        for x in left..left + width {
            let lhs_idx = ((y * lhs.width() + x) * lhs.bands()) as usize;
            let rhs_idx = ((y * rhs.width() + x) * rhs.bands()) as usize;
            for band in 0..bands {
                total += (f64::from(lhs.pixels()[lhs_idx + band])
                    - f64::from(rhs.pixels()[rhs_idx + band]))
                .abs();
                count += 1;
            }
        }
    }
    total / count as f64
}

#[cfg(feature = "fft")]
fn run_phasecor(lhs: &[f64], rhs: &[f64], width: u32, height: u32) -> Vec<f64> {
    let op = PhasecorOp::<F64>::new();
    let region = Region::new(0, 0, width, height);
    let mut output = vec![0.0_f64; lhs.len()];
    let inputs = [cast_slice(lhs), cast_slice(rhs)];
    let input_regions = [region; 2];
    let mut state = op.dyn_start();
    op.dyn_process_region_multi(
        state.as_mut(),
        &inputs,
        cast_slice_mut(output.as_mut_slice()),
        &input_regions,
        region,
    );
    output
}

#[cfg(feature = "fft")]
fn run_freqmult(lhs: &[f64], rhs: &[f64], width: u32, height: u32) -> Vec<f64> {
    let op = FreqMultOp::<F64>::new();
    let region = Region::new(0, 0, width, height);
    let mut output = vec![0.0_f64; lhs.len()];
    let inputs = [cast_slice(lhs), cast_slice(rhs)];
    let input_regions = [region; 2];
    let mut state = op.dyn_start();
    op.dyn_process_region_multi(
        state.as_mut(),
        &inputs,
        cast_slice_mut(output.as_mut_slice()),
        &input_regions,
        region,
    );
    output
}

#[cfg(feature = "fft")]
fn peak_position(image: &[f64], width: usize) -> (usize, usize, f64) {
    image
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).expect("finite values"))
        .map(|(index, value)| (index % width, index / width, value))
        .expect("non-empty correlation image")
}

#[cfg(feature = "fft")]
fn checkerboard_f32(width: u32, height: u32) -> Image<F32> {
    let pixels = (0..height)
        .flat_map(|y| (0..width).map(move |x| if (x + y) % 2 == 0 { 1.0 } else { -1.0 }))
        .collect();
    Image::<F32>::from_buffer(width, height, 1, pixels).expect("failed to build f32 image")
}

#[cfg(feature = "fft")]
fn circular_shift(image: &Image<F32>, dx: usize, dy: usize) -> Image<F32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut shifted = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let src_x = (x + width - dx % width) % width;
            let src_y = (y + height - dy % height) % height;
            shifted[y * width + x] = image.pixels()[src_y * width + src_x];
        }
    }
    Image::<F32>::from_buffer(image.width(), image.height(), 1, shifted)
        .expect("failed to build shifted image")
}

fn alpha_is_preserved(image: &Image<U8>) -> bool {
    image.bands() == 4 && image.pixels().chunks_exact(4).any(|px| px[3] != 0)
}

fn run_mapim_u8(
    op: &MapImOp<U8>,
    source: &[u8],
    source_region: Region,
    source_bands: u32,
    index: &[f32],
    index_region: Region,
    output_region: Region,
) -> Vec<u8> {
    let mut output = vec![0u8; output_region.pixel_count() * source_bands as usize];
    let inputs = [source, cast_slice(index)];
    let input_regions = [source_region, index_region];
    let mut state = op.dyn_start();
    op.dyn_process_region_multi(
        state.as_mut(),
        &inputs,
        &mut output,
        &input_regions,
        output_region,
    );
    output
}

fn to_f32_unit_rgba(image: &Image<U8>) -> Vec<f32> {
    to_rgba_u8(image)
        .pixels()
        .iter()
        .map(|sample| f32::from(*sample) / 255.0)
        .collect()
}

fn run_composite_over_rgba(base: &Image<U8>, overlay: &Image<U8>) -> Image<U8> {
    let base = to_rgba_u8(base);
    let overlay = to_rgba_u8(overlay);
    let base_f32 = to_f32_unit_rgba(&base);
    let overlay_f32 = to_f32_unit_rgba(&overlay);
    let op =
        CompositeOp::<F32>::new(BlendMode::Over, false, 4).expect("composite op configuration");
    let region = Region::new(0, 0, base.width(), base.height());
    let mut output = vec![0.0f32; base_f32.len()];
    let inputs = [
        cast_slice(base_f32.as_slice()),
        cast_slice(overlay_f32.as_slice()),
    ];
    let input_regions = [region; 2];
    let mut state = op.dyn_start();
    op.dyn_process_region_multi(
        state.as_mut(),
        &inputs,
        cast_slice_mut(output.as_mut_slice()),
        &input_regions,
        region,
    );
    let pixels = output
        .into_iter()
        .map(|sample| (sample.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    Image::<U8>::from_buffer(base.width(), base.height(), 4, pixels)
        .expect("failed to build composite output")
        .with_metadata(ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        })
}

#[test]
#[cfg(all(feature = "jpeg", feature = "png"))]
fn flow1_full_thumbnail_pipeline_preserves_geometry_alpha_and_hashes() {
    let jpeg = load_u8_fixture("bench_2048x2048.jpg");
    let rgba = load_u8_fixture("bench_2048x2048_rgba.png");
    let small = load_u8_fixture("bench_512x512.jpg");

    let (thumb_pipeline, thumb_output) =
        execute_u8_pipeline_to_image(&jpeg, |builder| builder.thumbnail(thumbnail_config(400)));
    let thumb_output_repeat =
        execute_u8_pipeline_to_image(&jpeg, |builder| builder.thumbnail(thumbnail_config(400))).1;
    let (rgba_pipeline, rgba_output) =
        execute_u8_pipeline_to_image(&rgba, |builder| builder.thumbnail(thumbnail_config(256)));
    let rgba_output_repeat =
        execute_u8_pipeline_to_image(&rgba, |builder| builder.thumbnail(thumbnail_config(256))).1;
    let (upscale_pipeline, upscale_output) =
        execute_u8_pipeline_to_image(&small, |builder| builder.thumbnail(thumbnail_config(1600)));

    assert_thumbnail_dimensions(&thumb_pipeline, &jpeg, 400);
    assert_thumbnail_dimensions(&rgba_pipeline, &rgba, 256);
    assert_eq!(
        rgba_output.bands(),
        4,
        "RGBA thumbnails must preserve alpha bands"
    );
    assert!(alpha_is_preserved(&rgba_output));
    assert_eq!(
        upscale_pipeline.width, 1600,
        "upscale path should honour requested width"
    );
    assert_eq!(upscale_output.height(), 1600);
    assert!(
        thumb_output
            .pixels()
            .iter()
            .all(|sample| *sample <= u8::MAX)
    );
    assert_ne!(fnv1a64(thumb_output.pixels()), 0);
    assert_eq!(
        fnv1a64(thumb_output.pixels()),
        fnv1a64(thumb_output_repeat.pixels())
    );
    assert_eq!(
        fnv1a64(rgba_output.pixels()),
        fnv1a64(rgba_output_repeat.pixels())
    );
    assert_ne!(fnv1a64(upscale_output.pixels()), 0);
}

#[test]
#[cfg(feature = "jpeg")]
fn flow2_affine_transform_gauntlet_covers_identity_rotation_bounds_and_errors() {
    let image = load_u8_fixture("bench_512x512.jpg");
    let identity = execute_u8_pipeline_to_image(&image, |builder| {
        builder.affine(
            [1.0, 0.0, 0.0, 1.0],
            0.0,
            0.0,
            image.width(),
            image.height(),
            InterpolationKernel::Nearest,
        )
    })
    .1;
    assert_eq!(
        identity.pixels(),
        image.pixels(),
        "identity affine must be pixel exact"
    );

    let rotated = execute_u8_pipeline_to_image(&image, |builder| {
        builder.similarity(1.0, 45.0, InterpolationKernel::Bicubic)
    })
    .1;
    let rotated_repeat = execute_u8_pipeline_to_image(&image, |builder| {
        builder.similarity(1.0, 45.0, InterpolationKernel::Bicubic)
    })
    .1;
    assert_ne!(fnv1a64(rotated.pixels()), 0);
    assert_eq!(fnv1a64(rotated.pixels()), fnv1a64(rotated_repeat.pixels()));

    let background_fill = execute_u8_pipeline_to_image(&image, |builder| {
        builder.affine(
            [1.0, 0.0, 0.0, 1.0],
            200.0,
            150.0,
            image.width(),
            image.height(),
            InterpolationKernel::Bilinear,
        )
    })
    .1;
    assert!(
        background_fill.pixels()[0..16]
            .iter()
            .all(|sample| *sample == 0)
    );

    let tiny = execute_u8_pipeline_to_image(&image, |builder| {
        builder.affine(
            [100.0, 0.0, 0.0, 100.0],
            0.0,
            0.0,
            1,
            1,
            InterpolationKernel::Lanczos3,
        )
    })
    .1;
    assert_eq!((tiny.width(), tiny.height()), (1, 1));

    let degenerate = PipelineBuilder::from_source(memory_source_from_image(&image)).affine(
        [1.0, 2.0, 2.0, 4.0],
        0.0,
        0.0,
        image.width(),
        image.height(),
        InterpolationKernel::Bilinear,
    );
    assert!(matches!(
        degenerate,
        Err(BuildError::DegenerateAffineTransform { .. })
    ));
}

#[test]
#[cfg(feature = "fft")]
fn flow3_phase_correlation_registration_detects_known_offsets_and_non_power_of_two_inputs() {
    let reference = checkerboard_f32(32, 24);
    let shifted = circular_shift(&reference, 5, 7);
    let ref_fft = fwfft(&reference).expect("fwfft reference");
    let shifted_fft = fwfft(&shifted).expect("fwfft shifted");
    let cross_power = run_phasecor(
        shifted_fft.pixels(),
        ref_fft.pixels(),
        reference.width(),
        reference.height(),
    );
    let correlation = invfft(
        &Image::<F64>::from_buffer(
            reference.width(),
            reference.height(),
            COMPLEX_BANDS,
            cross_power,
        )
        .expect("correlation image"),
    )
    .expect("invfft correlation");
    let (peak_x, peak_y, peak_value) =
        peak_position(correlation.pixels(), reference.width() as usize);
    assert_eq!((peak_x, peak_y), (5, 7));
    assert!(peak_value > 0.5);

    let identical_fft = fwfft(&reference).expect("fwfft identical");
    let identical_cross = run_phasecor(
        identical_fft.pixels(),
        identical_fft.pixels(),
        reference.width(),
        reference.height(),
    );
    let identical_corr = invfft(
        &Image::<F64>::from_buffer(
            reference.width(),
            reference.height(),
            COMPLEX_BANDS,
            identical_cross,
        )
        .expect("identical corr"),
    )
    .expect("invfft identical");
    assert_eq!(
        peak_position(identical_corr.pixels(), reference.width() as usize).0,
        0
    );
    assert!(
        checkerboard_f32(31, 29)
            .pixels()
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
#[cfg(feature = "icc")]
fn flow4_icc_roundtrip_preserves_rgb_and_rejects_corrupt_profiles() {
    let image = load_u8_fixture("bench_512x512.jpg");
    let lab_profile = profile_load("lab").expect("load lab profile");
    let srgb_profile = profile_load("srgb").expect("load srgb profile");

    let lab = icc_transform(&image, &lab_profile, &IccTransformOptions::default())
        .expect("srgb -> lab")
        .as_f32()
        .expect("lab f32")
        .clone();
    let roundtrip = icc_transform(&lab, &srgb_profile, &IccTransformOptions::default())
        .expect("lab -> srgb")
        .as_u8()
        .expect("srgb u8")
        .clone();

    assert_eq!(roundtrip.bands(), 3);
    assert!(mean_abs_diff(image.pixels(), roundtrip.pixels()) < 2.0);
    assert!(
        icc_transform(
            &image,
            b"not-an-icc-profile",
            &IccTransformOptions::default()
        )
        .is_err()
    );
}

#[test]
#[cfg(feature = "png")]
fn flow5_smartcrop_stress_preserves_rgba_alpha_and_entropy_differs_from_centre() {
    let rgba = load_u8_fixture("bench_512x512_rgba.png");
    let gray = load_u8_fixture("bench_512x512_gray.png");

    let attention = apply_smartcrop(&rgba, 256, 256, Interesting::Attention);
    let entropy = apply_smartcrop(&gray, 200, 200, Interesting::Entropy);
    let centre = apply_smartcrop(&gray, 200, 200, Interesting::Centre);
    let identity = apply_smartcrop(&rgba, rgba.width(), rgba.height(), Interesting::Attention);
    let oversized = apply_smartcrop(
        &rgba,
        rgba.width() + 10,
        rgba.height() + 10,
        Interesting::Attention,
    );

    assert_eq!((attention.width(), attention.height()), (256, 256));
    assert!(alpha_is_preserved(&attention));
    assert_eq!(
        entropy.bands(),
        1,
        "entropy crop must preserve single-band grayscale"
    );
    assert_eq!(
        identity.pixels(),
        rgba.pixels(),
        "target == source must be identity"
    );
    assert_eq!(
        (oversized.width(), oversized.height()),
        (rgba.width(), rgba.height())
    );
    assert_ne!(fnv1a64(entropy.pixels()), fnv1a64(centre.pixels()));
}

#[test]
#[cfg(any(feature = "jpeg", feature = "png"))]
fn flow6_clahe_spreads_histograms_and_preserves_rgb_band_count() {
    let gray = load_u8_fixture("bench_512x512_gray.png");
    let rgb = load_u8_fixture("bench_512x512.jpg");

    let clahe_gray = run_u8_to_u8_op(
        &gray,
        ClaheOp::<U8>::new(64, 64, 3.0).expect("clahe gray"),
        1,
    );
    let clahe_rgb = run_u8_to_u8_op(&rgb, ClaheOp::<U8>::new(32, 32, 2.0).expect("clahe rgb"), 3);

    assert!(histogram_spread(&clahe_gray) >= histogram_spread(&gray));
    assert_eq!(clahe_rgb.bands(), rgb.bands());
    assert!(clahe_rgb.pixels().iter().all(|sample| *sample <= u8::MAX));
    assert!(ClaheOp::<U8>::new(0, 64, 1.0).is_err());
}

#[test]
#[cfg(feature = "fft")]
fn flow7_frequency_chain_roundtrips_and_low_pass_reduces_high_frequency_energy() {
    let image = checkerboard_f32(32, 32);
    let spectrum = fwfft(&image).expect("forward fft");
    let roundtrip = invfft(&spectrum).expect("inverse fft");
    let max_diff = image
        .pixels()
        .iter()
        .zip(roundtrip.pixels())
        .map(|(lhs, rhs)| (f64::from(*lhs) - *rhs).abs())
        .fold(0.0, f64::max);
    assert!(
        max_diff < 1e-6,
        "fft roundtrip must stay within floating-point tolerance"
    );

    let image_width = image.width();
    let image_height = image.height();
    let low_pass_mask: Vec<f64> = (0..image_height)
        .flat_map(|y| {
            (0..image_width).flat_map(move |x| {
                let dx = x as f64 - (image_width as f64 / 2.0);
                let dy = y as f64 - (image_height as f64 / 2.0);
                let weight = (-((dx * dx + dy * dy) / 32.0)).exp();
                [weight, 0.0]
            })
        })
        .collect();
    let filtered = run_freqmult(
        spectrum.pixels(),
        &low_pass_mask,
        image.width(),
        image.height(),
    );
    let filtered_image = invfft(
        &Image::<F64>::from_buffer(image.width(), image.height(), COMPLEX_BANDS, filtered)
            .expect("filtered fft image"),
    )
    .expect("inverse filtered fft");

    let original_energy: f64 = image
        .pixels()
        .iter()
        .map(|value| f64::from(*value).abs())
        .sum();
    let filtered_energy: f64 = filtered_image
        .pixels()
        .iter()
        .map(|value| value.abs())
        .sum();
    assert!(filtered_energy < original_energy);
    assert!(
        filtered_image
            .pixels()
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
#[cfg(feature = "png")]
fn flow8_mapim_identity_flip_and_background_fill_are_stable_for_rgba_inputs() {
    let source = load_u8_fixture("bench_512x512_rgba.png");
    let source_region = Region::new(0, 0, source.width(), source.height());
    let output_region = source_region;
    let identity_index: Vec<f32> = (0..source.height())
        .flat_map(|y| (0..source.width()).flat_map(move |x| [x as f32, y as f32]))
        .collect();
    let source_width = source.width();
    let flip_index: Vec<f32> = (0..source.height())
        .flat_map(|y| {
            (0..source_width).flat_map(move |x| [(source_width - 1 - x) as f32, y as f32])
        })
        .collect();
    let oob_index = vec![
        -10.0_f32,
        -10.0,
        source.width() as f32 + 10.0,
        source.height() as f32 + 10.0,
    ];

    let identity = run_mapim_u8(
        &MapImOp::<U8>::new(
            source.width(),
            source.height(),
            source.bands(),
            source.width(),
            source.height(),
            viprs::BandFormatId::F32,
        )
        .with_premultiplied(true),
        source.pixels(),
        source_region,
        source.bands(),
        &identity_index,
        output_region,
        output_region,
    );
    assert_eq!(identity, source.pixels());

    let flipped = run_mapim_u8(
        &MapImOp::<U8>::new(
            source.width(),
            source.height(),
            source.bands(),
            source.width(),
            source.height(),
            viprs::BandFormatId::F32,
        )
        .with_premultiplied(true),
        source.pixels(),
        source_region,
        source.bands(),
        &flip_index,
        output_region,
        output_region,
    );
    assert_eq!(
        &flipped[0..4],
        &source.pixels()
            [((source.width() - 1) * 4) as usize..((source.width() - 1) * 4 + 4) as usize]
    );

    let oob = run_mapim_u8(
        &MapImOp::<U8>::new(2, 2, 4, 2, 1, viprs::BandFormatId::F32)
            .with_extend(MapImExtend::Background)
            .with_premultiplied(true),
        &source.pixels()[0..16],
        Region::new(0, 0, 2, 2),
        4,
        &oob_index,
        Region::new(0, 0, 2, 1),
        Region::new(0, 0, 2, 1),
    );
    assert!(oob.iter().all(|sample| *sample == 0));
}

#[test]
fn flow9_filter_chain_canny_open_and_conva_remain_binary_and_reduce_noise() {
    let mut pixels = vec![0u8; 64 * 64];
    for y in 16..48 {
        for x in 16..48 {
            pixels[(y * 64 + x) as usize] = u8::MAX;
        }
    }
    pixels[(8 * 64 + 8) as usize] = u8::MAX;
    let image = Image::<U8>::from_buffer(64, 64, 1, pixels).expect("synthetic edge image");

    let canny = run_u8_to_f32_op(&image, Canny::<U8>::new(1.4), 1);
    let binary = Image::<U8>::from_buffer(
        canny.width(),
        canny.height(),
        1,
        canny
            .pixels()
            .iter()
            .map(|sample| if *sample > 0.0 { u8::MAX } else { 0 })
            .collect(),
    )
    .expect("binary canny image");
    let opened = run_u8_to_u8_op(
        &run_u8_to_u8_op(
            &binary,
            Erode::new(vec![vec![255; 3]; 3]).expect("erode mask"),
            1,
        ),
        Dilate::new(vec![vec![255; 3]; 3]).expect("dilate mask"),
        1,
    );
    let sobel = run_u8_to_u8_op(&image, EdgeOp::<U8>::sobel(), 1);
    let conva = run_f32_to_f32_op(
        &Image::<F32>::from_buffer(9, 9, 1, (0..81).map(|value| value as f32).collect())
            .expect("conva input"),
        ConvaOp::<F32>::with_mask(
            ConvolutionMask2d::from_coefficients(vec![vec![1.0; 7]; 7]).expect("mask"),
        )
        .expect("conva op"),
        1,
    );

    assert!(
        binary
            .pixels()
            .iter()
            .all(|sample| *sample == 0 || *sample == u8::MAX)
    );
    assert_eq!(
        opened.pixels()[(8 * 64 + 8) as usize],
        0,
        "morphological open should remove isolated noise"
    );
    assert!(
        sobel.pixels().iter().any(|sample| *sample > 0),
        "sobel should detect the synthetic edge"
    );
    assert!(conva.pixels().iter().all(|sample| sample.is_finite()));
}

#[test]
#[cfg(all(feature = "jpeg", feature = "png", feature = "icc"))]
fn flow10_composite_affine_icc_pipeline_keeps_output_visible_and_roundtrips_jpeg() {
    let base = load_u8_fixture("bench_512x512.jpg");
    let overlay = load_u8_fixture("bench_512x512_rgba.png");
    let watermark = load_u8_fixture("bench_512x512.jpg");
    let srgb_profile = profile_load("srgb").expect("load srgb");
    let lab_profile = profile_load("lab").expect("load lab");

    let lab = icc_transform(&base, &lab_profile, &IccTransformOptions::default())
        .expect("base to lab")
        .as_f32()
        .expect("lab output")
        .clone();
    let roundtripped = icc_transform(&lab, &srgb_profile, &IccTransformOptions::default())
        .expect("lab to srgb")
        .as_u8()
        .expect("srgb output")
        .clone();
    let transformed_overlay = execute_u8_pipeline_to_image(&overlay, |builder| {
        builder.similarity(0.5, 15.0, InterpolationKernel::Bilinear)
    })
    .1;
    let overlay_canvas = place_overlay_on_canvas(
        base.width(),
        base.height(),
        &to_rgba_u8(&transformed_overlay),
        100,
        100,
    );
    let watermark_thumb = execute_u8_pipeline_to_image(&watermark, |builder| {
        builder.thumbnail(thumbnail_config(64))
    })
    .1;
    let watermark_rgba = {
        let rgba = to_rgba_u8(&watermark_thumb);
        let mut pixels = rgba.pixels().to_vec();
        for pixel in pixels.chunks_exact_mut(4) {
            pixel[3] = 77;
        }
        Image::<U8>::from_buffer(rgba.width(), rgba.height(), 4, pixels)
            .expect("watermark rgba")
            .with_metadata(rgba.metadata().clone())
    };
    let watermark_canvas = place_overlay_on_canvas(
        base.width(),
        base.height(),
        &watermark_rgba,
        base.width() - 84,
        base.height() - 84,
    );

    let composited = run_composite_over_rgba(
        &run_composite_over_rgba(&roundtripped, &overlay_canvas),
        &watermark_canvas,
    );
    let cropped = apply_smartcrop(&composited, 320, 180, Interesting::Attention);
    let encoded = JpegCodec.encode(&cropped).expect("encode jpeg");
    let decoded = JpegCodec.decode::<U8>(&encoded).expect("decode jpeg");

    assert_eq!((decoded.width(), decoded.height()), (320, 180));
    assert!(
        mean_rect_diff(&cropped, &decoded, 236, 96, 64, 64) > 1.0,
        "watermark region must remain visible after jpeg roundtrip"
    );
    assert_ne!(fnv1a64(decoded.pixels()), 0);
}
