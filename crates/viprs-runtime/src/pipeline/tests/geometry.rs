use super::*;

/// View nodes must not allocate an extra buffer.
#[test]
fn extract_area_view_node_does_not_add_buffer() {
    use crate::sources::memory::MemorySource;
    // A pipeline with a single ExtractArea should have buffer_count == 1
    // (only the source buffer; no extra output buffer since it's a view node).
    let source = MemorySource::<U8>::new(4, 4, 1, vec![0u8; 16]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .extract_area(0, 0, 2, 2)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(
        pipeline.buffer_count, 1,
        "view-only pipeline must have exactly 1 buffer (shared source + view)"
    );
    assert_eq!(pipeline.width, 2);
    assert_eq!(pipeline.height, 2);
}

/// View node followed by a transform must use the same input buffer index.
#[test]
fn extract_area_then_transform_buffer_layout() {
    use crate::sources::memory::MemorySource;
    let source = MemorySource::<U8>::new(8, 8, 1, vec![0u8; 64]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .extract_area(0, 0, 4, 4)
        .unwrap()
        .invert()
        .unwrap()
        .build()
        .unwrap();
    // Node 0: view (input_bufs=[0], output_buf=0)
    // Node 1: transform (input_bufs=[0], output_buf=1)
    assert_eq!(pipeline.nodes[0].input_bufs(), &[0]);
    assert_eq!(pipeline.nodes[0].output_buf(), 0);
    assert_eq!(pipeline.nodes[1].input_bufs(), &[0]);
    assert_eq!(pipeline.nodes[1].output_buf(), 1);
    assert_eq!(pipeline.buffer_count, 2);
}

/// `compile()` propagates output dimensions automatically via `output_width`/`output_height`.
///
/// This test verifies that `PipelineBuilder::extract_area` does NOT need to call
/// `arena.set_dimensions` manually — the dimension propagation loop in `compile()`
/// produces the correct `width`/`height` on the `CompiledPipeline`.
#[test]
fn extract_area_dimensions_propagated_automatically_by_compile() {
    use crate::sources::memory::MemorySource;
    // 16x8 source; crop to 6x3 starting at (2, 1).
    let source = MemorySource::<U8>::new(16, 8, 1, vec![0u8; 128]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .extract_area(2, 1, 6, 3)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(
        pipeline.width, 6,
        "compile() must propagate extract_area width automatically"
    );
    assert_eq!(
        pipeline.height, 3,
        "compile() must propagate extract_area height automatically"
    );
}

fn patterned_extract_area_source(width: u32, height: u32) -> (Image<U8>, MemorySource<U8>) {
    let pixels = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                [
                    ((x * 3 + y * 5) % 251) as u8,
                    ((x * 7 + y * 11 + 13) % 251) as u8,
                    ((x * 17 + y * 19 + 29) % 251) as u8,
                ]
            })
        })
        .collect::<Vec<_>>();
    let image = Image::<U8>::from_buffer(width, height, 3, pixels).unwrap();
    let source = MemorySource::<U8>::new(width, height, 3, image.pixels().to_vec())
        .unwrap()
        .with_metadata(image.metadata().clone());
    (image, source)
}

fn expected_crop_pixels(
    image: &Image<U8>,
    left: u32,
    top: u32,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let bands = image.bands() as usize;
    let image_width = image.width() as usize;
    let crop_width = width as usize;
    let mut expected = Vec::with_capacity(crop_width * height as usize * bands);

    for row in 0..height as usize {
        let src_start = ((top as usize + row) * image_width + left as usize) * bands;
        let src_end = src_start + crop_width * bands;
        expected.extend_from_slice(&image.pixels()[src_start..src_end]);
    }

    expected
}

fn run_extract_area(left: u32, top: u32, width: u32, height: u32) -> (Image<U8>, Image<U8>) {
    let (image, source) = patterned_extract_area_source(400, 300);
    let pipeline = PipelineBuilder::from_source(source)
        .extract_area(left, top, width, height)
        .unwrap()
        .build()
        .unwrap();
    let output = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(2).unwrap())
        .unwrap();
    (image, output)
}

#[test]
fn extract_area_origin_crop_is_identity() {
    let (image, output) = run_extract_area(0, 0, 400, 300);
    assert_eq!(output.width(), image.width());
    assert_eq!(output.height(), image.height());
    assert_eq!(output.pixels(), image.pixels());
}

#[test]
fn extract_area_uses_requested_offset_for_top_left_pixel() {
    let (image, output) = run_extract_area(100, 50, 200, 200);
    let expected = expected_crop_pixels(&image, 100, 50, 200, 200);
    assert_eq!(&output.pixels()[..3], &expected[..3]);
    assert_eq!(output.pixels(), expected);
}

#[test]
fn extract_area_near_edge_reads_requested_region() {
    let (image, output) = run_extract_area(225, 120, 175, 180);
    let expected = expected_crop_pixels(&image, 225, 120, 175, 180);
    assert_eq!(&output.pixels()[..3], &expected[..3]);
    assert_eq!(output.pixels(), expected);
}

#[test]
fn extract_area_then_embed_uses_requested_offset() {
    use crate::domain::ops::conversion::ExtendMode;

    let image = Image::<U8>::from_buffer(4, 4, 1, (0u8..16).collect()).unwrap();
    let source = MemorySource::<U8>::new(4, 4, 1, image.pixels().to_vec())
        .unwrap()
        .with_metadata(image.metadata().clone());
    let pipeline = PipelineBuilder::from_source(source)
        .extract_area(2, 1, 2, 2)
        .unwrap()
        .embed(4, 4, 2, 1, 2, 2, ExtendMode::Black)
        .unwrap()
        .build()
        .unwrap();

    let output = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(2).unwrap())
        .unwrap();
    assert_eq!(
        output.pixels(),
        &[0, 0, 0, 0, 0, 0, 6, 7, 0, 0, 10, 11, 0, 0, 0, 0]
    );
}

#[test]
fn extract_area_rejects_crops_larger_than_current_stage() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<U8>::new(7, 5, 1, vec![0u8; 35]).unwrap();
    let result = PipelineBuilder::from_source(source).extract_area(0, 0, 11, 8);

    assert!(matches!(
        result,
        Err(BuildError::InvalidExtractAreaParameters {
            x: 0,
            y: 0,
            width: 11,
            height: 8,
            image_width: 7,
            image_height: 5,
        })
    ));
}

#[test]
fn extract_area_rejects_zero_width_crops() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<U8>::new(7, 5, 1, vec![0u8; 35]).unwrap();
    let result = PipelineBuilder::from_source(source).extract_area(0, 0, 0, 4);

    assert!(matches!(
        result,
        Err(BuildError::InvalidExtractAreaParameters {
            x: 0,
            y: 0,
            width: 0,
            height: 4,
            image_width: 7,
            image_height: 5,
        })
    ));
}

#[test]
fn extract_area_rejects_zero_height_crops() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<U8>::new(7, 5, 1, vec![0u8; 35]).unwrap();
    let result = PipelineBuilder::from_source(source).extract_area(0, 0, 4, 0);

    assert!(matches!(
        result,
        Err(BuildError::InvalidExtractAreaParameters {
            x: 0,
            y: 0,
            width: 4,
            height: 0,
            image_width: 7,
            image_height: 5,
        })
    ));
}

/// `NodeSpec::identity` is the default for all pixel-local ops — buffer sizes
/// must not regress when `node_spec` is not overridden.
#[test]
fn node_spec_identity_default_matches_old_buffer_sizing() {
    use crate::sources::memory::MemorySource;
    // 4x4 single-band U8 pipeline with one Invert op.
    let source = MemorySource::<U8>::new(4, 4, 1, vec![0u8; 16]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .invert()
        .unwrap()
        .build()
        .unwrap();
    let tile_w = pipeline.demand_hint.tile_width(pipeline.width) as usize;
    let tile_h = pipeline
        .demand_hint
        .tile_height(pipeline.width, pipeline.height) as usize;
    let expected = tile_w * tile_h; // bands=1, bps=1 (U8)
    for &size in &pipeline.buffer_sizes {
        if size > 0 {
            assert_eq!(
                size, expected,
                "NodeSpec::identity must produce the same buffer sizes as the old uniform sizing"
            );
        }
    }
}

#[test]
fn affine_bilinear_source_buffer_matches_required_input_region() {
    use crate::{
        domain::{kernel::InterpolationKernel, ops::resample::affine::Affine},
        sources::memory::MemorySource,
    };

    let source = MemorySource::<U8>::new(512, 512, 1, vec![0u8; 512 * 512]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .affine(
            [1.0, 0.0, 0.0, 1.0],
            0.0,
            0.0,
            512,
            512,
            InterpolationKernel::Bilinear,
        )
        .unwrap()
        .build()
        .unwrap();

    let tile_w = pipeline.demand_hint.tile_width(pipeline.width);
    let tile_h = pipeline
        .demand_hint
        .tile_height(pipeline.width, pipeline.height);
    let required = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Bilinear,
        512,
        512,
    )
    .required_input_region(&Region::new(0, 0, tile_w, tile_h));

    assert!(
        pipeline.buffer_sizes[0] >= required.pixel_count(),
        "source buffer must cover affine required_input_region for one scheduler tile"
    );
}

#[test]
fn gauss_blur_chain_source_buffer_matches_backpropagated_region() {
    use crate::{
        domain::{
            op::OperationBridge,
            ops::convolution::{GaussBlurH, GaussBlurV},
        },
        sources::memory::MemorySource,
    };

    let source = MemorySource::<F32>::new(64, 64, 1, vec![0.0f32; 64 * 64]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new(
            GaussBlurH::<F32>::new(1.5),
            1,
        )))
        .unwrap()
        .then(Box::new(OperationBridge::new(
            GaussBlurV::<F32>::new(1.5),
            1,
        )))
        .unwrap()
        .build()
        .unwrap();

    let tile_w = pipeline.demand_hint.tile_width(pipeline.width);
    let tile_h = pipeline
        .demand_hint
        .tile_height(pipeline.width, pipeline.height);
    let output_region = Region::new(0, 0, tile_w, tile_h);
    let required = GaussBlurH::<F32>::new(1.5)
        .required_input_region(&GaussBlurV::<F32>::new(1.5).required_input_region(&output_region));

    assert!(
        pipeline.buffer_sizes[0] >= required.pixel_count() * std::mem::size_of::<f32>(),
        "source buffer must cover the fully back-propagated region for a GaussBlurH → GaussBlurV chain"
    );
}

/// `PipelineBuilder::embed` propagates dst dimensions to the compiled pipeline.
#[test]
fn embed_dimensions_propagated_by_compile() {
    use crate::{domain::ops::conversion::embed::ExtendMode, sources::memory::MemorySource};
    // 4×4 source; embed into 8×8 canvas at (2, 2).
    let source = MemorySource::<U8>::new(4, 4, 1, vec![0u8; 16]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .embed(8, 8, 2, 2, 4, 4, ExtendMode::Black)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(
        pipeline.width, 8,
        "embed must propagate dst_width to pipeline output dimensions"
    );
    assert_eq!(
        pipeline.height, 8,
        "embed must propagate dst_height to pipeline output dimensions"
    );
}

/// `PipelineBuilder::embed` with same src and dst dimensions and zero offset
/// must preserve the source format and band count.
#[test]
fn embed_format_and_bands_preserved() {
    use crate::{domain::ops::conversion::embed::ExtendMode, sources::memory::MemorySource};
    let source = MemorySource::<U8>::new(4, 4, 3, vec![0u8; 48]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .embed(4, 4, 0, 0, 4, 4, ExtendMode::Black)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(pipeline.output_format, BandFormatId::U8);
    assert_eq!(pipeline.output_bands, 3);
}

#[test]
fn embed_signed_accepts_negative_offsets() {
    use crate::{domain::ops::conversion::embed::ExtendMode, sources::memory::MemorySource};
    let source = MemorySource::<U8>::new(4, 4, 1, vec![0u8; 16]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .embed_signed(6, 6, -1, -1, 4, 4, ExtendMode::Copy)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(pipeline.width, 6);
    assert_eq!(pipeline.height, 6);
}

#[test]
fn embed_rejects_unsigned_offsets_beyond_i32_range() {
    use crate::{domain::ops::conversion::embed::ExtendMode, sources::memory::MemorySource};

    let source = MemorySource::<U8>::new(4, 4, 1, vec![0u8; 16]).unwrap();
    let result = PipelineBuilder::from_source(source).embed(
        4,
        4,
        i32::MAX as u32 + 1,
        0,
        4,
        4,
        ExtendMode::Black,
    );

    assert!(matches!(
        result,
        Err(BuildError::InvalidEmbedParameters {
            message: "unsigned embed offsets must fit within i32",
        })
    ));
}

#[test]
fn embed_with_gravity_centres_the_source() {
    use crate::{
        domain::ops::conversion::embed::{ExtendMode, Gravity},
        sources::memory::MemorySource,
    };

    let image = Image::<U8>::from_buffer(2, 1, 1, vec![5u8, 6]).unwrap();
    let source = MemorySource::<U8>::new(2, 1, 1, image.pixels().to_vec())
        .unwrap()
        .with_metadata(image.metadata().clone());
    let pipeline = PipelineBuilder::from_source(source)
        .embed_with_gravity(4, 3, Gravity::Centre, 2, 1, ExtendMode::Black)
        .unwrap()
        .build()
        .unwrap();

    let output = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(2).unwrap())
        .unwrap();
    assert_eq!(output.width(), 4);
    assert_eq!(output.height(), 3);
    assert_eq!(output.pixels(), &[0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0]);
}
