// ── Rotate90 ─────────────────────────────────────────────────────────────────

#[test]
fn rotate90_pipeline_changes_dimensions() {
    use viprs::{
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::memory::MemorySource,
        },
        domain::format::U8,
        ports::scheduler::TileScheduler,
    };

    // Landscape 4×2 (W=4, H=2). CW 90° → 2×4 output.
    // Formula: out(ox, oy) = in(x=oy, y=H-1-ox) with H=2.
    // Input:       row 0: [1, 2, 3, 4]   row 1: [5, 6, 7, 8]
    // Output (row-major, 2 wide × 4 tall):
    //   [5, 1, 6, 2, 7, 3, 8, 4]
    let source = MemorySource::<U8>::new(4, 2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let pipeline = viprs_runtime::pipeline::PipelineBuilder::from_source(source)
        .rotate90()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(
        pipeline.width, 2,
        "landscape: width must equal original height"
    );
    assert_eq!(
        pipeline.height, 4,
        "landscape: height must equal original width"
    );

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    assert_eq!(sink.into_buffer(), vec![5u8, 1, 6, 2, 7, 3, 8, 4]);

    // Square 4×4: dimensions unchanged, pixel positions rotate.
    // out(ox,oy) = in(x=oy, y=H-1-ox) = in(x=oy, y=3-ox) with H=4.
    // Expected (row-major): [13,9,5,1, 14,10,6,2, 15,11,7,3, 16,12,8,4]
    let source4 = MemorySource::<U8>::new(4, 4, 1, (1u8..=16).collect()).unwrap();
    let pipeline4 = viprs_runtime::pipeline::PipelineBuilder::from_source(source4)
        .rotate90()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline4.width, 4);
    assert_eq!(pipeline4.height, 4);

    let mut sink4 = MemorySink::for_pipeline(&pipeline4).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline4, &mut sink4)
        .unwrap();
    assert_eq!(
        sink4.into_buffer(),
        vec![13u8, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3, 16, 12, 8, 4]
    );
}

#[test]
fn thumbnail_width_only_end_to_end_sets_expected_dimensions() {
    // Validates that thumbnailing preserves non-zero gradient content after resize so
    // the test fails for zero-filled output buffers as well as wrong geometry.
    use viprs::{
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::memory::MemorySource,
        },
        domain::{
            format::U8,
            kernel::InterpolationKernel,
            ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
        },
        ports::scheduler::TileScheduler,
    };

    let input_width = 1024usize;
    let input_height = 768usize;
    let mut pixels = Vec::with_capacity(input_width * input_height);
    for _y in 0..input_height {
        for x in 0..input_width {
            pixels.push(((x * 255) / (input_width - 1)) as u8);
        }
    }
    let source =
        MemorySource::<U8>::new(input_width as u32, input_height as u32, 1, pixels).unwrap();
    let thumbnail = Thumbnail::new(ThumbnailTarget::Width(256), InterpolationKernel::Lanczos3);

    let pipeline = viprs_runtime::pipeline::PipelineBuilder::from_source(source)
        .thumbnail(thumbnail)
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.width, 256);
    assert_eq!(pipeline.height, 192);

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    assert_eq!(output.len(), 256 * 192);

    let mid_row = 96usize;
    let left = output[mid_row * 256];
    let center = output[mid_row * 256 + 128];
    let right = output[mid_row * 256 + 255];

    assert!(
        left < center,
        "gradient should rise across the row: {left} !< {center}"
    );
    assert!(
        center < right,
        "gradient should continue rising across the row: {center} !< {right}"
    );
    assert!(
        left <= 8,
        "left edge should stay near black after resize, got {left}"
    );
    assert!(
        (115..=140).contains(&center),
        "center sample should stay in the resized gradient range, got {center}"
    );
    assert!(
        right >= 247,
        "right edge should stay near white after resize, got {right}"
    );
}

fn run_thumbnail_sharpen_with_native_jpeg_shrink(input_size: u32, target_width: u32) {
    use std::num::NonZeroU8;
    use viprs::{
        ViprsError,
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::decoder_source::DecoderSource,
        },
        domain::{
            codec_options::LoadOptions,
            colorspace::ColorspaceId,
            format::{BandFormat, U8},
            image::Image,
            kernel::InterpolationKernel,
            ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
        },
        ports::{codec::ImageDecoder, scheduler::TileScheduler},
    };

    struct NativeShrinkRoundingDecoder {
        input_size: u32,
    }

    impl ImageDecoder for NativeShrinkRoundingDecoder {
        fn format_name(&self) -> &'static str {
            "jpeg"
        }

        fn sniff(&self, _: &[u8]) -> bool {
            true
        }

        fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
            self.decode_with_options(src, &LoadOptions::default())
        }

        fn decode_with_options<F: BandFormat>(
            &self,
            _: &[u8],
            opts: &LoadOptions,
        ) -> Result<Image<F>, ViprsError> {
            if F::ID != U8::ID {
                return Err(ViprsError::Codec(
                    "native shrink rounding decoder only supports U8".into(),
                ));
            }

            let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
            let actual_size = if factor > 1 {
                (self.input_size / u32::from(factor))
                    .saturating_sub(1)
                    .max(1)
            } else {
                self.input_size
            };
            let image = Image::from_buffer(
                actual_size,
                actual_size,
                3,
                vec![64u8; (actual_size * actual_size * 3) as usize],
            )
            .map_err(|err| ViprsError::Codec(err.to_string()))?;

            // SAFETY: F::ID == U8 implies F::Sample == u8 because BandFormat is sealed.
            let cast = unsafe { std::mem::transmute::<Image<U8>, Image<F>>(image) };
            Ok(cast)
        }

        fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
            Ok((self.input_size, self.input_size, 3))
        }
    }

    let source =
        DecoderSource::<_, U8>::new(NativeShrinkRoundingDecoder { input_size }, b"jpeg").unwrap();
    let pipeline = viprs_runtime::pipeline::PipelineBuilder::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(target_width),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .sharpen(0.5, 2.0, 10.0, 20.0, 0.0, 3.0)
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.width, target_width);
    assert_eq!(pipeline.height, target_width);

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    assert_eq!(
        sink.into_buffer().len(),
        (target_width * target_width * 3) as usize
    );
}

#[test]
fn thumbnail_then_sharpen_handles_benchmark_image_sizes() {
    for input_size in [512, 1024, 2048, 8192] {
        run_thumbnail_sharpen_with_native_jpeg_shrink(input_size, 400);
    }
}

#[test]
fn thumbnail_then_sharpen_handles_multiple_target_widths() {
    for target_width in [100, 200, 400, 800] {
        run_thumbnail_sharpen_with_native_jpeg_shrink(2048, target_width);
    }
}
