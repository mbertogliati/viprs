mod chaos_monkey_7 {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        thread,
    };

    use bytemuck::Pod;
    use viprs::{
      BuildError, CompiledPipeline, InMemoryImage, ImageMetadata, Interpretation, U8,
      adapters::{
          pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
          sinks::memory::MemorySink, sources::memory::MemorySource,
        },
      domain::{
            colorspace::{ColorspaceId, Lab, SRgb, Ucs},
            kernel::InterpolationKernel,
            ops::{
                conversion::ExtendMode,
                resample::{Resize, Thumbnail, thumbnail::ThumbnailTarget},
            },
        },
      ports::scheduler::TileScheduler,
    };

    const SHARPEN_X1: f32 = 2.0;
    const SHARPEN_Y2: f32 = 10.0;
    const SHARPEN_Y3: f32 = 20.0;
    const SHARPEN_M1: f32 = 0.0;
    const SHARPEN_M2: f32 = 3.0;

    fn rgb_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        }
    }

    fn patterned_rgb_u8(width: u32, height: u32) -> InMemoryImage<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 13 + 19) % 256) as u8);
                pixels.push(((x * 11 + y * 29 + 7) % 256) as u8);
                pixels.push(((x * 5 + y * 19 + 191) % 256) as u8);
            }
        }

        InMemoryImage::from_buffer(width, height, 3, pixels)
            .unwrap()
            .with_metadata(rgb_metadata())
    }

    fn zero_band_u8(width: u32, height: u32) -> InMemoryImage<U8> {
        InMemoryImage::from_buffer(width, height, 0, Vec::new()).unwrap()
    }

    fn memory_source_from_image<F>(image: &InMemoryImage<F>) -> MemorySource<F>
    where
        F: viprs::BandFormat,
        F::Sample: Pod,
    {
        MemorySource::new(
            image.width(),
            image.height(),
            image.bands(),
            image.pixels().to_vec(),
        )
        .unwrap()
        .with_metadata(image.metadata().clone())
    }

    fn execute_same_format<F, S: viprs::pipeline::Commit>(
      image: &InMemoryImage<F>,
      configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(CompiledPipeline, InMemoryImage<F>), String>
    where
        F: viprs::BandFormat,
        F::Sample: Pod,
    {
        let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(
            image,
        )))
        .map_err(|error| format!("stage failed: {error:?}"))?
        .build()
        .map_err(|error| format!("build failed: {error:?}"))?;

        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(2)
            .map_err(|error| format!("scheduler construction failed: {error}"))?
            .run(&pipeline, &mut sink)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        let output = sink
            .into_image::<F>(
                pipeline.width,
                pipeline.height,
                pipeline.output_bands,
                image.metadata().clone(),
            )
            .map_err(|error| format!("failed to materialize output: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn thumbnail(width: u32) -> Thumbnail {
        Thumbnail::new(ThumbnailTarget::Width(width), InterpolationKernel::Lanczos3)
    }

    fn assert_u8_pixels_within_tolerance(expected: &[u8], actual: &[u8], tolerance: u8) {
        assert_eq!(expected.len(), actual.len());
        for (index, (&lhs, &rhs)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = lhs.abs_diff(rhs);
            assert!(
                diff <= tolerance,
                "pixel mismatch at index {index}: expected {lhs}, got {rhs}, tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn zero_band_colourspace_returns_typed_error() {
        let image = zero_band_u8(8, 8).with_metadata(rgb_metadata());
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            ImagePipeline::from_source(memory_source_from_image(&image))
                .with_colorspace(ColorspaceId::SRgb)
                .colourspace::<Lab>()
        }));

        let result = match outcome {
            Ok(result) => result,
            Err(_) => panic!("zero-band colourspace should return a typed error, not panic"),
        };

        assert!(
            result.is_err(),
            "zero-band colourspace unexpectedly succeeded"
        );
    }

    #[test]
    fn colourspace_chain_srgb_lab_cmc_lab_srgb_roundtrip_stays_close() {
        let image = patterned_rgb_u8(64, 64);
        let (_pipeline, output) = execute_same_format(&image, |builder| {
            builder
                .with_colorspace(ColorspaceId::SRgb)
                .colourspace::<Lab>()?
                .colourspace::<Ucs>()?
                .colourspace::<Lab>()?
                .colourspace::<SRgb>()
        })
        .expect("colourspace chain should succeed");

        assert_u8_pixels_within_tolerance(image.pixels(), output.pixels(), 2);
    }
}
