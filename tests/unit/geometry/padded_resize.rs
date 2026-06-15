mod chaos_monkey_7 {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        thread,
    };

    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, Image, ImageMetadata, Interpretation, U8,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
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

    fn patterned_rgb_u8(width: u32, height: u32) -> Image<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 13 + 19) % 256) as u8);
                pixels.push(((x * 11 + y * 29 + 7) % 256) as u8);
                pixels.push(((x * 5 + y * 19 + 191) % 256) as u8);
            }
        }

        Image::from_buffer(width, height, 3, pixels)
            .unwrap()
            .with_metadata(rgb_metadata())
    }

    fn zero_band_u8(width: u32, height: u32) -> Image<U8> {
        Image::from_buffer(width, height, 0, Vec::new()).unwrap()
    }

    fn memory_source_from_image<F>(image: &Image<F>) -> MemorySource<F>
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

    fn execute_same_format<F>(
        image: &Image<F>,
        configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder, BuildError>,
    ) -> Result<(CompiledPipeline, Image<F>), String>
    where
        F: viprs::BandFormat,
        F::Sample: Pod,
    {
        let pipeline = configure(PipelineBuilder::from_source(memory_source_from_image(
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
    fn embed_then_sharpen_keeps_padded_border_valid() {
        let image = patterned_rgb_u8(11, 9);
        let (_pipeline, output) = execute_same_format(&image, |builder| {
            builder
                .embed(
                    21,
                    19,
                    5,
                    5,
                    image.width(),
                    image.height(),
                    ExtendMode::Background(vec![0.0, 0.0, 0.0]),
                )?
                .with_colorspace(ColorspaceId::SRgb)
                .sharpen(
                    1.5, SHARPEN_X1, SHARPEN_Y2, SHARPEN_Y3, SHARPEN_M1, SHARPEN_M2,
                )
        })
        .expect("embed then sharpen should succeed");

        assert_eq!((output.width(), output.height()), (21, 19));
        assert_eq!(&output.pixels()[0..3], &[0, 0, 0]);
    }

    #[test]
    fn resize_exact_one_is_identity_after_embed_chain() {
        let image = patterned_rgb_u8(17, 13);
        let (_pipeline, output) = execute_same_format(&image, |builder| {
            builder
                .embed(
                    21,
                    17,
                    2,
                    2,
                    image.width(),
                    image.height(),
                    ExtendMode::Background(vec![3.0, 5.0, 7.0]),
                )?
                .resize(Resize::new(1.0, 1.0, InterpolationKernel::Lanczos3))
        })
        .expect("resize(1.0) after embed should succeed");

        assert_eq!((output.width(), output.height()), (21, 17));
        assert_eq!(&output.pixels()[0..3], &[3, 5, 7]);
    }
}
