mod chaos_monkey {
    use bytemuck::Pod;
    use proptest::prelude::*;
    use viprs::{
        BuildError, F32, Image, ImageMetadata, Interpretation, U8, U16,
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::memory::MemorySource,
        },
        domain::{
            colorspace::{ColorspaceId, Lab, SRgb},
            kernel::InterpolationKernel,
            ops::resample::{Resize, Thumbnail, thumbnail::ThumbnailTarget},
        },
        ports::scheduler::TileScheduler,
    };

    fn rgb_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        }
    }

    fn make_u8_image(width: u32, height: u32, bands: u32, pixels: Vec<u8>) -> Image<U8> {
        Image::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(if bands >= 3 {
                rgb_metadata()
            } else {
                ImageMetadata::default()
            })
    }

    fn make_u16_image(width: u32, height: u32, bands: u32, pixels: Vec<u16>) -> Image<U16> {
        Image::from_buffer(width, height, bands, pixels).unwrap()
    }

    fn make_f32_image(
        width: u32,
        height: u32,
        bands: u32,
        pixels: Vec<f32>,
        metadata: ImageMetadata,
    ) -> Image<F32> {
        Image::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(metadata)
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

    fn execute_to_image<F, S: viprs_runtime::pipeline::internal::Flush>(
        image: &Image<F>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelineBuilder,
        ) -> Result<
            viprs_runtime::pipeline::internal::PipelineBuilder<S>,
            BuildError,
        >,
    ) -> Result<(viprs_runtime::pipeline::CompiledPipeline, Image<F>), String>
    where
        F: viprs::BandFormat,
        F::Sample: Pod,
    {
        let pipeline = configure(
            viprs_runtime::pipeline::internal::PipelineBuilder::from_source(
                memory_source_from_image(image),
            ),
        )
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

    fn execute_to_buffer<F, S: viprs_runtime::pipeline::internal::Flush>(
        image: &Image<F>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelineBuilder,
        ) -> Result<
            viprs_runtime::pipeline::internal::PipelineBuilder<S>,
            BuildError,
        >,
    ) -> Result<(viprs_runtime::pipeline::CompiledPipeline, Vec<u8>), String>
    where
        F: viprs::BandFormat,
        F::Sample: Pod,
    {
        let pipeline = configure(
            viprs_runtime::pipeline::internal::PipelineBuilder::from_source(
                memory_source_from_image(image),
            ),
        )
        .map_err(|error| format!("stage failed: {error:?}"))?
        .build()
        .map_err(|error| format!("build failed: {error:?}"))?;

        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(2)
            .map_err(|error| format!("scheduler construction failed: {error}"))?
            .run(&pipeline, &mut sink)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, sink.into_buffer()))
    }

    fn thumbnail(width: u32) -> Thumbnail {
        Thumbnail::new(ThumbnailTarget::Width(width), InterpolationKernel::Lanczos3)
    }

    fn patterned_rgb_u8(width: u32, height: u32) -> Image<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 13) % 256) as u8);
                pixels.push(((x * 11 + y * 29 + 7) % 256) as u8);
                pixels.push(((x * 5 + y * 19 + 191) % 256) as u8);
            }
        }
        make_u8_image(width, height, 3, pixels)
    }

    proptest! {
        #[test]
        fn pass1_double_invert_is_identity_random_rgb_u8(
            pixels in prop::collection::vec(any::<u8>(), 100 * 100 * 3)
        ) {
            let image = make_u8_image(100, 100, 3, pixels);
            let (_pipeline, output) = execute_to_image(&image, |builder| builder.invert()?.invert())
                .map_err(TestCaseError::fail)?;

            prop_assert_eq!(output.width(), image.width());
            prop_assert_eq!(output.height(), image.height());
            prop_assert_eq!(output.pixels(), image.pixels());
        }
    }

    #[test]
    fn pass4_linear_clamps_u8_boundary_values() {
        let image = make_u8_image(1, 1, 1, vec![200]);
        let (_pipeline, output) = execute_to_image(&image, |builder| builder.linear(2.0, 0.0))
            .expect("linear should execute on U8");

        assert_eq!(output.pixels(), &[255]);
    }

    #[test]
    fn pass4_invert_u16_max_maps_to_zero() {
        let image = make_u16_image(1, 1, 1, vec![u16::MAX]);
        let (_pipeline, output) = execute_to_image(&image, |builder| builder.invert())
            .expect("invert should execute on U16");

        assert_eq!(output.pixels(), &[0]);
    }

    #[test]
    fn pass4_invert_f32_nan_preserves_nan_without_panicking() {
        let image = make_f32_image(1, 1, 1, vec![f32::NAN], ImageMetadata::default());
        let (_pipeline, output) = execute_to_image(&image, |builder| builder.invert())
            .expect("invert should not panic on NaN");

        assert!(output.pixels()[0].is_nan());
    }
}
