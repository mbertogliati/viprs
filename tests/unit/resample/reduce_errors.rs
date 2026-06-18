mod chaos_monkey_12 {
    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, F32, Image, ImageMetadata, Interpretation, OperationBridge,
        RecombOp, U8,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            colorspace::Lab,
            kernel::InterpolationKernel,
            ops::{
                arithmetic::Matrix,
                conversion::ExtendMode,
                histogram::HistEqualOp,
                resample::{Thumbnail, thumbnail::ThumbnailTarget},
            },
            reducer::TileReducer,
            reducers::HistEqualReducer,
        },
        ports::scheduler::TileScheduler,
    };

    #[cfg(feature = "jpeg")]
    use std::sync::Arc;
    #[cfg(feature = "jpeg")]
    use viprs::adapters::codecs::JpegCodec;
    #[cfg(feature = "jpeg")]
    use viprs::ports::codec::{ImageDecoder, ImageEncoder};

    fn rgb_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        }
    }

    fn gray_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::BW),
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

    fn equalized_gray_u8() -> Image<U8> {
        let pixels = (0u8..=255).collect::<Vec<_>>();
        Image::from_buffer(16, 16, 1, pixels)
            .unwrap()
            .with_metadata(gray_metadata())
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

    fn execute_to_image<F, S: viprs::pipeline::Flush>(
        image: &Image<F>,
        configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder<S>, BuildError>,
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

        let output = pipeline
            .run_to_image::<F, _>(&RayonScheduler::new(2).map_err(|error| error.to_string())?)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn execute_to_image_with_output<InputF, OutputF, S: viprs::pipeline::Flush>(
        image: &Image<InputF>,
        configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Image<OutputF>), String>
    where
        InputF: viprs::BandFormat,
        InputF::Sample: Pod,
        OutputF: viprs::BandFormat,
    {
        let pipeline = configure(PipelineBuilder::from_source(memory_source_from_image(
            image,
        )))
        .map_err(|error| format!("stage failed: {error:?}"))?
        .build()
        .map_err(|error| format!("build failed: {error:?}"))?;

        let output = pipeline
            .run_to_image::<OutputF, _>(&RayonScheduler::new(2).map_err(|error| error.to_string())?)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn execute_to_buffer<F, S: viprs::pipeline::Flush>(
        image: &Image<F>,
        configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Vec<u8>), String>
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
            .map_err(|error| error.to_string())?
            .run(&pipeline, &mut sink)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, sink.into_buffer()))
    }

    fn thumbnail(width: u32) -> Thumbnail {
        Thumbnail::new(ThumbnailTarget::Width(width), InterpolationKernel::Lanczos3)
    }

    #[test]
    fn rotate90_twice_matches_rotate180_on_non_square_image() {
        let image = patterned_rgb_u8(7, 5);

        let (_pipeline, twice) = execute_to_image(&image, |builder| builder.rotate90()?.rotate90())
            .expect("double rotate90 should succeed");
        let (_pipeline, rotate180) = execute_to_image(&image, |builder| builder.rotate180())
            .expect("rotate180 should succeed");

        assert_eq!((twice.width(), twice.height()), (7, 5));
        assert_eq!(twice.pixels(), rotate180.pixels());
    }

    #[test]
    fn reduce_below_one_returns_typed_error_instead_of_panicking() {
        let image = patterned_rgb_u8(64, 64);

        let outcome = std::panic::catch_unwind(|| {
            execute_to_buffer(&image, |builder| {
                builder.reduce(0.5, 0.5, InterpolationKernel::Bilinear)
            })
        });

        match outcome {
            Ok(Ok((_pipeline, _buffer))) => panic!("reduce(0.5, 0.5) unexpectedly succeeded"),
            Ok(Err(message)) => assert!(
                message.contains("Invalid")
                    || message.contains("factor")
                    || message.contains("reduce"),
                "reduce<1 should surface a typed validation error, got: {message}"
            ),
            Err(_) => panic!("reduce(0.5, 0.5) panicked instead of returning a typed error"),
        }
    }
}
