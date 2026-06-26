mod chaos_monkey_12 {
    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, F32, InMemoryImage, ImageMetadata, Interpretation, OperationBridge,
        RecombOp, U8,
        adapters::{
          pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
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

    fn equalized_gray_u8() -> InMemoryImage<U8> {
        let pixels = (0u8..=255).collect::<Vec<_>>();
        InMemoryImage::from_buffer(16, 16, 1, pixels)
            .unwrap()
            .with_metadata(gray_metadata())
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

    fn execute_to_image<F, S: viprs::pipeline::Commit>(
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

        let output = pipeline
            .run_to_image::<F, _>(&RayonScheduler::new(2).map_err(|error| error.to_string())?)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn execute_to_image_with_output<InputF, OutputF, S: viprs::pipeline::Commit>(
        image: &InMemoryImage<InputF>,
        configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(CompiledPipeline, InMemoryImage<OutputF>), String>
    where
        InputF: viprs::BandFormat,
        InputF::Sample: Pod,
        OutputF: viprs::BandFormat,
    {
        let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(
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

    fn execute_to_buffer<F, S: viprs::pipeline::Commit>(
        image: &InMemoryImage<F>,
        configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Vec<u8>), String>
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
            .map_err(|error| error.to_string())?
            .run(&pipeline, &mut sink)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, sink.into_buffer()))
    }

    fn thumbnail(width: u32) -> Thumbnail {
        Thumbnail::new(ThumbnailTarget::Width(width), InterpolationKernel::Lanczos3)
    }

    #[test]
    fn embed_exact_size_canvas_is_identity() {
        let image = patterned_rgb_u8(19, 13);

        let (_pipeline, output) = execute_to_image(&image, |builder| {
            builder.embed(
                image.width(),
                image.height(),
                0,
                0,
                image.width(),
                image.height(),
                ExtendMode::Black,
            )
        })
        .expect("embed identity should succeed");

        assert_eq!(output.pixels(), image.pixels());
    }

    #[test]
    fn extract_area_then_embed_preserves_cropped_pixels_at_same_origin() {
        let image = patterned_rgb_u8(100, 100);

        let (_pipeline, output) = execute_to_image(&image, |builder| {
            builder
                .extract_area(0, 0, 50, 50)?
                .embed(100, 100, 0, 0, 50, 50, ExtendMode::Black)
        })
        .expect("extract then embed should succeed");

        let mut expected = vec![0u8; image.pixels().len()];
        for y in 0..50usize {
            let src_start = y * 100 * 3;
            let src_end = src_start + 50 * 3;
            let dst_start = y * 100 * 3;
            expected[dst_start..dst_start + 50 * 3]
                .copy_from_slice(&image.pixels()[src_start..src_end]);
        }

        assert_eq!(output.pixels(), expected);
    }
}
