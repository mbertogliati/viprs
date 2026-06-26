mod chaos_monkey_5 {
    use bytemuck::Pod;
    use viprs::{
      BuildError, CompiledPipeline, F32, InMemoryImage, ImageMetadata, Interpretation, U8,
      adapters::{
          pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
          sinks::memory::MemorySink, sources::memory::MemorySource,
        },
      domain::{
            colorspace::{ColorspaceId, Lab, SRgb},
            kernel::InterpolationKernel,
            ops::{
                conversion::ExtendMode, resample::Thumbnail, resample::thumbnail::ThumbnailTarget,
            },
        },
      ports::scheduler::TileScheduler,
    };

    const SHARPEN_X1: f32 = 2.0;
    const SHARPEN_Y2: f32 = 10.0;
    const SHARPEN_Y3: f32 = 20.0;

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

    fn palette_rgb_u8(width: u32, height: u32) -> InMemoryImage<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(x as u8);
                pixels.push(y as u8);
                pixels.push(((x * 73 + y * 19) & 0xff) as u8);
            }
        }

        InMemoryImage::from_buffer(width, height, 3, pixels)
            .unwrap()
            .with_metadata(rgb_metadata())
    }

    fn patterned_rgb_f32(width: u32, height: u32) -> InMemoryImage<F32> {
        let image = patterned_rgb_u8(width, height);
        let pixels = image
            .pixels()
            .iter()
            .map(|&value| f32::from(value))
            .collect::<Vec<_>>();

        InMemoryImage::from_buffer(width, height, 3, pixels)
            .unwrap()
            .with_metadata(rgb_metadata())
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

    fn thumbnail(width: u32) -> Thumbnail {
        Thumbnail::new(ThumbnailTarget::Width(width), InterpolationKernel::Lanczos3)
    }

    #[test]
    fn nested_embed_compounds_dimensions_correctly() {
        let image = patterned_rgb_u8(3, 2);
        let (pipeline, output) = execute_to_image(&image, |builder| {
            builder
                .embed(5, 4, 1, 1, image.width(), image.height(), ExtendMode::Black)?
                .embed(8, 7, 2, 3, 5, 4, ExtendMode::Black)
        })
        .expect("nested embed should succeed");

        assert_eq!((pipeline.width, pipeline.height), (8, 7));
        assert_eq!((output.width(), output.height()), (8, 7));
    }
}
