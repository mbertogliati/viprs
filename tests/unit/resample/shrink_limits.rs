mod chaos_monkey_19 {
    use bytemuck::Pod;
    use viprs::{
      BuildError, F32, InMemoryImage, ImageMetadata, Interpretation, U8, U16,
      adapters::{pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler},
      domain::{
            colorspace::{ColorspaceId, SRgb},
            kernel::InterpolationKernel,
            op::Op,
            ops::{
                arithmetic::{Matrix, RecombOp},
                conversion::embed::ExtendMode,
                resample::{Thumbnail, thumbnail::ThumbnailTarget},
            },
        },
    };

    fn srgb_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        }
    }

    fn make_u8_image(width: u32, height: u32, bands: u32, pixels: Vec<u8>) -> InMemoryImage<U8> {
        let image = InMemoryImage::from_buffer(width, height, bands, pixels).unwrap();
        if bands >= 3 {
            image.with_metadata(srgb_metadata())
        } else {
            image
        }
    }

    fn make_u16_image(width: u32, height: u32, bands: u32, pixels: Vec<u16>) -> InMemoryImage<U16> {
        InMemoryImage::from_buffer(width, height, bands, pixels).unwrap()
    }

    fn make_f32_image(
        width: u32,
        height: u32,
        bands: u32,
        pixels: Vec<f32>,
        metadata: ImageMetadata,
    ) -> InMemoryImage<F32> {
        InMemoryImage::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(metadata)
    }

    fn execute_to_image<FIn, FOut, S: viprs::pipeline::Commit>(
      image: &InMemoryImage<FIn>,
      configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(viprs::CompiledPipeline, InMemoryImage<FOut>), String>
    where
        FIn: viprs::BandFormat,
        FOut: viprs::BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let pipeline = configure(ImagePipeline::from_source(
            viprs::adapters::sources::memory::MemorySource::<FIn>::new(
                image.width(),
                image.height(),
                image.bands(),
                image.pixels().to_vec(),
            )
            .unwrap()
            .with_metadata(image.metadata().clone()),
        ))
        .map_err(|error| format!("stage failed: {error:?}"))?
        .build()
        .map_err(|error| format!("build failed: {error:?}"))?;

        let output = pipeline
            .run_to_image::<FOut, _>(
                &RayonScheduler::new(2).map_err(|error| format!("scheduler failed: {error}"))?,
            )
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn patterned_rgb(width: u32, height: u32) -> InMemoryImage<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 13 + 3) % 256) as u8);
                pixels.push(((x * 11 + y * 29 + 7) % 256) as u8);
                pixels.push(((x * 5 + y * 19 + 191) % 256) as u8);
            }
        }
        make_u8_image(width, height, 3, pixels)
    }

    fn rgba_image_with_partial_alpha() -> InMemoryImage<U8> {
        make_u8_image(2, 1, 4, vec![200, 100, 50, 128, 30, 60, 90, 64])
    }

    #[test]
    #[ignore = "B-330"]
    fn thumbnail_zero_band_image_returns_typed_error() {
        let image = make_u8_image(4, 4, 0, Vec::new());
        let result = execute_to_image::<U8, U8, _>(&image, |builder| {
            builder.thumbnail(Thumbnail::new(
                ThumbnailTarget::Width(2),
                InterpolationKernel::Lanczos3,
            ))
        });

        match result {
            Ok((pipeline, output)) => panic!(
                "0-band thumbnail should fail gracefully, but built {}x{} {}-band output with {} bytes",
                pipeline.width,
                pipeline.height,
                pipeline.output_bands,
                output.pixels().len()
            ),
            Err(error) => assert!(
                error.contains("Build")
                    || error.contains("BuildError")
                    || error.contains("Unsupported")
                    || error.contains("Invalid")
                    || error.contains("pipeline execution failed"),
                "expected typed error, got {error}"
            ),
        }
    }

    #[test]
    fn shrink_h_max_factor_produces_single_output_pixel() {
        let image = make_u8_image(65_535, 1, 1, vec![255; 65_535]);
        let (pipeline, output) =
            execute_to_image::<U8, U8, _>(&image, |builder| builder.shrink_h(65_535))
                .expect("shrink_h(65535) should succeed");

        assert_eq!((pipeline.width, pipeline.height), (1, 1));
        assert_eq!(output.pixels(), &[255]);
    }

    #[test]
    fn reduce_factor_one_is_identity() {
        let image = patterned_rgb(11, 7);
        let (_pipeline, output) = execute_to_image::<U8, U8, _>(&image, |builder| {
            builder.reduce(1.0, 1.0, InterpolationKernel::Lanczos3)
        })
        .expect("reduce(1.0) should succeed");

        assert_eq!(output.width(), image.width());
        assert_eq!(output.height(), image.height());
        assert_eq!(output.pixels(), image.pixels());
    }
}
