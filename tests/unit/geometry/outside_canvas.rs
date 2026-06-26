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
    fn embed_entirely_outside_canvas_returns_typed_error() {
        let image = make_u8_image(2, 2, 1, vec![1, 2, 3, 4]);
        let result = execute_to_image::<U8, U8, _>(&image, |builder| {
            builder.embed(4, 4, 4, 0, image.width(), image.height(), ExtendMode::Black)
        });

        assert!(
            matches!(result, Err(error) if error.contains("InvalidEmbedParameters") || error.contains("invalid embed parameters")),
            "expected typed embed error"
        );
    }
}
