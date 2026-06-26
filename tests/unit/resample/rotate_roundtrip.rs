mod chaos_monkey_14 {
    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, ImageMetadata, InMemoryImage, Interpretation, U8,
        adapters::{
            pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
            sources::memory::MemorySource,
        },
        domain::ops::conversion::ExtendMode,
    };

    #[cfg(feature = "jpeg")]
    use viprs::adapters::codecs::JpegCodec;
    #[cfg(feature = "jpeg")]
    use viprs::{
        domain::codec_options::SaveOptions,
        ports::codec::{ImageDecoder, ImageEncoder},
    };

    fn srgb_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        }
    }

    fn patterned_u8(width: u32, height: u32, bands: u32) -> InMemoryImage<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * bands as usize);
        for y in 0..height {
            for x in 0..width {
                for band in 0..bands {
                    let value = match band {
                        0 => ((x * 17 + y * 13 + 19) % 251) as u8,
                        1 => ((x * 11 + y * 29 + 7) % 253) as u8,
                        2 => ((x * 5 + y * 19 + 191) % 255) as u8,
                        _ => ((x * 23 + y * 31 + band * 47 + 3) % 249) as u8,
                    };
                    pixels.push(value);
                }
            }
        }

        InMemoryImage::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(srgb_metadata())
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

    fn execute_to_image<FIn, FOut, S: viprs::pipeline::Commit>(
        image: &InMemoryImage<FIn>,
        configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(CompiledPipeline, InMemoryImage<FOut>), String>
    where
        FIn: viprs::BandFormat,
        FOut: viprs::BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(image)))
            .map_err(|error| format!("stage failed: {error:?}"))?
            .build()
            .map_err(|error| format!("build failed: {error:?}"))?;
        let output = pipeline
            .run_to_image::<FOut, _>(&RayonScheduler::new(2).map_err(|error| error.to_string())?)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;
        Ok((pipeline, output))
    }

    #[test]
    fn rotate270_then_rotate90_is_pixel_identity() {
        let image = patterned_u8(9, 7, 4);
        let (_pipeline, output) =
            execute_to_image::<U8, U8, _>(&image, |builder| builder.rotate270()?.rotate90())
                .expect("rotate270 then rotate90 should succeed");

        assert_eq!((output.width(), output.height(), output.bands()), (9, 7, 4));
        assert_eq!(output.pixels(), image.pixels());
    }
}
