mod chaos_monkey_14 {
    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, Image, ImageMetadata, Interpretation, U8,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
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

    fn patterned_u8(width: u32, height: u32, bands: u32) -> Image<U8> {
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

        Image::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(srgb_metadata())
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

    fn execute_to_image<FIn, FOut>(
        image: &Image<FIn>,
        configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder, BuildError>,
    ) -> Result<(CompiledPipeline, Image<FOut>), String>
    where
        FIn: viprs::BandFormat,
        FOut: viprs::BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let pipeline = configure(PipelineBuilder::from_source(memory_source_from_image(
            image,
        )))
        .map_err(|error| format!("stage failed: {error:?}"))?
        .build()
        .map_err(|error| format!("build failed: {error:?}"))?;
        let output = pipeline
            .run_to_image::<FOut, _>(&RayonScheduler::new(2).map_err(|error| error.to_string())?)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;
        Ok((pipeline, output))
    }

    #[test]
    #[ignore = "BUG: embed() accepts src_width/src_height that do not match the current stage"]
    fn embed_with_mismatched_source_dimensions_returns_typed_error() {
        let image = patterned_u8(5, 4, 4);
        let result = PipelineBuilder::from_source(memory_source_from_image(&image)).embed(
            8,
            8,
            1,
            1,
            image.width() - 1,
            image.height(),
            ExtendMode::Black,
        );

        assert!(matches!(
            result,
            Err(BuildError::InvalidEmbedParameters { .. })
        ));
    }
}
