mod chaos_monkey_9 {
    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, F32, Image, ImageMetadata, Interpretation, U8,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sources::memory::MemorySource,
        },
        domain::{
            colorspace::{ColorspaceId, Lab, SRgb},
            kernel::InterpolationKernel,
            op::{NodeSpec, Op},
            ops::{conversion::ExtendMode, resample::ReduceH},
        },
    };

    #[cfg(feature = "png")]
    use png::{BitDepth, ColorType, Encoder};
    #[cfg(feature = "jpeg")]
    use std::sync::Arc;
    #[cfg(feature = "jpeg")]
    use viprs::adapters::codecs::JpegCodec;
    #[cfg(feature = "jpeg")]
    use viprs::ports::codec::ImageEncoder;
    #[cfg(feature = "png")]
    use viprs::{adapters::codecs::PngCodec, ports::codec::ImageDecoder};

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

    fn patterned_rgb_f32(width: u32, height: u32) -> Image<F32> {
        let image = patterned_rgb_u8(width, height);
        let pixels = image
            .pixels()
            .iter()
            .map(|&value| f32::from(value))
            .collect::<Vec<_>>();

        Image::from_buffer(width, height, 3, pixels)
            .unwrap()
            .with_metadata(rgb_metadata())
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

    fn run_pipeline<F, S: viprs::pipeline::Flush>(
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

        let scheduler = RayonScheduler::new(2).map_err(|error| error.to_string())?;
        let output = pipeline
            .run_to_image::<F, _>(&scheduler)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;
        Ok((pipeline, output))
    }

    #[cfg(feature = "png")]
    fn indexed_png_bytes() -> Vec<u8> {
        let mut output = Vec::new();
        let mut encoder = Encoder::new(&mut output, 2, 1);
        encoder.set_color(ColorType::Indexed);
        encoder.set_depth(BitDepth::Eight);
        encoder.set_palette(vec![255, 0, 0, 0, 255, 0]);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[0, 1]).unwrap();
        drop(writer);
        output
    }

    #[cfg(feature = "png")]
    #[test]
    fn colourspace_to_lab_updates_output_metadata_and_invalidates_source_icc() {
        let mut metadata = rgb_metadata();
        metadata.icc_profile = Some((0u8..32).collect());
        let image = patterned_rgb_u8(2, 2).with_metadata(metadata);

        let pipeline = PipelineBuilder::from_source(memory_source_from_image(&image))
            .with_colorspace(ColorspaceId::SRgb)
            .colourspace::<Lab>()
            .unwrap()
            .build()
            .unwrap();
        let output = pipeline
            .run_to_image::<F32, _>(&RayonScheduler::new(1).unwrap())
            .unwrap();

        assert_eq!(output.metadata().interpretation, Some(Interpretation::Lab));
        assert!(output.metadata().icc_profile.is_none());
    }
}
