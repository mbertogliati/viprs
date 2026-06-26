mod chaos_monkey_9 {
    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, F32, Image, ImageMetadata, Interpretation, U8,
        adapters::{scheduler::rayon_scheduler::RayonScheduler, sources::memory::MemorySource},
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
    use viprs::{adapters::codecs::PngCodec, ports::codec::ImageDecoder};
    #[cfg(feature = "png")]

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

    fn run_pipeline<F, S: viprs_runtime::pipeline::internal::Flush>(
        image: &Image<F>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelineBuilder,
        ) -> Result<
            viprs_runtime::pipeline::internal::PipelineBuilder<S>,
            BuildError,
        >,
    ) -> Result<(CompiledPipeline, Image<F>), String>
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
    fn reduceh_factor_one_is_identity() {
        let image = patterned_rgb_u8(9, 3);
        let op = ReduceH::<U8>::new(1.0, InterpolationKernel::Nearest)
            .unwrap()
            .with_input_width(image.width());
        let input_region = viprs::Region::new(0, 0, image.width(), image.height());
        let output_region = viprs::Region::new(0, 0, image.width(), image.height());
        let input = viprs::Tile::new(input_region, image.bands(), image.pixels());
        let mut output_pixels = vec![0u8; image.pixels().len()];
        let mut output = viprs::TileMut::new(output_region, image.bands(), &mut output_pixels);
        let mut state = op.start_with_tile(8192, image.height());

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_pixels, image.pixels());
    }
}
