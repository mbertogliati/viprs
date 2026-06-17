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
    fn no_op_pipeline_builds_identity_output() {
        let image = patterned_rgb_u8(4, 3);
        let source = memory_source_from_image(&image);
        let pipeline = PipelineBuilder::from_source(source).build().unwrap();
        let output = pipeline
            .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
            .unwrap();

        assert_eq!(output.pixels(), image.pixels());
    }

    #[test]
    fn oversized_tile_proxy_full_image_hint_handles_small_images() {
        struct FullImagePass;

        impl Op for FullImagePass {
            type Input = U8;
            type Output = U8;
            type State = ();

            fn demand_hint(&self) -> viprs::DemandHint {
                viprs::DemandHint::FullImage
            }

            fn required_input_region(&self, region: &viprs::Region) -> viprs::Region {
                *region
            }

            fn node_spec(&self, _tile_w: u32, _tile_h: u32) -> NodeSpec {
                NodeSpec::identity(8192, 8192)
            }

            fn start(&self) {}

            fn process_region(
                &self,
                _state: &mut (),
                input: &viprs::Tile<U8>,
                output: &mut viprs::TileMut<U8>,
            ) {
                output.data.copy_from_slice(input.data);
            }
        }

        let image = patterned_rgb_u8(512, 512);
        let pipeline = PipelineBuilder::from_source(memory_source_from_image(&image))
            .then(Box::new(viprs::OperationBridge::new(
                FullImagePass,
                image.bands(),
            )))
            .unwrap()
            .build()
            .unwrap();
        let output = pipeline
            .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
            .unwrap();

        assert_eq!(output.pixels(), image.pixels());
    }
}
