mod chaos_monkey_8 {
    use std::sync::Arc;
    use std::thread;

    use bytemuck::Pod;
    use viprs::{
        BandFormat, BandFormatId, BuildError, CompiledPipeline, HistFindOp, Image, ImageMetadata,
        Interpretation, U8, ViprsError,
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::memory::MemorySource,
        },
        domain::{
            colorspace::{ColorspaceId, Hsv, SRgb},
            image::{Region, Tile},
            kernel::InterpolationKernel,
            ops::{
                conversion::ExtendMode,
                resample::{Resize, Thumbnail, thumbnail::ThumbnailTarget},
            },
            reducer::TileReducer,
        },
        ports::scheduler::TileScheduler,
    };

    use viprs::{
        adapters::codecs::PngCodec,
        ports::codec::{ImageDecoder, ImageEncoder},
    };
    #[cfg(feature = "png")]

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
                        0 => ((x * 17 + y * 13 + 19) % 256) as u8,
                        1 => ((x * 11 + y * 29 + 7) % 256) as u8,
                        2 => ((x * 5 + y * 19 + 191) % 256) as u8,
                        _ => ((x * 23 + y * 31 + band * 47 + 3) % 256) as u8,
                    };
                    pixels.push(value);
                }
            }
        }

        let image =
            Image::from_buffer(width, height, bands, pixels).expect("pattern image must build");
        if bands >= 3 {
            image.with_metadata(srgb_metadata())
        } else {
            image
        }
    }

    fn hsv_pixels(width: u32, height: u32) -> Vec<f32> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 61 + y * 17) % 360) as f32);
                pixels.push(((x + y) % 10) as f32 / 9.0);
                pixels.push((1 + ((x * 3 + y * 5) % 9)) as f32 / 10.0);
            }
        }
        pixels
    }

    fn memory_source_from_image<F>(image: &Image<F>) -> MemorySource<F>
    where
        F: BandFormat,
        F::Sample: Pod,
    {
        MemorySource::new(
            image.width(),
            image.height(),
            image.bands(),
            image.pixels().to_vec(),
        )
        .expect("memory source construction must succeed")
        .with_metadata(image.metadata().clone())
    }

    fn execute_pipeline_to_image<FIn, FOut, S: viprs_runtime::pipeline::internal::Flush>(
        image: &Image<FIn>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelineBuilder,
        ) -> Result<
            viprs_runtime::pipeline::internal::PipelineBuilder<S>,
            BuildError,
        >,
    ) -> Result<(CompiledPipeline, Image<FOut>), String>
    where
        FIn: BandFormat,
        FOut: BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let pipeline = configure(
            viprs_runtime::pipeline::internal::PipelineBuilder::from_source(
                memory_source_from_image(image),
            ),
        )
        .map_err(|error| format!("stage failed: {error:?}"))?
        .build()
        .map_err(|error| format!("build failed: {error:?}"))?;

        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(2)
            .map_err(|error| format!("scheduler construction failed: {error}"))?
            .run(&pipeline, &mut sink)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        let output = sink
            .into_image::<FOut>(
                pipeline.width,
                pipeline.height,
                pipeline.output_bands,
                image.metadata().clone(),
            )
            .map_err(|error| format!("failed to materialize output image: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn execute_pipeline_to_buffer<F, S: viprs_runtime::pipeline::internal::Flush>(
        image: &Image<F>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelineBuilder,
        ) -> Result<
            viprs_runtime::pipeline::internal::PipelineBuilder<S>,
            BuildError,
        >,
    ) -> Result<(CompiledPipeline, Vec<u8>), String>
    where
        F: BandFormat,
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

        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(2)
            .map_err(|error| format!("scheduler construction failed: {error}"))?
            .run(&pipeline, &mut sink)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, sink.into_buffer()))
    }

    fn bytes_per_sample(format: BandFormatId) -> usize {
        match format {
            BandFormatId::U8 => 1,
            BandFormatId::U16 | BandFormatId::I16 => 2,
            BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
            BandFormatId::F64 => 8,
        }
    }

    fn assert_valid_buffer(pipeline: &CompiledPipeline, buffer: &[u8]) {
        let expected = pipeline.width as usize
            * pipeline.height as usize
            * pipeline.output_bands as usize
            * bytes_per_sample(pipeline.output_format);
        assert_eq!(buffer.len(), expected);
    }

    #[cfg(feature = "png")]
    #[test]
    fn resize_1x1_by_100_produces_uniform_image() {
        let image = Image::from_buffer(1, 1, 3, vec![17, 89, 203])
            .expect("1x1 image must build")
            .with_metadata(srgb_metadata());
        let (_pipeline, output) = execute_pipeline_to_image::<U8, U8, _>(&image, |builder| {
            builder.resize(Resize::new(100.0, 100.0, InterpolationKernel::Nearest))
        })
        .expect("1x1 upscale should succeed");

        assert_eq!((output.width(), output.height()), (100, 100));
        for pixel in output.pixels().chunks_exact(3) {
            assert_eq!(pixel, &[17, 89, 203]);
        }
    }

    #[test]
    fn resize_2x2_by_100_keeps_expected_corners() {
        let image =
            Image::from_buffer(2, 2, 1, vec![11, 22, 33, 44]).expect("2x2 image must build");
        let (_pipeline, output) = execute_pipeline_to_image::<U8, U8, _>(&image, |builder| {
            builder.resize(Resize::new(100.0, 100.0, InterpolationKernel::Nearest))
        })
        .expect("2x2 upscale should succeed");

        assert_eq!((output.width(), output.height()), (200, 200));
        let top_left = output.pixels()[0];
        let top_right = output.pixels()[199];
        let bottom_left = output.pixels()[199 * 200];
        let bottom_right = output.pixels()[200 * 200 - 1];
        assert_eq!(
            (top_left, top_right, bottom_left, bottom_right),
            (11, 22, 33, 44)
        );
    }

    #[test]
    fn extreme_affine_cases_produce_valid_buffers() {
        let image = patterned_u8(17, 13, 3);
        let cases = [
            ([1.0, 57.289_961_630_759_144, 0.0, 1.0], 0.0, 0.0, 32, 32),
            ([0.001, 0.0, 0.0, 0.001], 0.0, 0.0, 16, 16),
            ([1000.0, 0.0, 0.0, 1000.0], 0.0, 0.0, 16, 16),
        ];

        for (matrix, tx, ty, out_w, out_h) in cases {
            let (pipeline, buffer) = execute_pipeline_to_buffer(&image, |builder| {
                builder.affine(matrix, tx, ty, out_w, out_h, InterpolationKernel::Nearest)
            })
            .expect("extreme affine pipeline should succeed");

            assert_eq!((pipeline.width, pipeline.height), (out_w, out_h));
            assert_valid_buffer(&pipeline, &buffer);
        }
    }
}
