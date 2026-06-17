mod chaos_monkey_8 {
    use std::sync::Arc;
    use std::thread;

    use bytemuck::Pod;
    use viprs::{
        BandFormat, BandFormatId, BuildError, CompiledPipeline, HistFindOp, Image, ImageMetadata,
        Interpretation, U8, ViprsError,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
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

    #[cfg(feature = "png")]
    use viprs::{
        adapters::codecs::PngCodec,
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

    fn execute_pipeline_to_image<FIn, FOut, S: viprs::pipeline::Flush>(
        image: &Image<FIn>,
        configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Image<FOut>), String>
    where
        FIn: BandFormat,
        FOut: BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let pipeline = configure(PipelineBuilder::from_source(memory_source_from_image(
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
            .into_image::<FOut>(
                pipeline.width,
                pipeline.height,
                pipeline.output_bands,
                image.metadata().clone(),
            )
            .map_err(|error| format!("failed to materialize output image: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn execute_pipeline_to_buffer<F, S: viprs::pipeline::Flush>(
        image: &Image<F>,
        configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Vec<u8>), String>
    where
        F: BandFormat,
        F::Sample: Pod,
    {
        let pipeline = configure(PipelineBuilder::from_source(memory_source_from_image(
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
    fn hsv_roundtrip_preserves_rgba_alpha() {
        let image = patterned_u8(9, 6, 4);
        let (_pipeline, output) = execute_pipeline_to_image::<U8, U8, _>(&image, |builder| {
            builder
                .with_colorspace(ColorspaceId::SRgb)
                .colourspace::<Hsv>()?
                .colourspace::<SRgb>()
        })
        .expect("RGBA HSV roundtrip should succeed");

        for (input, output) in image
            .pixels()
            .chunks_exact(4)
            .zip(output.pixels().chunks_exact(4))
        {
            assert_eq!(
                output[3], input[3],
                "alpha channel must be preserved exactly"
            );
            for band in 0..3 {
                let diff = (i16::from(output[band]) - i16::from(input[band])).unsigned_abs();
                assert!(diff <= 1, "rgb band {band} drifted by {diff}");
            }
        }
    }

    #[test]
    fn colourspace_to_lab_rejects_one_band_images_with_typed_error() {
        let image = patterned_u8(8, 8, 1);
        let result = PipelineBuilder::from_source(memory_source_from_image(&image))
            .with_colorspace(ColorspaceId::SRgb)
            .colourspace::<viprs::domain::colorspace::Lab>();

        assert!(
            result.is_err(),
            "one-band sRGB to Lab unexpectedly succeeded"
        );
    }

    #[test]
    fn hsv_arithmetic_roundtrip_stays_in_u8_range() {
        let image = patterned_u8(13, 9, 3);
        let (_pipeline, output) = execute_pipeline_to_image::<U8, U8, _>(&image, |builder| {
            builder
                .with_colorspace(ColorspaceId::SRgb)
                .colourspace::<Hsv>()?
                .linear(1.75, 15.0)?
                .colourspace::<SRgb>()
        })
        .expect("HSV arithmetic pipeline should succeed");

        assert_eq!(output.width(), image.width());
        assert_eq!(output.height(), image.height());
        assert_eq!(output.bands(), image.bands());
        assert_eq!(
            output.pixels().len(),
            output.width() as usize * output.height() as usize * output.bands() as usize
        );
    }

    #[test]
    fn hsv_source_image_after_histogram_still_builds_pipeline() {
        let pixels = hsv_pixels(4, 3);
        let image =
            Image::<viprs::F32>::from_buffer(4, 3, 3, pixels).expect("HSV image must build");
        let region = Region::new(0, 0, image.width(), image.height());
        let tile = Tile::<viprs::F32>::new(region, image.bands(), image.pixels());
        let reducer = HistFindOp::for_format(image.bands(), None, u8::MAX as u32);
        let partial = reducer.reduce_tile(&tile, &region);
        let histogram = <HistFindOp as TileReducer<viprs::F32>>::finalize(&reducer, partial);

        assert!(histogram.total() > 0);

        let (_pipeline, output) =
            execute_pipeline_to_image::<viprs::F32, U8, _>(&image, |builder| {
                builder
                    .with_colorspace(ColorspaceId::Hsv)
                    .colourspace::<SRgb>()
            })
            .expect("HSV image should remain usable after histogram computation");
        assert_eq!(output.bands(), 3);
    }
}
