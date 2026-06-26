mod chaos_monkey_8 {
    use std::sync::Arc;
    use std::thread;

    use bytemuck::Pod;
    use viprs::{
        BandFormat, BandFormatId, BuildError, CompiledPipeline, HistFindOp, ImageMetadata,
        InMemoryImage, Interpretation, U8, ViprsError,
        adapters::{
            pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
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

    fn patterned_u8(width: u32, height: u32, bands: u32) -> InMemoryImage<U8> {
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

        let image = InMemoryImage::from_buffer(width, height, bands, pixels)
            .expect("pattern image must build");
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

    fn memory_source_from_image<F>(image: &InMemoryImage<F>) -> MemorySource<F>
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

    fn execute_pipeline_to_image<FIn, FOut, S: viprs::pipeline::Commit>(
        image: &InMemoryImage<FIn>,
        configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(CompiledPipeline, InMemoryImage<FOut>), String>
    where
        FIn: BandFormat,
        FOut: BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(image)))
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

    fn execute_pipeline_to_buffer<F, S: viprs::pipeline::Commit>(
        image: &InMemoryImage<F>,
        configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Vec<u8>), String>
    where
        F: BandFormat,
        F::Sample: Pod,
    {
        let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(image)))
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
    fn resize_then_extract_area_overlap_edge_returns_typed_error() {
        let image = patterned_u8(2, 2, 3);
        let result = ImagePipeline::from_source(memory_source_from_image(&image))
            .resize(Resize::new(50.0, 50.0, InterpolationKernel::Nearest))
            .and_then(|builder| builder.extract_area(99, 99, 10, 10));

        assert!(matches!(
            result,
            Err(BuildError::InvalidExtractAreaParameters { .. })
        ));
    }

    #[test]
    fn embed_negative_offsets_succeed_when_source_overlaps_canvas() {
        let image = patterned_u8(4, 4, 1);
        let (pipeline, _output) = execute_pipeline_to_image::<U8, U8, _>(&image, |builder| {
            builder.embed_signed(4, 4, -1, -1, 4, 4, ExtendMode::Black)
        })
        .expect("embed_signed with negative offsets must succeed when source overlaps canvas");

        assert_eq!(pipeline.width, 4);
        assert_eq!(pipeline.height, 4);
    }

    #[test]
    fn embed_negative_offsets_return_error_when_source_outside_canvas() {
        let image = patterned_u8(4, 4, 1);
        let result = execute_pipeline_to_image::<U8, U8, _>(&image, |builder| {
            builder.embed_signed(4, 4, -4, -4, 4, 4, ExtendMode::Black)
        });

        assert!(
            result.is_err(),
            "embed with source completely outside canvas must fail"
        );
    }
}
