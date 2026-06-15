mod chaos_monkey_13 {
    use bytemuck::Pod;
    use viprs::{
        BandFormat, BandFormatId, BuildError, CompiledPipeline, Image, ImageMetadata,
        Interpretation, Tile, TileMut, U8,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            colorspace::{ColorspaceId, Lab},
            image::Region,
            kernel::InterpolationKernel,
            op::{Op, OperationBridge},
            ops::{
                arithmetic::{Matrix, RecombOp},
                histogram::{HistEqualOp, HistMatchOp},
                resample::{Thumbnail, thumbnail::ThumbnailTarget},
            },
            reducer::TileReducer,
            reducers::HistEqualReducer,
        },
        ports::scheduler::TileScheduler,
    };

    #[cfg(all(feature = "png", feature = "jpeg", feature = "webp"))]
    use std::{sync::Arc, thread};
    #[cfg(all(feature = "png", feature = "jpeg", feature = "webp"))]
    use viprs::{
        adapters::codecs::{JpegCodec, PngCodec, WebpCodec},
        ports::codec::ImageEncoder,
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

    fn run_builder_to_image<FOut>(
        builder: PipelineBuilder,
        metadata: ImageMetadata,
    ) -> Result<(CompiledPipeline, Image<FOut>), String>
    where
        FOut: BandFormat,
        FOut::Sample: Pod,
    {
        let pipeline = builder
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
                metadata,
            )
            .map_err(|error| format!("failed to materialize output image: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn execute_pipeline_to_image<FIn, FOut>(
        image: &Image<FIn>,
        configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder, BuildError>,
    ) -> Result<(CompiledPipeline, Image<FOut>), String>
    where
        FIn: BandFormat,
        FOut: BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let builder = configure(PipelineBuilder::from_source(memory_source_from_image(
            image,
        )))
        .map_err(|error| format!("stage failed: {error:?}"))?;
        run_builder_to_image(builder, image.metadata().clone())
    }

    fn cumulative_histogram(data: &[u8]) -> Vec<u64> {
        let mut hist = [0u64; 256];
        for &sample in data {
            hist[sample as usize] += 1;
        }

        let mut cumulative = Vec::with_capacity(256);
        let mut sum = 0u64;
        for bin in hist {
            sum += bin;
            cumulative.push(sum);
        }
        cumulative
    }

    #[test]
    fn rotate90_then_extract_area_uses_rotated_coordinates() {
        let image = Image::from_buffer(4, 3, 1, (0u8..12).collect::<Vec<_>>())
            .expect("input image must build");
        let (pipeline, output) = execute_pipeline_to_image::<U8, U8>(&image, |builder| {
            builder.rotate90()?.extract_area(2, 1, 1, 3)
        })
        .expect("rotate90 then extract_area should succeed");

        assert_eq!((pipeline.width, pipeline.height), (1, 3));
        assert_eq!(output.pixels(), &[1, 2, 3]);
    }
}
