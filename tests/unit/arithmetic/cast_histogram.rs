mod chaos_monkey_13 {
    use bytemuck::Pod;
    use viprs::{
        BandFormat, BandFormatId, BuildError, CompiledPipeline, Image, ImageMetadata,
        Interpretation, Tile, TileMut, U8,
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::memory::MemorySource,
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
    use viprs::{
        adapters::codecs::{JpegCodec, PngCodec, WebpCodec},
        ports::codec::ImageEncoder,
    };
    #[cfg(all(feature = "png", feature = "jpeg", feature = "webp"))]

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

    fn run_builder_to_image<FOut, S: viprs_runtime::pipeline::internal::Flush>(
        builder: viprs_runtime::pipeline::internal::PipelineBuilder<S>,
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
        let builder = configure(
            viprs_runtime::pipeline::internal::PipelineBuilder::from_source(
                memory_source_from_image(image),
            ),
        )
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
    fn cast_u8_f32_u8_roundtrip_is_lossless() {
        let image = patterned_u8(17, 9, 3);
        let (pipeline, output) = execute_pipeline_to_image::<U8, U8, _>(&image, |builder| {
            builder.cast(BandFormatId::F32)?.cast(BandFormatId::U8)
        })
        .expect("U8 -> F32 -> U8 roundtrip should succeed");

        assert_eq!(pipeline.output_format, BandFormatId::U8);
        assert_eq!(output.pixels(), image.pixels());
    }

    #[test]
    #[ignore = "BUG: flatten() on RGB silently drops a colour band instead of erroring or acting as a no-op"]
    fn flatten_on_rgb_is_typed_error_or_noop() {
        let image = patterned_u8(6, 4, 3);
        let builder = viprs_runtime::pipeline::internal::PipelineBuilder::from_source(
            memory_source_from_image(&image),
        )
        .flatten([0.0, 0.0, 0.0, 1.0]);

        match builder {
            Err(_) => {}
            Ok(builder) => {
                let (pipeline, output) =
                    run_builder_to_image::<U8, _>(builder, image.metadata().clone())
                        .expect("flatten on RGB should not panic if it succeeds");
                assert_eq!(
                    pipeline.output_bands, 3,
                    "flatten on RGB must not silently drop a colour band"
                );
                assert_eq!(
                    output.pixels(),
                    image.pixels(),
                    "flatten on RGB should behave like a no-op"
                );
            }
        }
    }

    #[test]
    fn histequal_on_single_pixel_preserves_the_only_value() {
        let reducer = HistEqualReducer::new(1, 0, 256).unwrap();
        let region = Region::new(0, 0, 1, 1);
        let input_data = [123u8];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let bins = <HistEqualReducer as TileReducer<U8>>::reduce_tile(&reducer, &input, &region);
        let lut = <HistEqualReducer as TileReducer<U8>>::finalize(&reducer, bins);
        let op = HistEqualOp::<U8>::from_lut(lut).expect("single-pixel LUT must build");
        let mut output_data = [0u8; 1];
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, input_data);
    }

    #[test]
    fn histmatch_on_single_pixel_identity_reference_is_identity() {
        let input_data = [123u8];
        let cum = cumulative_histogram(&input_data);
        let op = HistMatchOp::<U8>::from_cumulative_hists(&cum, &cum)
            .expect("single-pixel cumulative histograms must build");
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output_data = [0u8; 1];
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, input_data);
    }

    #[test]
    fn linear_with_large_positive_offset_clamps_u8() {
        let image =
            Image::from_buffer(3, 1, 1, vec![0u8, 10, 255]).expect("input image must build");
        let (_pipeline, output) =
            execute_pipeline_to_image::<U8, U8, _>(&image, |builder| builder.linear(1.0, 1000.0))
                .expect("linear with large positive offset should not panic");

        assert_eq!(output.pixels(), &[255, 255, 255]);
    }
}
