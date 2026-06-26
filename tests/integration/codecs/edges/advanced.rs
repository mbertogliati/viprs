#![cfg(any(feature = "jpeg", feature = "png", feature = "webp", feature = "gif"))]

mod chaos_monkey_12 {
    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, F32, Image, ImageMetadata, Interpretation, OperationBridge,
        RecombOp, U8,
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::memory::MemorySource,
        },
        domain::{
            colorspace::Lab,
            kernel::InterpolationKernel,
            ops::{
                arithmetic::Matrix,
                conversion::ExtendMode,
                histogram::HistEqualOp,
                resample::{Thumbnail, thumbnail::ThumbnailTarget},
            },
            reducer::TileReducer,
            reducers::HistEqualReducer,
        },
        ports::scheduler::TileScheduler,
    };

    #[cfg(feature = "jpeg")]
    use std::sync::Arc;
    #[cfg(feature = "jpeg")]
    use viprs::adapters::codecs::JpegCodec;
    #[cfg(feature = "jpeg")]
    use viprs::ports::codec::{ImageDecoder, ImageEncoder};

    fn rgb_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        }
    }

    fn gray_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::BW),
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

    fn equalized_gray_u8() -> Image<U8> {
        let pixels = (0u8..=255).collect::<Vec<_>>();
        Image::from_buffer(16, 16, 1, pixels)
            .unwrap()
            .with_metadata(gray_metadata())
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

    fn execute_to_image<F, S: viprs_runtime::pipeline::internal::CommitPlan>(
        image: &Image<F>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelinePlan,
        )
            -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Image<F>), String>
    where
        F: viprs::BandFormat,
        F::Sample: Pod,
    {
        let pipeline = configure(
            viprs_runtime::pipeline::internal::PipelinePlan::from_source(memory_source_from_image(
                image,
            )),
        )
        .map_err(|error| format!("stage failed: {error:?}"))?
        .compile()
        .map_err(|error| format!("build failed: {error:?}"))?;

        let output = pipeline
            .run_to_image::<F, _>(&RayonScheduler::new(2).map_err(|error| error.to_string())?)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn execute_to_image_with_output<
        InputF,
        OutputF,
        S: viprs_runtime::pipeline::internal::CommitPlan,
    >(
        image: &Image<InputF>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelinePlan,
        )
            -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Image<OutputF>), String>
    where
        InputF: viprs::BandFormat,
        InputF::Sample: Pod,
        OutputF: viprs::BandFormat,
    {
        let pipeline = configure(
            viprs_runtime::pipeline::internal::PipelinePlan::from_source(memory_source_from_image(
                image,
            )),
        )
        .map_err(|error| format!("stage failed: {error:?}"))?
        .compile()
        .map_err(|error| format!("build failed: {error:?}"))?;

        let output = pipeline
            .run_to_image::<OutputF, _>(&RayonScheduler::new(2).map_err(|error| error.to_string())?)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn execute_to_buffer<F, S: viprs_runtime::pipeline::internal::CommitPlan>(
        image: &Image<F>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelinePlan,
        )
            -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Vec<u8>), String>
    where
        F: viprs::BandFormat,
        F::Sample: Pod,
    {
        let pipeline = configure(
            viprs_runtime::pipeline::internal::PipelinePlan::from_source(memory_source_from_image(
                image,
            )),
        )
        .map_err(|error| format!("stage failed: {error:?}"))?
        .compile()
        .map_err(|error| format!("build failed: {error:?}"))?;

        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(2)
            .map_err(|error| error.to_string())?
            .run(&pipeline, &mut sink)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, sink.into_buffer()))
    }

    fn thumbnail(width: u32) -> Thumbnail {
        Thumbnail::new(ThumbnailTarget::Width(width), InterpolationKernel::Lanczos3)
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn jpeg_codec_handles_simultaneous_encode_and_decode() {
        let codec = Arc::new(JpegCodec);
        let image = Arc::new(patterned_rgb_u8(64, 64));
        let encoded = Arc::new(codec.encode(&image).expect("jpeg encode should succeed"));

        let mut threads = Vec::new();
        for _ in 0..4 {
            let codec = Arc::clone(&codec);
            let image = Arc::clone(&image);
            let encoded = Arc::clone(&encoded);
            threads.push(std::thread::spawn(move || {
                let roundtrip = codec
                    .decode::<U8>(&encoded)
                    .expect("jpeg decode should succeed");
                let encoded_again = codec
                    .encode(&image)
                    .expect("jpeg encode should stay stateless");
                assert_eq!((roundtrip.width(), roundtrip.height()), (64, 64));
                assert!(!encoded_again.is_empty());
            }));
        }

        for thread in threads {
            thread.join().expect("jpeg worker should not panic");
        }
    }
}

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

    fn run_builder_to_image<FOut, S: viprs_runtime::pipeline::internal::CommitPlan>(
        builder: viprs_runtime::pipeline::internal::PipelinePlan<S>,
        metadata: ImageMetadata,
    ) -> Result<(CompiledPipeline, Image<FOut>), String>
    where
        FOut: BandFormat,
        FOut::Sample: Pod,
    {
        let pipeline = builder
            .compile()
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

    fn execute_pipeline_to_image<FIn, FOut, S: viprs_runtime::pipeline::internal::CommitPlan>(
        image: &Image<FIn>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelinePlan,
        )
            -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Image<FOut>), String>
    where
        FIn: BandFormat,
        FOut: BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let builder = configure(
            viprs_runtime::pipeline::internal::PipelinePlan::from_source(memory_source_from_image(
                image,
            )),
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

    #[cfg(all(feature = "jpeg", feature = "png", feature = "webp"))]
    #[test]
    fn concurrent_png_jpeg_webp_encode_share_image_safely() {
        let image = Arc::new(patterned_u8(64, 48, 3));
        let png_image = Arc::clone(&image);
        let jpeg_image = Arc::clone(&image);
        let webp_image = Arc::clone(&image);

        let png_thread = thread::spawn(move || PngCodec::default().encode(&png_image));
        let jpeg_thread = thread::spawn(move || JpegCodec.encode(&jpeg_image));
        let webp_thread = thread::spawn(move || WebpCodec.encode(&webp_image));

        let png = png_thread.join().unwrap().unwrap();
        let jpeg = jpeg_thread.join().unwrap().unwrap();
        let webp = webp_thread.join().unwrap().unwrap();

        assert!(!png.is_empty());
        assert!(!jpeg.is_empty());
        assert!(!webp.is_empty());
    }
}

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

    fn run_pipeline<F, S: viprs_runtime::pipeline::internal::CommitPlan>(
        image: &Image<F>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelinePlan,
        )
            -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Image<F>), String>
    where
        F: viprs::BandFormat,
        F::Sample: Pod,
    {
        let pipeline = configure(
            viprs_runtime::pipeline::internal::PipelinePlan::from_source(memory_source_from_image(
                image,
            )),
        )
        .map_err(|error| format!("stage failed: {error:?}"))?
        .compile()
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
    fn palette_png_decode_then_invert_does_not_panic() {
        let encoded = indexed_png_bytes();
        let decoded = PngCodec::default().decode::<U8>(&encoded).unwrap();

        let (_pipeline, output) = run_pipeline(&decoded, |builder| builder.plan_invert()).unwrap();

        assert_eq!((output.width(), output.height(), output.bands()), (2, 1, 3));
    }

    #[test]
    fn invert_pipeline_preserves_icc_profile_metadata() {
        let mut metadata = rgb_metadata();
        metadata.icc_profile = Some((0u8..32).collect());
        let image = patterned_rgb_u8(4, 3).with_metadata(metadata);

        let (_pipeline, output) = run_pipeline(&image, |builder| builder.plan_invert()).unwrap();

        assert_eq!(output.metadata().icc_profile, image.metadata().icc_profile);
    }

    #[cfg(all(feature = "jpeg", feature = "png"))]
    #[test]
    fn concurrent_png_and_jpeg_encode_share_image_safely() {
        let image = Arc::new(patterned_rgb_u8(64, 48));
        let png_image = Arc::clone(&image);
        let jpeg_image = Arc::clone(&image);

        let png_thread = std::thread::spawn(move || PngCodec::default().encode(&png_image));
        let jpeg_thread = std::thread::spawn(move || JpegCodec.encode(&jpeg_image));

        let png = png_thread.join().unwrap().unwrap();
        let jpeg = jpeg_thread.join().unwrap().unwrap();

        assert!(!png.is_empty());
        assert!(!jpeg.is_empty());
    }
}
