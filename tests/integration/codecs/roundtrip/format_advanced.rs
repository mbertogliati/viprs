#![cfg(any(feature = "jpeg", feature = "png", feature = "webp", feature = "gif"))]

mod chaos_monkey_18 {
    use std::{
        num::NonZeroUsize,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, F32, Image, ImageMetadata, Interpretation, U8, U16,
        adapters::{scheduler::rayon_scheduler::RayonScheduler, sources::memory::MemorySource},
        domain::{
            kernel::InterpolationKernel,
            op::{Op, OperationBridge, PixelLocalOp},
            ops::{
                arithmetic::{Matrix, RecombOp},
                conversion::embed::ExtendMode,
                histogram::HistEqualOp,
                resample::{Thumbnail, thumbnail::ThumbnailTarget},
            },
            reducer::TileReducer,
            reducers::HistEqualReducer,
        },
        ports::codec::{ImageDecoder, ImageEncoder},
    };

    #[cfg(feature = "png")]
    use viprs::adapters::codecs::PngCodec;

    fn srgb_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        }
    }

    fn lab_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Lab),
            ..ImageMetadata::default()
        }
    }

    fn make_u8_image(width: u32, height: u32, bands: u32, pixels: Vec<u8>) -> Image<U8> {
        let image = Image::from_buffer(width, height, bands, pixels).unwrap();
        if bands >= 3 {
            image.with_metadata(srgb_metadata())
        } else {
            image
        }
    }

    fn make_u16_image(width: u32, height: u32, bands: u32, pixels: Vec<u16>) -> Image<U16> {
        Image::from_buffer(width, height, bands, pixels).unwrap()
    }

    fn make_f32_image(
        width: u32,
        height: u32,
        bands: u32,
        pixels: Vec<f32>,
        metadata: ImageMetadata,
    ) -> Image<F32> {
        Image::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(metadata)
    }

    fn grayscale_pattern(width: u32, height: u32) -> Image<U8> {
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 29 + 11) % 256) as u8);
            }
        }
        make_u8_image(width, height, 1, pixels)
    }

    fn rgb_pattern(width: u32, height: u32) -> Image<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 13 + 3) % 256) as u8);
                pixels.push(((x * 11 + y * 29 + 7) % 256) as u8);
                pixels.push(((x * 5 + y * 19 + 191) % 256) as u8);
            }
        }
        make_u8_image(width, height, 3, pixels)
    }

    fn rgba_pattern(width: u32, height: u32) -> Image<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 13 + 3) % 256) as u8);
                pixels.push(((x * 11 + y * 29 + 7) % 256) as u8);
                pixels.push(((x * 5 + y * 19 + 191) % 256) as u8);
                pixels.push(((x * 23 + y * 31 + 255) % 256) as u8);
            }
        }
        make_u8_image(width, height, 4, pixels)
    }

    fn u16_pattern(width: u32, height: u32, bands: u32) -> Image<U16> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * bands as usize);
        for y in 0..height {
            for x in 0..width {
                for band in 0..bands {
                    pixels
                        .push(((x * 257 + y * 911 + band * 12_345 + 17) % u16::MAX as u32) as u16);
                }
            }
        }
        make_u16_image(width, height, bands, pixels)
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

    fn execute_to_image<FIn, FOut, S: viprs_runtime::pipeline::internal::CommitPlan>(
        image: &Image<FIn>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelinePlan,
        )
            -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Image<FOut>), String>
    where
        FIn: viprs::BandFormat,
        FOut: viprs::BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
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
            .run_to_image::<FOut, _>(
                &RayonScheduler::new(2).map_err(|error| format!("scheduler failed: {error}"))?,
            )
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn apply_hist_equal(image: &Image<U8>) -> Image<U8> {
        let reducer = HistEqualReducer::new(1, 0, 256).unwrap();
        let region = viprs::Region::new(0, 0, image.width(), image.height());
        let tile = viprs::Tile::<U8>::new(region, image.bands(), image.pixels());
        let bins = <HistEqualReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);
        let lut = <HistEqualReducer as TileReducer<U8>>::finalize(&reducer, bins);
        let op = HistEqualOp::<U8>::from_lut(lut).unwrap();
        let mut output = vec![0u8; image.pixels().len()];
        let mut output_tile = viprs::TileMut::<U8>::new(region, image.bands(), &mut output);
        let mut state = ();
        op.process_region(&mut state, &tile, &mut output_tile);
        Image::from_buffer(image.width(), image.height(), image.bands(), output)
            .unwrap()
            .with_metadata(image.metadata().clone())
    }

    #[derive(Clone)]
    struct CountingPass {
        calls: Arc<AtomicUsize>,
    }

    impl Op for CountingPass {
        type Input = U8;
        type Output = U8;
        type State = ();

        fn demand_hint(&self) -> viprs::DemandHint {
            viprs::DemandHint::ThinStrip
        }

        fn required_input_region(&self, output: &viprs::Region) -> viprs::Region {
            *output
        }

        fn start(&self) {}

        fn process_region(
            &self,
            _: &mut Self::State,
            input: &viprs::Tile<U8>,
            output: &mut viprs::TileMut<U8>,
        ) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            output.data.copy_from_slice(input.data);
        }
    }

    impl PixelLocalOp for CountingPass {}

    #[cfg(feature = "png")]
    #[test]
    fn png_u16_roundtrip_is_pixel_exact() {
        let image = u16_pattern(17, 11, 3);
        let codec = PngCodec::default();
        let encoded = codec.encode(&image).expect("png encode should succeed");
        let decoded: Image<U16> = codec.decode(&encoded).expect("png decode should succeed");

        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (image.width(), image.height(), image.bands())
        );
        assert_eq!(decoded.pixels(), image.pixels());
    }
}

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

    fn execute_pipeline_to_buffer<F, S: viprs_runtime::pipeline::internal::CommitPlan>(
        image: &Image<F>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelinePlan,
        )
            -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Vec<u8>), String>
    where
        F: BandFormat,
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
    fn png_rotate_roundtrip_rgb_preserves_pixels() {
        let image = patterned_u8(7, 5, 3);
        let (_pipeline, rotated) =
            execute_pipeline_to_image::<U8, U8, _>(&image, |builder| builder.plan_rotate90())
                .expect("rotate90 should succeed");

        let codec = PngCodec::default();
        let encoded = codec.encode(&rotated).expect("png encode should succeed");
        let decoded = codec
            .decode::<U8>(&encoded)
            .expect("png decode should succeed");

        assert_eq!(decoded.width(), rotated.width());
        assert_eq!(decoded.height(), rotated.height());
        assert_eq!(decoded.bands(), rotated.bands());
        assert_eq!(decoded.pixels(), rotated.pixels());
    }

    #[cfg(feature = "png")]
    #[test]
    fn png_rotate_roundtrip_rgba_preserves_pixels() {
        let image = patterned_u8(5, 7, 4);
        let (_pipeline, rotated) =
            execute_pipeline_to_image::<U8, U8, _>(&image, |builder| builder.plan_rotate90())
                .expect("rotate90 should succeed");

        let codec = PngCodec::default();
        let encoded = codec.encode(&rotated).expect("png encode should succeed");
        let decoded = codec
            .decode::<U8>(&encoded)
            .expect("png decode should succeed");

        assert_eq!(decoded.width(), rotated.width());
        assert_eq!(decoded.height(), rotated.height());
        assert_eq!(decoded.bands(), rotated.bands());
        assert_eq!(decoded.pixels(), rotated.pixels());
    }
}
