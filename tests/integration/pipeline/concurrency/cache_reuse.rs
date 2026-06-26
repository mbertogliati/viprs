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
      BuildError, CompiledPipeline, F32, InMemoryImage, ImageMetadata, Interpretation, U8, U16,
      adapters::{
          pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
          sources::memory::MemorySource,
        },
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

    fn make_u8_image(width: u32, height: u32, bands: u32, pixels: Vec<u8>) -> InMemoryImage<U8> {
        let image = InMemoryImage::from_buffer(width, height, bands, pixels).unwrap();
        if bands >= 3 {
            image.with_metadata(srgb_metadata())
        } else {
            image
        }
    }

    fn make_u16_image(width: u32, height: u32, bands: u32, pixels: Vec<u16>) -> InMemoryImage<U16> {
        InMemoryImage::from_buffer(width, height, bands, pixels).unwrap()
    }

    fn make_f32_image(
        width: u32,
        height: u32,
        bands: u32,
        pixels: Vec<f32>,
        metadata: ImageMetadata,
    ) -> InMemoryImage<F32> {
        InMemoryImage::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(metadata)
    }

    fn grayscale_pattern(width: u32, height: u32) -> InMemoryImage<U8> {
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 29 + 11) % 256) as u8);
            }
        }
        make_u8_image(width, height, 1, pixels)
    }

    fn rgb_pattern(width: u32, height: u32) -> InMemoryImage<U8> {
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

    fn rgba_pattern(width: u32, height: u32) -> InMemoryImage<U8> {
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

    fn u16_pattern(width: u32, height: u32, bands: u32) -> InMemoryImage<U16> {
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

    fn memory_source_from_image<F>(image: &InMemoryImage<F>) -> MemorySource<F>
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

    fn execute_to_image<FIn, FOut, S: viprs::pipeline::Commit>(
      image: &InMemoryImage<FIn>,
      configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(CompiledPipeline, InMemoryImage<FOut>), String>
    where
        FIn: viprs::BandFormat,
        FOut: viprs::BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(
            image,
        )))
        .map_err(|error| format!("stage failed: {error:?}"))?
        .build()
        .map_err(|error| format!("build failed: {error:?}"))?;

        let output = pipeline
            .run_to_image::<FOut, _>(
                &RayonScheduler::new(2).map_err(|error| format!("scheduler failed: {error}"))?,
            )
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn apply_hist_equal(image: &InMemoryImage<U8>) -> InMemoryImage<U8> {
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
        InMemoryImage::from_buffer(image.width(), image.height(), image.bands(), output)
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

    #[test]
    fn multiple_cached_nodes_reuse_second_cache_on_repeated_runs() {
        let image = grayscale_pattern(16, 16);
        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));
        let pipeline = ImagePipeline::from_source(memory_source_from_image(&image))
            .then(Box::new(OperationBridge::new_pixel_local(
                CountingPass {
                    calls: Arc::clone(&first_calls),
                },
                1,
            )))
            .unwrap()
            .cache_last_op(NonZeroUsize::new(1 << 20).unwrap())
            .unwrap()
            .then(Box::new(OperationBridge::new_pixel_local(
                CountingPass {
                    calls: Arc::clone(&second_calls),
                },
                1,
            )))
            .unwrap()
            .cache_last_op(NonZeroUsize::new(1 << 20).unwrap())
            .unwrap()
            .build()
            .unwrap();

        let scheduler = RayonScheduler::new(1).unwrap();
        let first = pipeline.run_to_image::<U8, _>(&scheduler).unwrap();
        let second = pipeline.run_to_image::<U8, _>(&scheduler).unwrap();

        assert_eq!(first.pixels(), second.pixels());
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    }
}
