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

    fn execute_to_image<FIn, FOut, S: viprs_runtime::pipeline::Flush>(
        image: &Image<FIn>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::PipelineBuilder,
        )
            -> Result<viprs_runtime::pipeline::PipelineBuilder<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Image<FOut>), String>
    where
        FIn: viprs::BandFormat,
        FOut: viprs::BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let pipeline = configure(viprs_runtime::pipeline::PipelineBuilder::from_source(
            memory_source_from_image(image),
        ))
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

    #[test]
    fn linear_scales_all_lab_channels() {
        let image = make_f32_image(
            1,
            2,
            3,
            vec![10.0, -5.0, 8.0, 30.0, 4.5, -2.5],
            lab_metadata(),
        );
        let (_pipeline, output) =
            execute_to_image::<F32, F32, _>(&image, |builder| builder.linear(2.0, 0.0))
                .expect("linear on Lab image should succeed");

        assert!(matches!(
            output.metadata().interpretation,
            Some(Interpretation::Lab) | Some(Interpretation::Srgb)
        ));
        assert_eq!(output.pixels(), &[20.0, -10.0, 16.0, 60.0, 9.0, -5.0]);
    }
}
