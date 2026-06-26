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

    #[test]
    fn recomb_out_of_range_matrix_values_clamp_u8_output() {
        let image =
            Image::<U8>::from_buffer(3, 1, 3, vec![200u8, 200, 200, 255, 1, 128, 0, 255, 42])
                .unwrap()
                .with_metadata(rgb_metadata());
        let matrix = Matrix::new(3, 3, vec![2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0]);

        let (_pipeline, output) = execute_to_image(&image, |builder| {
            builder.append_dyn_op(Box::new(OperationBridge::with_dynamic_bands_pixel_local(
                RecombOp::<U8>::new(matrix),
                3,
                3,
            )))
        })
        .expect("recomb should succeed");

        assert_eq!(output.pixels(), &[255, 0, 200, 255, 0, 128, 0, 0, 42]);
    }

    #[test]
    fn hist_equal_on_already_equalized_image_is_idempotent() {
        let image = equalized_gray_u8();
        let region = viprs::Region::new(0, 0, image.width(), image.height());
        let reducer = HistEqualReducer::new(1, 0, 256).unwrap();
        let tile = viprs::Tile::<U8>::new(region, 1, image.pixels());
        let bins = <HistEqualReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);
        let lut = <HistEqualReducer as TileReducer<U8>>::finalize(&reducer, bins);
        let op = HistEqualOp::<U8>::from_lut(lut).expect("valid LUT");

        let once = image
            .pixels()
            .iter()
            .map(|&value| op.lut()[value as usize])
            .collect::<Vec<_>>();
        let twice = once
            .iter()
            .map(|&value| op.lut()[value as usize])
            .collect::<Vec<_>>();

        assert_eq!(once, image.pixels());
        assert_eq!(twice, once);
    }
}
