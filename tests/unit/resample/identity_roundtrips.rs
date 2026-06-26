mod chaos_monkey {
    use bytemuck::Pod;
    use proptest::prelude::*;
    use viprs::{
        BuildError, F32, Image, ImageMetadata, Interpretation, U8, U16,
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::memory::MemorySource,
        },
        domain::{
            colorspace::{ColorspaceId, Lab, SRgb},
            kernel::InterpolationKernel,
            ops::resample::{Resize, Thumbnail, thumbnail::ThumbnailTarget},
        },
        ports::scheduler::TileScheduler,
    };

    fn rgb_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        }
    }

    fn make_u8_image(width: u32, height: u32, bands: u32, pixels: Vec<u8>) -> Image<U8> {
        Image::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(if bands >= 3 {
                rgb_metadata()
            } else {
                ImageMetadata::default()
            })
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
    ) -> Result<(viprs_runtime::pipeline::CompiledPipeline, Image<F>), String>
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
            .map_err(|error| format!("scheduler construction failed: {error}"))?
            .run(&pipeline, &mut sink)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        let output = sink
            .into_image::<F>(
                pipeline.width,
                pipeline.height,
                pipeline.output_bands,
                image.metadata().clone(),
            )
            .map_err(|error| format!("failed to materialize output: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn execute_to_buffer<F, S: viprs_runtime::pipeline::internal::CommitPlan>(
        image: &Image<F>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelinePlan,
        )
            -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
    ) -> Result<(viprs_runtime::pipeline::CompiledPipeline, Vec<u8>), String>
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
            .map_err(|error| format!("scheduler construction failed: {error}"))?
            .run(&pipeline, &mut sink)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, sink.into_buffer()))
    }

    fn thumbnail(width: u32) -> Thumbnail {
        Thumbnail::new(ThumbnailTarget::Width(width), InterpolationKernel::Lanczos3)
    }

    fn patterned_rgb_u8(width: u32, height: u32) -> Image<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 13) % 256) as u8);
                pixels.push(((x * 11 + y * 29 + 7) % 256) as u8);
                pixels.push(((x * 5 + y * 19 + 191) % 256) as u8);
            }
        }
        make_u8_image(width, height, 3, pixels)
    }

    #[test]
    fn pass1_rotate90_four_times_is_identity_for_rectangular_rgb_u8() {
        let image = patterned_rgb_u8(7, 5);
        let (_pipeline, output) = execute_to_image(&image, |builder| {
            builder
                .plan_rotate90()?
                .plan_rotate90()?
                .plan_rotate90()?
                .plan_rotate90()
        })
        .expect("rotate90 x4 should behave as identity");

        assert_eq!(output.width(), image.width());
        assert_eq!(output.height(), image.height());
        assert_eq!(output.pixels().len(), image.pixels().len());
    }

    #[test]
    fn pass2_double_thumbnail_matches_sequential_execution() {
        let image = patterned_rgb_u8(777, 333);

        let (_first_pipeline, first) =
            execute_to_image(&image, |builder| builder.plan_thumbnail(thumbnail(100)))
                .expect("first thumbnail should succeed");
        let (_second_pipeline, sequential) =
            execute_to_image(&first, |builder| builder.plan_thumbnail(thumbnail(50)))
                .expect("second thumbnail should succeed");

        let (chained_pipeline, chained) = execute_to_image(&image, |builder| {
            builder
                .plan_thumbnail(thumbnail(100))?
                .plan_thumbnail(thumbnail(50))
        })
        .expect("chained thumbnails should succeed");

        assert_eq!(
            (chained_pipeline.width, chained_pipeline.height),
            (sequential.width(), sequential.height()),
            "thumbnail chaining should use the intermediate output dimensions"
        );
        assert_eq!((chained.width(), chained.height()), (50, 22));
    }

    #[test]
    fn pass2_resize_half_then_double_matches_sequential_execution() {
        let image = patterned_rgb_u8(777, 333);
        let half = Resize::new(0.5, 0.5, InterpolationKernel::Lanczos3);
        let double = Resize::new(2.0, 2.0, InterpolationKernel::Lanczos3);

        let (_first_pipeline, first) =
            execute_to_image(&image, |builder| builder.plan_resize(half))
                .expect("first resize should succeed");
        let (_second_pipeline, sequential) =
            execute_to_image(&first, |builder| builder.plan_resize(double))
                .expect("second resize should succeed");

        let (chained_pipeline, chained) = execute_to_image(&image, |builder| {
            builder.plan_resize(half)?.plan_resize(double)
        })
        .expect("chained resizes should succeed");

        assert_eq!(
            (chained_pipeline.width, chained_pipeline.height),
            (sequential.width(), sequential.height()),
            "resize chaining should use the intermediate output dimensions"
        );
        assert_eq!(chained.pixels(), sequential.pixels());
    }

    #[test]
    fn pass5_thumbnail_colourspace_thumbnail_matches_sequential_execution() {
        let image = patterned_rgb_u8(777, 333);

        let (_first_pipeline, first) = execute_to_image(&image, |builder| {
            builder
                .plan_thumbnail(thumbnail(400))?
                .plan_colourspace::<Lab>()?
                .plan_colourspace::<SRgb>()
        })
        .expect("first thumbnail+colourspace+roundtrip should succeed");
        let (_second_pipeline, sequential) =
            execute_to_image(&first, |builder| builder.plan_thumbnail(thumbnail(200)))
                .expect("second thumbnail should succeed");

        let (chained_pipeline, chained) = execute_to_image(&image, |builder| {
            builder
                .plan_thumbnail(thumbnail(400))?
                .plan_colourspace::<Lab>()?
                .plan_colourspace::<SRgb>()?
                .plan_thumbnail(thumbnail(200))
        })
        .expect("chained thumbnail+colourspace+roundtrip+thumbnail should succeed");

        assert_eq!(
            (chained_pipeline.width, chained_pipeline.height),
            (sequential.width(), sequential.height()),
            "thumbnail after colourspace should use the intermediate output dimensions"
        );
        assert_eq!((chained.width(), chained.height()), (200, 86));
    }
}
