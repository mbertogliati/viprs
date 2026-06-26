mod chaos_monkey_3 {
    use bytemuck::Pod;
    use proptest::prelude::*;
    use viprs::{
        BuildError, CompiledPipeline, Image, ImageMetadata, Interpretation, U8, ViprsError,
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::memory::MemorySource,
        },
        domain::{
            colorspace::{ColorspaceId, Hsv, Lab, SRgb},
            kernel::InterpolationKernel,
            ops::{conversion::ExtendMode, resample::Resize},
        },
        ports::scheduler::TileScheduler,
    };

    const SHARPEN_X1: f32 = 2.0;
    const SHARPEN_Y2: f32 = 10.0;
    const SHARPEN_Y3: f32 = 20.0;
    const SHARPEN_M1: f32 = 0.0;
    const SHARPEN_M2: f32 = 3.0;

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

    fn zero_rgb_u8(width: u32, height: u32) -> Image<U8> {
        Image::from_buffer(
            width,
            height,
            3,
            vec![0u8; width as usize * height as usize * 3],
        )
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
            .map_err(|error| format!("scheduler construction failed: {error}"))?
            .run(&pipeline, &mut sink)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, sink.into_buffer()))
    }

    fn build_pipeline_only<F, S: viprs_runtime::pipeline::internal::CommitPlan>(
        image: &Image<F>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelinePlan,
        )
            -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
    ) -> Result<CompiledPipeline, ViprsError>
    where
        F: viprs::BandFormat,
        F::Sample: Pod,
    {
        configure(
            viprs_runtime::pipeline::internal::PipelinePlan::from_source(memory_source_from_image(
                image,
            )),
        )?
        .compile()
        .map_err(Into::into)
    }

    fn assert_u8_pixels_within_tolerance(expected: &[u8], actual: &[u8], tolerance: u8) {
        assert_eq!(expected.len(), actual.len());
        for (index, (&lhs, &rhs)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = lhs.abs_diff(rhs);
            assert!(
                diff <= tolerance,
                "pixel mismatch at index {index}: expected {lhs}, got {rhs}, tolerance {tolerance}"
            );
        }
    }

    fn flip_horizontal_buffer(buffer: &[u8], width: u32, height: u32, bands: u32) -> Vec<u8> {
        let width = width as usize;
        let height = height as usize;
        let bands = bands as usize;
        let mut flipped = vec![0u8; buffer.len()];

        for y in 0..height {
            for x in 0..width {
                let src = (y * width + (width - 1 - x)) * bands;
                let dst = (y * width + x) * bands;
                flipped[dst..dst + bands].copy_from_slice(&buffer[src..src + bands]);
            }
        }

        flipped
    }

    fn flip_vertical_buffer(buffer: &[u8], width: u32, height: u32, bands: u32) -> Vec<u8> {
        let width = width as usize;
        let height = height as usize;
        let bands = bands as usize;
        let row_len = width * bands;
        let mut flipped = vec![0u8; buffer.len()];

        for y in 0..height {
            let src = (height - 1 - y) * row_len;
            let dst = y * row_len;
            flipped[dst..dst + row_len].copy_from_slice(&buffer[src..src + row_len]);
        }

        flipped
    }

    fn assert_identity_sizes<S: viprs_runtime::pipeline::internal::CommitPlan>(
        configure: impl Copy
        + Fn(
            viprs_runtime::pipeline::internal::PipelinePlan,
            u32,
            u32,
        )
            -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
        tolerance: u8,
    ) {
        for image in [
            patterned_rgb_u8(1, 1),
            patterned_rgb_u8(3, 3),
            patterned_rgb_u8(7, 5),
            zero_rgb_u8(100, 100),
            patterned_rgb_u8(1, 8192),
            patterned_rgb_u8(8192, 1),
        ] {
            let (_pipeline, output) = execute_to_image(&image, |builder| {
                configure(builder, image.width(), image.height())
            })
            .expect("identity pipeline should succeed");
            assert_eq!(output.width(), image.width());
            assert_eq!(output.height(), image.height());
            assert_u8_pixels_within_tolerance(image.pixels(), output.pixels(), tolerance);
        }
    }

    #[test]
    fn pass3_single_pixel_images_survive_identityish_ops() {
        let image = patterned_rgb_u8(1, 1);
        let (_pipeline, output) = execute_to_image(&image, |builder| {
            builder
                .with_colorspace(ColorspaceId::SRgb)
                .plan_colourspace::<Lab>()?
                .plan_colourspace::<SRgb>()?
                .plan_gauss_blur(0.0)?
                .plan_linear(1.0, 0.0)?
                .plan_resize(Resize::new(1.0, 1.0, InterpolationKernel::Lanczos3))?
                .plan_affine(
                    [1.0, 0.0, 0.0, 1.0],
                    0.0,
                    0.0,
                    1,
                    1,
                    InterpolationKernel::Nearest,
                )
        })
        .expect("1x1 chain should succeed");

        assert_eq!((output.width(), output.height(), output.bands()), (1, 1, 3));
        assert_u8_pixels_within_tolerance(image.pixels(), output.pixels(), 2);
    }
}
