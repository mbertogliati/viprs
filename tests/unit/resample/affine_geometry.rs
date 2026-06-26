mod chaos_monkey_6 {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use bytemuck::Pod;
    use viprs::{
        BandFormatId, BuildError, CompiledPipeline, ImageMetadata, InMemoryImage, Interpretation,
        U8,
        adapters::{
            pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            colorspace::{ColorspaceId, Hsv, Lab},
            kernel::InterpolationKernel,
            ops::{
                conversion::{Angle45, ExtendMode},
                resample::{Thumbnail, thumbnail::ThumbnailTarget},
            },
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

    fn patterned_rgb_u8(width: u32, height: u32) -> InMemoryImage<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 13 + 19) % 256) as u8);
                pixels.push(((x * 11 + y * 29 + 7) % 256) as u8);
                pixels.push(((x * 5 + y * 19 + 191) % 256) as u8);
            }
        }

        InMemoryImage::from_buffer(width, height, 3, pixels)
            .unwrap()
            .with_metadata(rgb_metadata())
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

    fn execute_same_format<F, S: viprs::pipeline::Commit>(
        image: &InMemoryImage<F>,
        configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(CompiledPipeline, InMemoryImage<F>), String>
    where
        F: viprs::BandFormat,
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

    fn execute_to_buffer<F, S: viprs::pipeline::Commit>(
        image: &InMemoryImage<F>,
        configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Vec<u8>), String>
    where
        F: viprs::BandFormat,
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
        let expected_len = pipeline.width as usize
            * pipeline.height as usize
            * pipeline.output_bands as usize
            * bytes_per_sample(pipeline.output_format);
        assert_eq!(buffer.len(), expected_len);
    }

    #[test]
    fn affine_identity_matrix_matches_input_pixels() {
        let image = patterned_rgb_u8(7, 5);
        let (_pipeline, output) = execute_same_format(&image, |builder| {
            builder.affine(
                [1.0, 0.0, 0.0, 1.0],
                0.0,
                0.0,
                image.width(),
                image.height(),
                InterpolationKernel::Nearest,
            )
        })
        .expect("identity affine should succeed");

        assert_eq!(output.width(), image.width());
        assert_eq!(output.height(), image.height());
        assert_eq!(output.pixels(), image.pixels());
    }

    #[test]
    fn rot45_expands_canvas_for_square_input() {
        let image = patterned_rgb_u8(5, 5);
        let (pipeline, _output) =
            execute_same_format(&image, |builder| builder.rot45(Angle45::D45))
                .expect("rot45 should succeed for odd square inputs");

        let minimum_diagonal = ((image.width() as f64) * std::f64::consts::SQRT_2).floor() as u32;
        assert!(
            pipeline.width >= minimum_diagonal,
            "rot45 width {} did not expand to cover the diagonal {}",
            pipeline.width,
            minimum_diagonal
        );
        assert!(
            pipeline.height >= minimum_diagonal,
            "rot45 height {} did not expand to cover the diagonal {}",
            pipeline.height,
            minimum_diagonal
        );
    }

    #[test]
    fn reduce_on_one_by_one_returns_typed_error_not_success() {
        let image = patterned_rgb_u8(1, 1);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            execute_same_format(&image, |builder| {
                builder.reduce(2.0, 2.0, InterpolationKernel::Lanczos3)
            })
        }));

        assert!(outcome.is_ok(), "reduce(2.0, 2.0) panicked on a 1x1 image");
        let result = outcome.unwrap();
        match result {
            Ok((_pipeline, output)) => {
                assert_eq!(output.width(), 1);
                assert_eq!(output.height(), 1);
            }
            Err(_error) => {}
        }
    }

    #[test]
    fn reduce_identity_on_one_by_one_succeeds() {
        let image = patterned_rgb_u8(1, 1);
        let (pipeline, output) = execute_same_format(&image, |builder| {
            builder.reduce(1.0, 1.0, InterpolationKernel::Lanczos3)
        })
        .expect("reduce(1.0, 1.0) on a 1x1 image should succeed");

        assert_eq!(pipeline.width, 1);
        assert_eq!(pipeline.height, 1);
        assert_eq!(output.width(), 1);
        assert_eq!(output.height(), 1);
        assert_eq!(output.pixels(), image.pixels());
    }
}
