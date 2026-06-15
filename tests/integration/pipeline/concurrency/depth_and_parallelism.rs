mod chaos_monkey_7 {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        thread,
    };

    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, Image, ImageMetadata, Interpretation, U8,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            colorspace::{ColorspaceId, Lab, SRgb, Ucs},
            kernel::InterpolationKernel,
            ops::{
                conversion::ExtendMode,
                resample::{Resize, Thumbnail, thumbnail::ThumbnailTarget},
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

    fn zero_band_u8(width: u32, height: u32) -> Image<U8> {
        Image::from_buffer(width, height, 0, Vec::new()).unwrap()
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

    fn execute_same_format<F>(
        image: &Image<F>,
        configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder, BuildError>,
    ) -> Result<(CompiledPipeline, Image<F>), String>
    where
        F: viprs::BandFormat,
        F::Sample: Pod,
    {
        let pipeline = configure(PipelineBuilder::from_source(memory_source_from_image(
            image,
        )))
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

    fn thumbnail(width: u32) -> Thumbnail {
        Thumbnail::new(ThumbnailTarget::Width(width), InterpolationKernel::Lanczos3)
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

    #[test]
    fn pipeline_depth_linear_chain_stays_stable() {
        let image = patterned_rgb_u8(31, 17);
        let (pipeline, output) = execute_same_format(&image, |builder| {
            let mut builder = builder;
            for _ in 0..50 {
                builder = builder.linear(1.0, 0.0)?;
            }
            Ok(builder)
        })
        .expect("deep linear chain should succeed");

        assert_eq!(output.pixels(), image.pixels());
        assert!(
            pipeline.nodes.len() <= 2,
            "expected linear fusion to keep the pipeline shallow, got {} nodes",
            pipeline.nodes.len()
        );
    }

    #[test]
    fn concurrent_thumbnail_execution_is_deterministic() {
        let mut handles = Vec::new();

        for _ in 0..4 {
            handles.push(thread::spawn(|| {
                let image = patterned_rgb_u8(1024, 1024);
                let (pipeline, output) =
                    execute_same_format(&image, |builder| builder.thumbnail(thumbnail(400)))
                        .expect("thumbnail should succeed concurrently");
                (pipeline.width, pipeline.height, output.pixels().to_vec())
            }));
        }

        let first = handles.remove(0).join().expect("first thread panicked");
        assert_eq!((first.0, first.1), (400, 400));

        for handle in handles {
            let current = handle.join().expect("worker thread panicked");
            assert_eq!((current.0, current.1), (first.0, first.1));
            assert_eq!(current.2, first.2, "concurrent thumbnail outputs diverged");
        }
    }
}
