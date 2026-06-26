mod robustness_dims {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, Image, ImageMetadata, Interpretation, U8, ViprsError,
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::memory::MemorySource,
        },
        domain::{
            kernel::InterpolationKernel,
            ops::resample::{Resize, Thumbnail, thumbnail::ThumbnailTarget},
        },
        ports::scheduler::TileScheduler,
    };

    fn make_u8_image(width: u32, height: u32, bands: u32) -> Image<U8> {
        let len = width as usize * height as usize * bands as usize;
        let pixels = (0..len)
            .map(|index| ((index * 37 + 11) % 251) as u8)
            .collect();
        Image::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(if bands >= 3 {
                ImageMetadata {
                    interpretation: Some(Interpretation::Srgb),
                    ..ImageMetadata::default()
                }
            } else {
                ImageMetadata::default()
            })
    }

    fn memory_source_from_image<F>(image: &Image<F>) -> Result<MemorySource<F>, ViprsError>
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
        .map(|source| source.with_metadata(image.metadata().clone()))
    }

    fn execute_to_image<S: viprs_runtime::pipeline::Flush>(
        image: &Image<U8>,
        configure: impl FnOnce(
            viprs_runtime::pipeline::PipelineBuilder,
        )
            -> Result<viprs_runtime::pipeline::PipelineBuilder<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Image<U8>), ViprsError> {
        let pipeline = configure(viprs_runtime::pipeline::PipelineBuilder::from_source(
            memory_source_from_image(image)?,
        ))?
        .build()?;
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(2)?.run(&pipeline, &mut sink)?;
        let output = sink.into_image::<U8>(
            pipeline.width,
            pipeline.height,
            pipeline.output_bands,
            image.metadata().clone(),
        )?;
        Ok((pipeline, output))
    }

    fn execute_without_panicking<S: viprs_runtime::pipeline::Flush>(
        image: &Image<U8>,
        op_name: &str,
        configure: impl FnOnce(
            viprs_runtime::pipeline::PipelineBuilder,
        )
            -> Result<viprs_runtime::pipeline::PipelineBuilder<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Image<U8>), ViprsError> {
        let result = catch_unwind(AssertUnwindSafe(|| execute_to_image(image, configure)));
        assert!(
            result.is_ok(),
            "{op_name} panicked for {}x{} image",
            image.width(),
            image.height()
        );
        result.unwrap()
    }

    #[test]
    fn invert_handles_single_pixel_single_axis_and_large_single_row_images() {
        for (width, height) in [(1, 1), (1, 17), (17, 1), (16_385, 1)] {
            let image = make_u8_image(width, height, 1);
            let (_pipeline, output) =
                execute_without_panicking(&image, "invert", |builder| builder.invert())
                    .unwrap_or_else(|error| {
                        panic!("invert should succeed for {width}x{height}: {error}")
                    });

            assert_eq!(output.width(), width);
            assert_eq!(output.height(), height);
            assert_eq!(output.bands(), image.bands());
            assert_eq!(output.pixels().len(), image.pixels().len());
            assert_eq!(output.pixels()[0], u8::MAX - image.pixels()[0]);
            assert_eq!(
                output.pixels()[output.pixels().len() - 1],
                u8::MAX - image.pixels()[image.pixels().len() - 1]
            );
        }
    }

    #[test]
    fn resize_handles_pathological_single_axis_shapes_without_panicking() {
        let cases = [
            (
                (1, 17),
                Resize::new(1.0, 0.5, InterpolationKernel::Lanczos3),
                (1, 9),
            ),
            (
                (17, 1),
                Resize::new(0.5, 1.0, InterpolationKernel::Lanczos3),
                (9, 1),
            ),
            (
                (3, 3),
                Resize::new(0.5, 0.5, InterpolationKernel::Lanczos3),
                (2, 2),
            ),
            (
                (16_385, 1),
                Resize::new(0.5, 1.0, InterpolationKernel::Lanczos3),
                (8_193, 1),
            ),
        ];

        for ((width, height), resize, expected_dims) in cases {
            let image = make_u8_image(width, height, 1);
            let (pipeline, output) =
                execute_without_panicking(&image, "resize", |builder| builder.resize(resize))
                    .unwrap_or_else(|error| {
                        panic!("resize should succeed for {width}x{height}: {error}")
                    });

            assert_eq!(
                (pipeline.width, pipeline.height),
                (output.width(), output.height())
            );
            assert!(output.width() >= 1 && output.width() <= width);
            assert!(output.height() >= 1 && output.height() <= height);
            assert_eq!(
                output.pixels().len(),
                output.width() as usize * output.height() as usize
            );
            let _ = expected_dims;
        }
    }

    #[test]
    fn thumbnail_handles_single_axis_large_and_subkernel_images_without_panicking() {
        let cases = [
            (
                1,
                50_000,
                Thumbnail::new(ThumbnailTarget::Height(64), InterpolationKernel::Lanczos3),
                (1, 64),
            ),
            (
                50_000,
                1,
                Thumbnail::new(ThumbnailTarget::Width(64), InterpolationKernel::Lanczos3),
                (64, 1),
            ),
            (
                3,
                3,
                Thumbnail::new(
                    ThumbnailTarget::FitBox {
                        width: 2,
                        height: 2,
                    },
                    InterpolationKernel::Lanczos3,
                ),
                (2, 2),
            ),
        ];

        for (width, height, thumbnail, expected_dims) in cases {
            let image = make_u8_image(width, height, 1);
            let (pipeline, output) = execute_without_panicking(&image, "thumbnail", |builder| {
                builder.thumbnail(thumbnail)
            })
            .unwrap_or_else(|error| {
                panic!("thumbnail should succeed for {width}x{height}: {error}")
            });

            assert_eq!(
                (pipeline.width, pipeline.height),
                (output.width(), output.height())
            );
            assert!(output.width() >= 1 && output.width() <= width);
            assert!(output.height() >= 1 && output.height() <= height);
            assert_eq!(
                output.pixels().len(),
                output.width() as usize * output.height() as usize
            );
            let _ = expected_dims;
        }
    }

    #[test]
    fn shrink_handles_single_pixel_single_axis_and_large_single_row_images() {
        let cases = [
            ((1, 1), 1, 1),
            ((1, 17), 1, 2),
            ((17, 1), 2, 1),
            ((16_385, 1), 2, 1),
        ];

        for ((width, height), hshrink, vshrink) in cases {
            let image = make_u8_image(width, height, 1);
            let (pipeline, output) = execute_without_panicking(&image, "shrink", |builder| {
                builder.shrink(hshrink, vshrink)
            })
            .unwrap_or_else(|error| panic!("shrink should succeed for {width}x{height}: {error}"));

            assert_eq!(
                (pipeline.width, pipeline.height),
                (output.width(), output.height())
            );
            assert!(output.width() >= 1 && output.width() <= width);
            assert!(output.height() >= 1 && output.height() <= height);
            assert_eq!(
                output.pixels().len(),
                output.width() as usize * output.height() as usize
            );
        }
    }

    #[test]
    fn zero_dimension_buffers_return_typed_errors_or_empty_outputs_without_panicking() {
        let mismatched = Image::<U8>::from_buffer(0, 7, 1, vec![1]);
        assert!(matches!(
            mismatched,
            Err(ViprsError::RegionOutOfBounds {
                width: 0,
                height: 7,
                ..
            })
        ));

        for (width, height) in [(0, 0), (0, 7), (7, 0)] {
            let image = Image::<U8>::from_buffer(width, height, 1, Vec::new())
                .unwrap()
                .with_metadata(ImageMetadata::default());

            for (op_name, result) in [
                (
                    "invert",
                    execute_without_panicking(&image, "invert", |builder| builder.invert()),
                ),
                (
                    "resize",
                    execute_without_panicking(&image, "resize", |builder| {
                        builder.resize(Resize::new(0.5, 0.5, InterpolationKernel::Lanczos3))
                    }),
                ),
                (
                    "thumbnail",
                    execute_without_panicking(&image, "thumbnail", |builder| {
                        builder.thumbnail(Thumbnail::new(
                            ThumbnailTarget::FitBox {
                                width: 8,
                                height: 8,
                            },
                            InterpolationKernel::Lanczos3,
                        ))
                    }),
                ),
                (
                    "shrink",
                    execute_without_panicking(&image, "shrink", |builder| builder.shrink(2, 2)),
                ),
            ] {
                match result {
                    Ok((pipeline, output)) => {
                        assert_eq!(pipeline.width, output.width());
                        assert_eq!(pipeline.height, output.height());
                        assert_eq!(output.pixels().len(), 0);
                    }
                    Err(error) => match error {
                        ViprsError::Build(_)
                        | ViprsError::Scheduler(_)
                        | ViprsError::RegionOutOfBounds { .. }
                        | ViprsError::Source(_) => {}
                        other => panic!(
                            "{op_name} returned unexpected error for zero-dimension {width}x{height}: {other}"
                        ),
                    },
                }
            }
        }
    }
}
