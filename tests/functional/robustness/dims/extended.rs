mod robustez_dims {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, Image, ImageMetadata, Interpretation, U8, ViprsError,
        adapters::{scheduler::rayon_scheduler::RayonScheduler, sources::memory::MemorySource},
        domain::{
            colorspace::{Colorspace, ColorspaceId, Lab, ScRgb, Xyz},
            kernel::InterpolationKernel,
            op::OperationBridge,
            ops::{
                arithmetic::{Matrix, RecombOp},
                conversion::{ExtendMode, rot::Angle},
                resample::{Thumbnail, thumbnail::ThumbnailTarget},
            },
        },
    };

    fn metadata_for_bands(bands: u32) -> ImageMetadata {
        let interpretation = match bands {
            1 | 2 => Some(Interpretation::BW),
            3 | 4 => Some(Interpretation::Srgb),
            _ => None,
        };
        ImageMetadata {
            interpretation,
            ..ImageMetadata::default()
        }
    }

    fn patterned_u8(width: u32, height: u32, bands: u32) -> Image<U8> {
        let len = width as usize * height as usize * bands as usize;
        let pixels = (0..len)
            .map(|index| ((index * 29 + 17) % 251) as u8)
            .collect();
        Image::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(metadata_for_bands(bands))
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

    fn execute_without_panicking<FIn, FOut, S: viprs_runtime::pipeline::internal::Flush>(
        image: &Image<FIn>,
        op_name: &str,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelineBuilder,
        ) -> Result<
            viprs_runtime::pipeline::internal::PipelineBuilder<S>,
            BuildError,
        >,
    ) -> Result<(CompiledPipeline, Image<FOut>), ViprsError>
    where
        FIn: viprs::BandFormat,
        FOut: viprs::BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let pipeline = configure(
                viprs_runtime::pipeline::internal::PipelineBuilder::from_source(
                    memory_source_from_image(image)?,
                ),
            )?
            .build()?;
            let scheduler =
                RayonScheduler::new(2).map_err(|error| ViprsError::Scheduler(error.to_string()))?;
            let output = pipeline.run_to_image::<FOut, _>(&scheduler)?;
            Ok((pipeline, output))
        }));

        assert!(
            result.is_ok(),
            "{op_name} panicked for {}x{} image",
            image.width(),
            image.height()
        );

        result.unwrap()
    }

    fn configure_without_panicking<S: viprs_runtime::pipeline::internal::Flush>(
        image: &Image<U8>,
        op_name: &str,
        configure: impl FnOnce(
            viprs_runtime::pipeline::internal::PipelineBuilder,
        ) -> Result<
            viprs_runtime::pipeline::internal::PipelineBuilder<S>,
            BuildError,
        >,
    ) -> Result<viprs_runtime::pipeline::internal::PipelineBuilder<S>, ViprsError> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            configure(
                viprs_runtime::pipeline::internal::PipelineBuilder::from_source(
                    memory_source_from_image(image)?,
                ),
            )
            .map_err(Into::into)
        }));

        assert!(
            result.is_ok(),
            "{op_name} panicked for {}x{} image",
            image.width(),
            image.height()
        );

        result.unwrap()
    }

    fn expected_reduce_len(input_len: u32, factor: f64) -> u32 {
        ((input_len as f64 / factor).round().max(1.0)) as u32
    }

    fn pixel_at(image: &Image<U8>, x: u32, y: u32) -> &[u8] {
        let bands = image.bands() as usize;
        let start = ((y * image.width() + x) as usize) * bands;
        &image.pixels()[start..start + bands]
    }

    fn assert_colour_request_is_typed_or_builds<To: Colorspace>(image: &Image<U8>, label: &str) {
        match configure_without_panicking(image, label, |builder| {
            builder
                .with_colorspace(ColorspaceId::Greyscale)
                .colourspace::<To>()
        }) {
            Ok(builder) => {
                let built = catch_unwind(AssertUnwindSafe(|| builder.build()));
                assert!(built.is_ok(), "{label} build panicked");
                match built.unwrap() {
                    Ok(pipeline) => assert_eq!(
                        (pipeline.width, pipeline.height),
                        (image.width(), image.height())
                    ),
                    Err(BuildError::InvalidColourConversionInput { bands: 1, .. })
                    | Err(BuildError::UnsupportedColourConversion { .. }) => {}
                    Err(error) => panic!("{label} returned unexpected build error: {error:?}"),
                }
            }
            Err(ViprsError::Build(BuildError::InvalidColourConversionInput {
                bands: 1, ..
            }))
            | Err(ViprsError::Build(BuildError::UnsupportedColourConversion { .. })) => {}
            Err(error) => panic!("{label} returned unexpected error: {error:?}"),
        }
    }

    #[test]
    fn one_by_one_image_ops_return_results_without_panicking() {
        let grey = Image::from_buffer(1, 1, 1, vec![37u8])
            .unwrap()
            .with_metadata(metadata_for_bands(1));
        let rgba = Image::from_buffer(1, 1, 4, vec![10u8, 20, 30, 255])
            .unwrap()
            .with_metadata(metadata_for_bands(4));

        let (_pipeline, inverted) =
            execute_without_panicking::<U8, U8, _>(&grey, "invert", |builder| builder.invert())
                .unwrap_or_else(|error| panic!("invert should succeed on 1x1: {error:?}"));
        assert_eq!(
            (inverted.width(), inverted.height(), inverted.bands()),
            (1, 1, 1)
        );
        assert_eq!(inverted.pixels(), &[218]);

        let (_pipeline, linear) =
            execute_without_panicking::<U8, U8, _>(&grey, "linear", |builder| {
                builder.linear(2.0, 3.0)
            })
            .unwrap_or_else(|error| panic!("linear should succeed on 1x1: {error:?}"));
        assert_eq!(linear.pixels(), &[77]);

        let (_pipeline, recombined) =
            execute_without_panicking::<U8, U8, _>(&grey, "recomb", |builder| {
                builder.then(Box::new(OperationBridge::with_dynamic_bands_pixel_local(
                    RecombOp::<U8>::new(Matrix::identity(1)),
                    1,
                    1,
                )))
            })
            .unwrap_or_else(|error| panic!("recomb should succeed on 1x1: {error:?}"));
        assert_eq!(recombined.pixels(), grey.pixels());

        let (_pipeline, shrunk) =
            execute_without_panicking::<U8, U8, _>(&grey, "shrink", |builder| builder.shrink(1, 1))
                .unwrap_or_else(|error| panic!("shrink should succeed on 1x1: {error:?}"));
        assert_eq!((shrunk.width(), shrunk.height()), (1, 1));

        let reduce = execute_without_panicking::<U8, U8, _>(&grey, "reduce", |builder| {
            builder.reduce(2.0, 2.0, InterpolationKernel::Lanczos3)
        });
        match reduce {
            Ok((_pipeline, output)) => assert_eq!((output.width(), output.height()), (1, 1)),
            Err(ViprsError::Build(BuildError::InvalidReduceParameters { .. })) => {}
            Err(error) => panic!("reduce returned an unexpected error on 1x1: {error:?}"),
        }

        let (_pipeline, thumbed) =
            execute_without_panicking::<U8, U8, _>(&grey, "thumbnail", |builder| {
                builder.thumbnail(Thumbnail::new(
                    ThumbnailTarget::Width(1),
                    InterpolationKernel::Lanczos3,
                ))
            })
            .unwrap_or_else(|error| panic!("thumbnail should succeed on 1x1: {error:?}"));
        assert_eq!((thumbed.width(), thumbed.height()), (1, 1));

        let (_pipeline, extracted) =
            execute_without_panicking::<U8, U8, _>(&grey, "extract_area", |builder| {
                builder.extract_area(0, 0, 1, 1)
            })
            .unwrap_or_else(|error| panic!("extract_area should succeed on 1x1: {error:?}"));
        assert_eq!(extracted.pixels(), grey.pixels());

        let (_pipeline, embedded) =
            execute_without_panicking::<U8, U8, _>(&grey, "embed", |builder| {
                builder.embed(1, 1, 0, 0, 1, 1, ExtendMode::Black)
            })
            .unwrap_or_else(|error| panic!("embed should succeed on 1x1: {error:?}"));
        assert_eq!(embedded.pixels(), grey.pixels());

        let (_pipeline, flattened) =
            execute_without_panicking::<U8, U8, _>(&rgba, "flatten", |builder| {
                builder.flatten([0.0, 0.0, 0.0, 1.0])
            })
            .unwrap_or_else(|error| panic!("flatten should succeed on 1x1 RGBA: {error:?}"));
        assert_eq!(
            (flattened.width(), flattened.height(), flattened.bands()),
            (1, 1, 3)
        );
        assert_eq!(flattened.pixels(), &[10, 20, 30]);

        let (_pipeline, rotated) =
            execute_without_panicking::<U8, U8, _>(&grey, "rotate", |builder| {
                builder.rotate(Angle::D90)
            })
            .unwrap_or_else(|error| panic!("rotate should succeed on 1x1: {error:?}"));
        assert_eq!((rotated.width(), rotated.height()), (1, 1));
        assert_eq!(rotated.pixels(), grey.pixels());
    }

    #[test]
    fn single_axis_images_handle_shrink_reduce_and_thumbnail() {
        let cases = [
            (
                (1, 17),
                (1, 2),
                (1.0, 2.0),
                ThumbnailTarget::Height(9),
                (1, 9),
            ),
            (
                (17, 1),
                (2, 1),
                (2.0, 1.0),
                ThumbnailTarget::Width(9),
                (9, 1),
            ),
        ];

        for ((width, height), (h_shrink, v_shrink), (h_reduce, v_reduce), target, expected_dims) in
            cases
        {
            let image = patterned_u8(width, height, 1);

            let (_pipeline, shrunk) =
                execute_without_panicking::<U8, U8, _>(&image, "shrink", |builder| {
                    builder.shrink(h_shrink, v_shrink)
                })
                .unwrap_or_else(|error| {
                    panic!("shrink should succeed for {width}x{height}: {error:?}")
                });
            assert!(shrunk.width() >= 1 && shrunk.width() <= width);
            assert!(shrunk.height() >= 1 && shrunk.height() <= height);

            let (_pipeline, reduced) =
                execute_without_panicking::<U8, U8, _>(&image, "reduce", |builder| {
                    builder.reduce(h_reduce, v_reduce, InterpolationKernel::Lanczos3)
                })
                .unwrap_or_else(|error| {
                    panic!("reduce should succeed for {width}x{height}: {error:?}")
                });
            assert!(reduced.width() >= 1 && reduced.width() <= width);
            assert!(reduced.height() >= 1 && reduced.height() <= height);

            let thumbnail = Thumbnail::new(target, InterpolationKernel::Lanczos3);
            let plan =
                thumbnail.into_pipeline_nodes_without_shrink_hint(width, height, image.bands());
            let (_pipeline, thumbed) =
                execute_without_panicking::<U8, U8, _>(&image, "thumbnail", |builder| {
                    builder.thumbnail(thumbnail)
                })
                .unwrap_or_else(|error| {
                    panic!("thumbnail should succeed for {width}x{height}: {error:?}")
                });
            assert_eq!(
                (thumbed.width(), thumbed.height()),
                (plan.output_width, plan.output_height)
            );
            assert!(thumbed.width() >= 1 && thumbed.width() <= width);
            assert!(thumbed.height() >= 1 && thumbed.height() <= height);
            let _ = expected_dims;
        }
    }

    #[test]
    fn max_safe_dimensions_survive_a_cheap_operation() {
        let image = patterned_u8(65_535, 1, 1);
        let (_pipeline, output) =
            execute_without_panicking::<U8, U8, _>(&image, "invert", |builder| builder.invert())
                .unwrap_or_else(|error| panic!("invert should succeed on 65535x1: {error:?}"));

        assert_eq!(
            (output.width(), output.height(), output.bands()),
            (65_535, 1, 1)
        );
        assert_eq!(output.pixels().len(), image.pixels().len());
    }

    #[test]
    fn non_power_of_two_dimensions_handle_shrink_reduce_and_thumbnail() {
        for (width, height) in [(777, 333), (1001, 999)] {
            let image = patterned_u8(width, height, 3);

            let (_pipeline, shrunk) =
                execute_without_panicking::<U8, U8, _>(&image, "shrink", |builder| {
                    builder.shrink(3, 2)
                })
                .unwrap_or_else(|error| {
                    panic!("shrink should succeed for {width}x{height}: {error:?}")
                });
            assert!(shrunk.width() >= 1 && shrunk.width() <= width.div_ceil(3));
            assert!(shrunk.height() >= 1 && shrunk.height() <= height.div_ceil(2));

            let (_pipeline, reduced) =
                execute_without_panicking::<U8, U8, _>(&image, "reduce", |builder| {
                    builder.reduce(2.5, 3.0, InterpolationKernel::Lanczos3)
                })
                .unwrap_or_else(|error| {
                    panic!("reduce should succeed for {width}x{height}: {error:?}")
                });
            assert_eq!(
                (reduced.width(), reduced.height()),
                (
                    expected_reduce_len(width, 2.5),
                    expected_reduce_len(height, 3.0)
                )
            );

            let thumbnail = Thumbnail::new(
                ThumbnailTarget::FitBox {
                    width: 111,
                    height: 87,
                },
                InterpolationKernel::Lanczos3,
            );
            let plan =
                thumbnail.into_pipeline_nodes_without_shrink_hint(width, height, image.bands());
            let (_pipeline, thumbed) =
                execute_without_panicking::<U8, U8, _>(&image, "thumbnail", |builder| {
                    builder.thumbnail(thumbnail)
                })
                .unwrap_or_else(|error| {
                    panic!("thumbnail should succeed for {width}x{height}: {error:?}")
                });
            assert_eq!(
                (thumbed.width(), thumbed.height()),
                (plan.output_width, plan.output_height)
            );
        }
    }

    #[test]
    fn single_band_grayscale_colour_requests_are_typed_or_supported() {
        let image = patterned_u8(9, 7, 1);
        assert_colour_request_is_typed_or_builds::<Lab>(&image, "colourspace::<Lab>");
        assert_colour_request_is_typed_or_builds::<Xyz>(&image, "colourspace::<Xyz>");
        assert_colour_request_is_typed_or_builds::<ScRgb>(&image, "colourspace::<ScRgb>");
    }

    #[test]
    fn extract_area_handles_zero_offset_edges_and_center() {
        let image = patterned_u8(9, 7, 3);
        let points = [
            (0, 0),
            (image.width() - 1, image.height() - 1),
            (image.width() / 2, image.height() / 2),
        ];

        for (x, y) in points {
            let (_pipeline, extracted) =
                execute_without_panicking::<U8, U8, _>(&image, "extract_area", |builder| {
                    builder.extract_area(x, y, 1, 1)
                })
                .unwrap_or_else(|error| {
                    panic!("extract_area should succeed at ({x}, {y}): {error:?}")
                });

            assert_eq!(
                (extracted.width(), extracted.height(), extracted.bands()),
                (1, 1, image.bands())
            );
            assert_eq!(extracted.pixels(), pixel_at(&image, x, y));
        }
    }

    #[test]
    fn embed_exact_fit_preserves_dimensions_and_pixels() {
        let image = patterned_u8(13, 11, 3);
        let (_pipeline, embedded) =
            execute_without_panicking::<U8, U8, _>(&image, "embed", |builder| {
                builder.embed(
                    image.width(),
                    image.height(),
                    0,
                    0,
                    image.width(),
                    image.height(),
                    ExtendMode::Black,
                )
            })
            .unwrap_or_else(|error| panic!("embed exact fit should succeed: {error:?}"));

        assert_eq!(
            (embedded.width(), embedded.height(), embedded.bands()),
            (image.width(), image.height(), image.bands())
        );
        assert_eq!(embedded.pixels(), image.pixels());
    }

    #[test]
    fn thumbnail_target_matching_source_is_exact_identity() {
        let image = patterned_u8(37, 19, 3);
        let thumbnail = Thumbnail::new(
            ThumbnailTarget::FitBox {
                width: image.width(),
                height: image.height(),
            },
            InterpolationKernel::Lanczos3,
        );
        let (_pipeline, output) =
            execute_without_panicking::<U8, U8, _>(&image, "thumbnail", |builder| {
                builder.thumbnail(thumbnail)
            })
            .unwrap_or_else(|error| panic!("thumbnail identity should succeed: {error:?}"));

        assert_eq!(
            (output.width(), output.height(), output.bands()),
            (37, 19, 3)
        );
        assert_eq!(output.pixels(), image.pixels());
    }

    #[test]
    fn very_large_reduce_factor_collapses_to_one_pixel() {
        let image = patterned_u8(100, 100, 1);
        let (_pipeline, output) =
            execute_without_panicking::<U8, U8, _>(&image, "reduce", |builder| {
                builder.reduce(100.0, 100.0, InterpolationKernel::Lanczos3)
            })
            .unwrap_or_else(|error| panic!("reduce(100) should succeed: {error:?}"));

        assert_eq!((output.width(), output.height(), output.bands()), (1, 1, 1));
        assert_eq!(output.pixels().len(), 1);
    }

    #[test]
    fn ten_shrink_two_steps_end_at_one_pixel() {
        let image = patterned_u8(1024, 1024, 1);
        let (_pipeline, output) =
            execute_without_panicking::<U8, U8, _>(&image, "shrink-chain", |builder| {
                let mut builder = builder;
                for _ in 0..10 {
                    builder = builder.shrink(2, 2)?;
                }
                Ok(builder)
            })
            .unwrap_or_else(|error| panic!("ten shrink(2) steps should succeed: {error:?}"));

        assert_eq!((output.width(), output.height(), output.bands()), (1, 1, 1));
        assert_eq!(output.pixels().len(), 1);
    }
}
