mod chaos_monkey_2 {
    use bytemuck::{Pod, cast_slice};
    use proptest::prelude::*;
    use viprs::{
        BandFormat, BandFormatId, BuildError, CompiledPipeline, F32, InMemoryImage, ImageMetadata,
        Interpretation, U8,
        adapters::{
          pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
          sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            colorspace::{ColorspaceId, Hsv, Lab, SRgb},
            image::{Region, Tile, TileMut},
            kernel::InterpolationKernel,
            op::{DynOperation, Op, OperationBridge},
            ops::{
                arithmetic::{Matrix, RecombOp},
                conversion::{Angle45, ExtendMode},
                convolution::{Sharpen, Sobel},
                morphology::{Dilate, Erode},
                resample::{MapImOp, mapim::MapImExtend},
            },
        },
        ports::scheduler::TileScheduler,
    };

    fn rgb_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        }
    }

    fn patterned_u8_image(width: u32, height: u32, bands: u32) -> InMemoryImage<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * bands as usize);
        for y in 0..height {
            for x in 0..width {
                for band in 0..bands {
                    pixels.push(((x * 31 + y * 17 + band * 53 + 11) % 256) as u8);
                }
            }
        }

        InMemoryImage::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(if bands >= 3 {
                rgb_metadata()
            } else {
                ImageMetadata::default()
            })
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
        let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(
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

    fn execute_to_buffer<F, S: viprs::pipeline::Commit>(
        image: &InMemoryImage<F>,
        configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(CompiledPipeline, Vec<u8>), String>
    where
        F: viprs::BandFormat,
        F::Sample: Pod,
    {
        let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(
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

        Ok((pipeline, sink.into_buffer()))
    }

    fn expected_buffer_len(pipeline: &CompiledPipeline) -> usize {
        let bytes_per_sample = match pipeline.output_format {
            BandFormatId::U8 => 1,
            BandFormatId::U16 | BandFormatId::I16 => 2,
            BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
            BandFormatId::F64 => 8,
        };

        pipeline.width as usize
            * pipeline.height as usize
            * pipeline.output_bands as usize
            * bytes_per_sample
    }

    fn assert_valid_buffer_len(pipeline: &CompiledPipeline, buffer: &[u8]) {
        assert_eq!(
            buffer.len(),
            expected_buffer_len(pipeline),
            "buffer length mismatch for {}x{}x{} {:?}",
            pipeline.width,
            pipeline.height,
            pipeline.output_bands,
            pipeline.output_format
        );
    }

    fn assert_u8_pixels_within_tolerance(left: &[u8], right: &[u8], tolerance: u8) {
        assert_eq!(left.len(), right.len());
        for (index, (&lhs, &rhs)) in left.iter().zip(right.iter()).enumerate() {
            let diff = lhs.abs_diff(rhs);
            assert!(
                diff <= tolerance,
                "pixel mismatch at index {index}: {lhs} vs {rhs} (tolerance {tolerance})"
            );
        }
    }

    fn sample_count(region: Region, bands: u32) -> usize {
        region.width as usize * region.height as usize * bands as usize
    }

    fn run_unary_op<T>(
        op: &T,
        input_region: Region,
        input_bands: u32,
        input_data: &[<T::Input as BandFormat>::Sample],
        output_region: Region,
        output_bands: u32,
    ) -> Vec<<T::Output as BandFormat>::Sample>
    where
        T: Op,
        T::Input: BandFormat,
        T::Output: BandFormat,
        <T::Output as BandFormat>::Sample: Copy + Default,
    {
        let mut output = vec![
            <T::Output as BandFormat>::Sample::default();
            sample_count(output_region, output_bands)
        ];
        let input = Tile::<T::Input>::new(input_region, input_bands, input_data);
        let mut output_tile = TileMut::<T::Output>::new(output_region, output_bands, &mut output);
        let mut state =
            op.start_with_tile_and_bands(output_region.width, output_region.height, input_bands);
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    fn run_mapim_u8(
        op: &MapImOp<U8>,
        source: &[u8],
        source_region: Region,
        source_bands: u32,
        index: &[f32],
        index_region: Region,
        output_region: Region,
    ) -> Vec<u8> {
        let mut output = vec![0u8; sample_count(output_region, source_bands)];
        let inputs: &[&[u8]] = &[cast_slice(source), cast_slice(index)];
        let input_regions = [source_region, index_region];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            &mut output,
            &input_regions,
            output_region,
        );
        output
    }

    #[test]
    fn pass2_embed_extract_buffer_len_matches_declared_dimensions() {
        let image = patterned_u8_image(777, 333, 4);
        let (pipeline, buffer) = execute_to_buffer(&image, |builder| {
            builder
                .embed(
                    1024,
                    768,
                    123,
                    77,
                    image.width(),
                    image.height(),
                    ExtendMode::Black,
                )?
                .extract_area(123, 77, image.width(), image.height())
        })
        .expect("embed/extract pipeline should succeed");

        assert_eq!(
            (pipeline.width, pipeline.height),
            (image.width(), image.height())
        );
        assert_eq!(pipeline.output_bands, image.bands());
        assert_valid_buffer_len(&pipeline, &buffer);
    }

    #[test]
    fn pass2_extract_then_embed_restores_canvas_dimensions() {
        let image = patterned_u8_image(512, 512, 3);
        let (pipeline, buffer) = execute_to_buffer(&image, |builder| {
            builder.extract_area(10, 20, 100, 80)?.embed(
                512,
                512,
                10,
                20,
                100,
                80,
                ExtendMode::Black,
            )
        })
        .expect("extract then embed should succeed");

        assert_eq!((pipeline.width, pipeline.height), (512, 512));
        assert_valid_buffer_len(&pipeline, &buffer);
    }

    #[test]
    fn pass2_affine_declares_output_dimensions_consistently() {
        let image = patterned_u8_image(777, 333, 3);
        let output_w = 913;
        let output_h = 271;
        let (pipeline, buffer) = execute_to_buffer(&image, |builder| {
            builder.affine(
                [1.0, 0.2, -0.1, 1.0],
                3.5,
                7.25,
                output_w,
                output_h,
                InterpolationKernel::Bilinear,
            )
        })
        .expect("affine should succeed");

        assert_eq!((pipeline.width, pipeline.height), (output_w, output_h));
        assert_valid_buffer_len(&pipeline, &buffer);
    }

    #[test]
    fn pass2_similarity_declares_auto_canvas_consistently() {
        let image = patterned_u8_image(257, 129, 3);
        let (pipeline, buffer) = execute_to_buffer(&image, |builder| {
            builder.similarity(1.25, 17.0, InterpolationKernel::Bilinear)
        })
        .expect("similarity should succeed");

        assert!(pipeline.width > 0);
        assert!(pipeline.height > 0);
        assert_valid_buffer_len(&pipeline, &buffer);
    }

    #[test]
    fn pass2_fractional_reduce_matches_declared_dimensions() {
        let image = patterned_u8_image(777, 333, 3);
        for factor in [1.5, 2.7, 3.14, 7.5] {
            let (pipeline, buffer) = execute_to_buffer(&image, |builder| {
                builder.reduce(factor, factor, InterpolationKernel::Lanczos3)
            })
            .unwrap_or_else(|error| {
                panic!("fractional reduce should succeed for factor {factor}: {error}")
            });

            assert!(pipeline.width > 0);
            assert!(pipeline.height > 0);
            assert_valid_buffer_len(&pipeline, &buffer);
        }
    }

    #[test]
    fn pass2_rot45_on_odd_dimensions_matches_declared_dimensions() {
        for size in [5, 11] {
            let image = patterned_u8_image(size, size, 3);
            let (pipeline, buffer) =
                execute_to_buffer(&image, |builder| builder.rot45(Angle45::D45)).unwrap_or_else(
                    |error| panic!("rot45 should succeed on odd {size}x{size}: {error}"),
                );

            assert!(pipeline.width > 0);
            assert!(pipeline.height > 0);
            assert_valid_buffer_len(&pipeline, &buffer);
        }
    }

    #[test]
    fn pass3_affine_identity_handles_common_band_counts() {
        for bands in [1, 2, 3, 4] {
            let image = patterned_u8_image(9, 7, bands);
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
            .expect("band-count affine identity should succeed");

            assert_eq!(output.pixels(), image.pixels(), "failed for {bands} bands");
        }
    }

    #[test]
    fn pass3_embed_extract_handles_common_band_counts() {
        for bands in [1, 2, 3, 4] {
            let image = patterned_u8_image(11, 9, bands);
            let (_pipeline, output) = execute_same_format(&image, |builder| {
                builder
                    .embed(
                        17,
                        15,
                        3,
                        2,
                        image.width(),
                        image.height(),
                        ExtendMode::Black,
                    )?
                    .extract_area(3, 2, image.width(), image.height())
            })
            .expect("band-count embed/extract should succeed");

            assert_eq!(output.pixels(), image.pixels(), "failed for {bands} bands");
        }
    }

    #[test]
    fn pass5_hsv_similarity_srgb_output_buffer_matches_declared_dimensions() {
        let image = patterned_u8_image(41, 23, 3);
        let (pipeline, buffer) = execute_to_buffer(&image, |builder| {
            builder
                .with_colorspace(ColorspaceId::SRgb)
                .colourspace::<Hsv>()?
                .similarity(1.0, 0.0, InterpolationKernel::Nearest)?
                .colourspace::<SRgb>()
        })
        .expect("HSV similarity roundtrip should succeed");

        assert_eq!(
            (pipeline.width, pipeline.height),
            (image.width(), image.height())
        );
        assert_valid_buffer_len(&pipeline, &buffer);
    }

    #[test]
    fn pass5_recomb_pipeline_identity_preserves_dimensions_and_values() {
        let image: InMemoryImage<F32> =
            InMemoryImage::from_buffer(2, 1, 3, vec![0.25f32, -0.5, 1.0, 0.0, 0.5, -1.0]).unwrap();
        let matrix = Matrix::identity(3);
        let (_pipeline, output) = execute_same_format(&image, |builder| {
            builder.then(Box::new(OperationBridge::with_dynamic_bands_pixel_local(
                RecombOp::<F32>::new(matrix.clone()),
                3,
                3,
            )))
        })
        .expect("recomb pipeline identity should succeed");

        assert_eq!(output.pixels(), image.pixels());
    }

    #[test]
    fn pass5_sobel_pipeline_reports_f32_output_length_correctly() {
        let image = patterned_u8_image(9, 7, 3);
        let (pipeline, buffer) = execute_to_buffer(&image, |builder| {
            builder.then(Box::new(OperationBridge::new(
                Sobel::<U8>::new(),
                image.bands(),
            )))
        })
        .expect("sobel pipeline should succeed");

        assert_eq!(pipeline.output_format, BandFormatId::F32);
        assert_eq!(pipeline.output_bands, image.bands());
        assert_valid_buffer_len(&pipeline, &buffer);
        let samples = cast_slice::<u8, f32>(&buffer);
        assert!(samples.iter().all(|value| value.is_finite()));
    }
}
