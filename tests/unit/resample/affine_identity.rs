mod chaos_monkey_2 {
    use bytemuck::{Pod, cast_slice};
    use proptest::prelude::*;
    use viprs::{
        BandFormat, BandFormatId, BuildError, CompiledPipeline, F32, Image, ImageMetadata,
        Interpretation, U8,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
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

    fn patterned_u8_image(width: u32, height: u32, bands: u32) -> Image<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * bands as usize);
        for y in 0..height {
            for x in 0..width {
                for band in 0..bands {
                    pixels.push(((x * 31 + y * 17 + band * 53 + 11) % 256) as u8);
                }
            }
        }

        Image::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(if bands >= 3 {
                rgb_metadata()
            } else {
                ImageMetadata::default()
            })
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

    fn execute_to_buffer<F>(
        image: &Image<F>,
        configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder, BuildError>,
    ) -> Result<(CompiledPipeline, Vec<u8>), String>
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
    fn pass1_affine_identity_nearest_is_identity() {
        let image = patterned_u8_image(7, 5, 3);
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

        assert_eq!(output.pixels(), image.pixels());
    }

    #[test]
    fn pass1_similarity_identity_is_identity() {
        let image = patterned_u8_image(7, 5, 3);
        let (_pipeline, output) = execute_same_format(&image, |builder| {
            builder.similarity(1.0, 0.0, InterpolationKernel::Nearest)
        })
        .expect("identity similarity should succeed");

        assert_eq!(output.width(), image.width());
        assert_eq!(output.height(), image.height());
        assert_eq!(output.pixels(), image.pixels());
    }

    #[test]
    fn pass4_mapim_copy_extend_far_outside_matches_current_golden() {
        let source = vec![10u8, 20, 30, 40];
        let source_region = Region::new(0, 0, 2, 2);
        let index_region = Region::new(0, 0, 2, 1);
        let output_region = index_region;
        let index = vec![-100.0f32, -100.0, 250.0, 250.0];
        let op =
            MapImOp::<U8>::new(2, 2, 1, 2, 1, BandFormatId::F32).with_extend(MapImExtend::Copy);

        let output = run_mapim_u8(
            &op,
            &source,
            source_region,
            1,
            &index,
            index_region,
            output_region,
        );
        assert_eq!(output, vec![0, 0]);
    }

    #[test]
    fn pass5_lab_affine_srgb_output_stays_in_u8_range() {
        let image = patterned_u8_image(33, 19, 3);
        let (pipeline, buffer) = execute_to_buffer(&image, |builder| {
            builder
                .with_colorspace(ColorspaceId::SRgb)
                .colourspace::<Lab>()?
                .affine(
                    [1.0, 0.05, -0.05, 1.0],
                    1.0,
                    2.0,
                    33,
                    19,
                    InterpolationKernel::Bilinear,
                )?
                .colourspace::<SRgb>()
        })
        .expect("Lab affine roundtrip should succeed");

        assert_valid_buffer_len(&pipeline, &buffer);
    }

    #[test]
    fn pass5_mapim_identity_preserves_rgba_pixels() {
        let source = patterned_u8_image(3, 2, 4);
        let source_region = Region::new(0, 0, source.width(), source.height());
        let index_region = Region::new(0, 0, source.width(), source.height());
        let output_region = index_region;
        let index: Vec<f32> = (0..source.height())
            .flat_map(|y| (0..source.width()).flat_map(move |x| [x as f32, y as f32]))
            .collect();
        let op = MapImOp::<U8>::new(
            source.width(),
            source.height(),
            4,
            source.width(),
            source.height(),
            BandFormatId::F32,
        )
        .with_premultiplied(true);

        let output = run_mapim_u8(
            &op,
            source.pixels(),
            source_region,
            4,
            &index,
            index_region,
            output_region,
        );
        assert_eq!(output, source.pixels());
    }
}
