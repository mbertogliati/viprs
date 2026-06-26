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
    fn pass3_dilate_identity_mask_handles_multiband_tiles() {
        let op = Dilate::new(vec![vec![255u8]]).unwrap();
        let region = Region::new(0, 0, 5, 3);

        for bands in [1, 2, 4] {
            let image = patterned_u8_image(region.width, region.height, bands);
            let output = run_unary_op(&op, region, bands, image.pixels(), region, bands);
            assert_eq!(
                output,
                image.pixels(),
                "dilate identity failed for {bands} bands"
            );
        }
    }

    #[test]
    fn pass3_erode_identity_mask_handles_multiband_tiles() {
        let op = Erode::new(vec![vec![255u8]]).unwrap();
        let region = Region::new(0, 0, 5, 3);

        for bands in [1, 2, 4] {
            let image = patterned_u8_image(region.width, region.height, bands);
            let output = run_unary_op(&op, region, bands, image.pixels(), region, bands);
            assert_eq!(
                output,
                image.pixels(),
                "erode identity failed for {bands} bands"
            );
        }
    }

    #[test]
    fn pass3_sobel_constant_field_zeroes_all_bands() {
        let op = Sobel::<U8>::new();
        let input_region = Region::new(0, 0, 7, 7);
        let output_region = Region::new(0, 0, 5, 5);

        for bands in [1, 2, 4] {
            let input = vec![42u8; sample_count(input_region, bands)];
            let output = run_unary_op(&op, input_region, bands, &input, output_region, bands);
            assert!(output.iter().all(|value| value.abs() < f32::EPSILON));
        }
    }

    #[test]
    fn pass3_sharpen_constant_field_preserves_multiband_values() {
        let op = Sharpen::<F32>::new(1.0, 1.5);
        let output_region = Region::new(0, 0, 5, 4);
        let spec = op.node_spec(output_region.width, output_region.height);
        let input_region = Region::new(0, 0, spec.input_tile_w, spec.input_tile_h);

        for bands in [1, 2, 4] {
            let input = vec![7.0f32; sample_count(input_region, bands)];
            let output = run_unary_op(&op, input_region, bands, &input, output_region, bands);
            assert!(
                output.iter().all(|value| (*value - 7.0).abs() < 1e-5),
                "sharpen changed a constant field for {bands} bands"
            );
        }
    }
}
