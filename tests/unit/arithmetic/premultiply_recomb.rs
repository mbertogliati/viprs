mod chaos_monkey_19 {
    use bytemuck::Pod;
    use viprs::{
        BuildError, F32, Image, ImageMetadata, Interpretation, U8, U16,
        adapters::{pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler},
        domain::{
            colorspace::{ColorspaceId, SRgb},
            kernel::InterpolationKernel,
            op::Op,
            ops::{
                arithmetic::{Matrix, RecombOp},
                conversion::embed::ExtendMode,
                resample::{Thumbnail, thumbnail::ThumbnailTarget},
            },
        },
    };

    fn srgb_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        }
    }

    fn make_u8_image(width: u32, height: u32, bands: u32, pixels: Vec<u8>) -> Image<U8> {
        let image = Image::from_buffer(width, height, bands, pixels).unwrap();
        if bands >= 3 {
            image.with_metadata(srgb_metadata())
        } else {
            image
        }
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

    fn execute_to_image<FIn, FOut, S: viprs::pipeline::Flush>(
        image: &Image<FIn>,
        configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder<S>, BuildError>,
    ) -> Result<(viprs::CompiledPipeline, Image<FOut>), String>
    where
        FIn: viprs::BandFormat,
        FOut: viprs::BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let pipeline = configure(PipelineBuilder::from_source(
            viprs::adapters::sources::memory::MemorySource::<FIn>::new(
                image.width(),
                image.height(),
                image.bands(),
                image.pixels().to_vec(),
            )
            .unwrap()
            .with_metadata(image.metadata().clone()),
        ))
        .map_err(|error| format!("stage failed: {error:?}"))?
        .build()
        .map_err(|error| format!("build failed: {error:?}"))?;

        let output = pipeline
            .run_to_image::<FOut, _>(
                &RayonScheduler::new(2).map_err(|error| format!("scheduler failed: {error}"))?,
            )
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

        Ok((pipeline, output))
    }

    fn patterned_rgb(width: u32, height: u32) -> Image<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 13 + 3) % 256) as u8);
                pixels.push(((x * 11 + y * 29 + 7) % 256) as u8);
                pixels.push(((x * 5 + y * 19 + 191) % 256) as u8);
            }
        }
        make_u8_image(width, height, 3, pixels)
    }

    fn rgba_image_with_partial_alpha() -> Image<U8> {
        make_u8_image(2, 1, 4, vec![200, 100, 50, 128, 30, 60, 90, 64])
    }

    #[test]
    fn cast_f32_nan_to_u8_does_not_panic_and_clamps_to_an_endpoint() {
        let image = make_f32_image(2, 1, 1, vec![f32::NAN, 0.5], ImageMetadata::default());
        let (_pipeline, output) =
            execute_to_image::<F32, U8, _>(&image, |builder| builder.cast(viprs::BandFormatId::U8))
                .expect("F32->U8 cast should succeed for NaN inputs");

        assert!(matches!(output.pixels()[0], 0 | 255));
        assert_eq!(output.pixels()[1], 128);
    }

    #[test]
    fn linear_u8_offset_254_saturates_instead_of_wrapping() {
        let image = make_u8_image(2, 1, 1, vec![1, 250]);
        let (_pipeline, output) =
            execute_to_image::<U8, U8, _>(&image, |builder| builder.linear(1.0, 254.0))
                .expect("linear should execute on U8");

        assert_eq!(output.pixels(), &[255, 255]);
    }

    #[test]
    fn double_premultiply_does_not_panic_and_preserves_alpha() {
        let image = rgba_image_with_partial_alpha();
        let (_pipeline, single) =
            execute_to_image::<U8, U8, _>(&image, |builder| builder.premultiply())
                .expect("single premultiply should succeed");
        let (_pipeline, doubled) =
            execute_to_image::<U8, U8, _>(&image, |builder| builder.premultiply()?.premultiply())
                .expect("double premultiply should succeed");

        assert_eq!(single.pixels()[3], image.pixels()[3]);
        assert_eq!(single.pixels()[7], image.pixels()[7]);
        assert_eq!(doubled.pixels()[3], image.pixels()[3]);
        assert_eq!(doubled.pixels()[7], image.pixels()[7]);
        assert!(doubled.pixels()[0] <= single.pixels()[0]);
        assert!(doubled.pixels()[1] <= single.pixels()[1]);
        assert!(doubled.pixels()[2] <= single.pixels()[2]);
    }

    #[test]
    fn invert_u16_uses_full_u16_range() {
        let image = make_u16_image(2, 1, 1, vec![0, u16::MAX]);
        let (_pipeline, output) =
            execute_to_image::<U16, U16, _>(&image, |builder| builder.invert())
                .expect("invert should execute on U16");

        assert_eq!(output.pixels(), &[u16::MAX, 0]);
    }

    #[test]
    fn recomb_negative_matrix_clamps_to_u8_range() {
        let matrix = Matrix::new(
            3,
            3,
            vec![
                -1.0, 1.0, 0.0, //
                0.0, -1.0, 1.0, //
                1.0, 0.0, -1.0,
            ],
        );
        let op = RecombOp::<U8>::new(matrix);
        let region = viprs::Region::new(0, 0, 1, 1);
        let input_data = vec![10u8, 20, 30];
        let mut output_data = vec![0u8; 3];
        let input = viprs::Tile::<U8>::new(region, 3, &input_data);
        let mut output = viprs::TileMut::<U8>::new(region, 3, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![10, 10, 0]);
    }
}
