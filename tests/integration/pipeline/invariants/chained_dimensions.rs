mod chaos_monkey_16 {
    use std::num::NonZeroUsize;

    use bytemuck::Pod;
    use viprs::{
        BandFormatId, BuildError, F32, ImageMetadata, InMemoryImage, Interpretation, U8, U16,
        adapters::{
            pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
            sources::memory::MemorySource,
        },
        domain::{
            colorspace::{Cmyk, Lab, SRgb},
            kernel::InterpolationKernel,
            op::OperationBridge,
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

    fn make_u8_image(width: u32, height: u32, bands: u32, pixels: Vec<u8>) -> InMemoryImage<U8> {
        let image = InMemoryImage::from_buffer(width, height, bands, pixels).unwrap();
        if bands >= 3 {
            image.with_metadata(srgb_metadata())
        } else {
            image
        }
    }

    fn make_u16_image(width: u32, height: u32, bands: u32, pixels: Vec<u16>) -> InMemoryImage<U16> {
        InMemoryImage::from_buffer(width, height, bands, pixels).unwrap()
    }

    fn make_f32_image(
        width: u32,
        height: u32,
        bands: u32,
        pixels: Vec<f32>,
        metadata: ImageMetadata,
    ) -> InMemoryImage<F32> {
        InMemoryImage::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(metadata)
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

    fn execute_to_image<FIn, FOut, S: viprs::pipeline::Commit>(
        image: &InMemoryImage<FIn>,
        configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> Result<(viprs::CompiledPipeline, InMemoryImage<FOut>), String>
    where
        FIn: viprs::BandFormat,
        FOut: viprs::BandFormat,
        FIn::Sample: Pod,
        FOut::Sample: Pod,
    {
        let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(image)))
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

    fn patterned_rgb(width: u32, height: u32) -> InMemoryImage<U8> {
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

    fn patterned_rgba(width: u32, height: u32) -> InMemoryImage<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 13 + 3) % 256) as u8);
                pixels.push(((x * 11 + y * 29 + 7) % 256) as u8);
                pixels.push(((x * 5 + y * 19 + 191) % 256) as u8);
                pixels.push(((x * 23 + y * 31 + 255) % 256) as u8);
            }
        }
        make_u8_image(width, height, 4, pixels)
    }

    fn thumbnail(width: u32) -> Thumbnail {
        Thumbnail::new(ThumbnailTarget::Width(width), InterpolationKernel::Lanczos3)
    }

    fn max_abs_diff(a: &[u8], b: &[u8]) -> u8 {
        a.iter()
            .zip(b)
            .map(|(left, right)| left.abs_diff(*right))
            .max()
            .unwrap_or(0)
    }

    fn recomb_matrix() -> Matrix {
        Matrix::new(
            3,
            3,
            vec![
                1.0, 0.0, 0.0, //
                0.25, 0.5, 0.25, //
                0.0, 0.0, 1.0,
            ],
        )
    }

    fn append_recomb(builder: ImagePipeline) -> Result<ImagePipeline, BuildError> {
        builder.then(Box::new(OperationBridge::with_dynamic_bands_pixel_local(
            RecombOp::<U8>::new(recomb_matrix()),
            3,
            3,
        )))
    }

    fn expected_embed_repeat(
        src: &InMemoryImage<U8>,
        dst_width: u32,
        dst_height: u32,
        x_off: i32,
        y_off: i32,
    ) -> Vec<u8> {
        let mut output =
            Vec::with_capacity(dst_width as usize * dst_height as usize * src.bands() as usize);
        let src_width = src.width() as i32;
        let src_height = src.height() as i32;
        let bands = src.bands() as usize;

        for y in 0..dst_height as i32 {
            for x in 0..dst_width as i32 {
                let src_x = (x - x_off).rem_euclid(src_width) as usize;
                let src_y = (y - y_off).rem_euclid(src_height) as usize;
                let index = (src_y * src.width() as usize + src_x) * bands;
                output.extend_from_slice(&src.pixels()[index..index + bands]);
            }
        }

        output
    }

    #[test]
    fn thumbnail_recomb_thumbnail_matches_sequential_dimensions() {
        let image = patterned_rgb(37, 23);
        let (_first_pipeline, first) =
            execute_to_image::<U8, U8, _>(&image, |builder| builder.thumbnail_with(thumbnail(19)))
                .expect("first thumbnail should succeed");
        let (_second_pipeline, recombined) =
            execute_to_image::<U8, U8, _>(&first, append_recomb).expect("recomb should succeed");
        let (_third_pipeline, sequential) = execute_to_image::<U8, U8, _>(&recombined, |builder| {
            builder.thumbnail_with(thumbnail(9))
        })
        .expect("second thumbnail should succeed");

        let (pipeline, chained) = execute_to_image::<U8, U8, _>(&image, |builder| {
            let builder = builder.thumbnail_with(thumbnail(19))?;
            let builder = append_recomb(builder)?;
            builder.thumbnail_with(thumbnail(9))
        })
        .expect("chained thumbnail -> recomb -> thumbnail should succeed");

        assert_eq!(
            (pipeline.width, pipeline.height, chained.bands()),
            (sequential.width(), sequential.height(), sequential.bands())
        );
        assert_eq!(chained.pixels().len(), sequential.pixels().len());
    }
}
