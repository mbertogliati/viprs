#![cfg(any(feature = "jpeg", feature = "png", feature = "webp", feature = "gif"))]

mod chaos_monkey_14 {
    use bytemuck::Pod;
    use viprs::{
        BuildError, CompiledPipeline, ImageMetadata, InMemoryImage, Interpretation, U8,
        adapters::{
            pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
            sources::memory::MemorySource,
        },
        domain::ops::conversion::ExtendMode,
    };

    #[cfg(feature = "jpeg")]
    use viprs::adapters::codecs::JpegCodec;
    #[cfg(feature = "jpeg")]
    use viprs::{
        domain::codec_options::SaveOptions,
        ports::codec::{ImageDecoder, ImageEncoder},
    };

    fn srgb_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        }
    }

    fn patterned_u8(width: u32, height: u32, bands: u32) -> InMemoryImage<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * bands as usize);
        for y in 0..height {
            for x in 0..width {
                for band in 0..bands {
                    let value = match band {
                        0 => ((x * 17 + y * 13 + 19) % 251) as u8,
                        1 => ((x * 11 + y * 29 + 7) % 253) as u8,
                        2 => ((x * 5 + y * 19 + 191) % 255) as u8,
                        _ => ((x * 23 + y * 31 + band * 47 + 3) % 249) as u8,
                    };
                    pixels.push(value);
                }
            }
        }

        InMemoryImage::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(srgb_metadata())
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
    ) -> Result<(CompiledPipeline, InMemoryImage<FOut>), String>
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
            .run_to_image::<FOut, _>(&RayonScheduler::new(2).map_err(|error| error.to_string())?)
            .map_err(|error| format!("pipeline execution failed: {error:?}"))?;
        Ok((pipeline, output))
    }

    #[cfg(feature = "jpeg")]
    #[test]
    #[ignore = "BUG: JPEG quality=0 is documented as valid but encode_with_options rejects it"]
    fn jpeg_quality_zero_produces_valid_decodable_jpeg() {
        let image = patterned_u8(17, 11, 3);
        let codec = JpegCodec;
        let encoded = codec
            .encode_with_options(&image, &SaveOptions::default().with_quality(0))
            .expect("quality=0 should still encode a JPEG");
        let decoded = codec
            .decode::<U8>(&encoded)
            .expect("quality=0 JPEG should decode successfully");

        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (17, 11, 3)
        );
    }
}

mod chaos_monkey_15 {
    use bytemuck::Pod;
    use viprs::{
        BuildError, F32, ImageMetadata, InMemoryImage, Interpretation, U8,
        adapters::{
            pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
            sources::memory::MemorySource,
        },
        domain::{
            codec_options::SaveOptions,
            colorspace::{Lab, SRgb},
            kernel::InterpolationKernel,
            ops::{
                arithmetic::{Matrix, RecombOp},
                resample::{Thumbnail, thumbnail::ThumbnailTarget},
            },
        },
        ports::codec::{ImageDecoder, ImageEncoder},
    };

    #[cfg(feature = "png")]
    use viprs::adapters::codecs::PngCodec;
    #[cfg(feature = "webp")]
    use viprs::adapters::codecs::WebpCodec;

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

    #[cfg(feature = "png")]
    #[test]
    fn png_roundtrip_is_pixel_exact() {
        let image = patterned_rgba(13, 9);
        let codec = PngCodec::default();
        let encoded = codec.encode(&image).expect("png encode should succeed");
        let decoded: InMemoryImage<U8> = codec.decode(&encoded).expect("png decode should succeed");

        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (image.width(), image.height(), image.bands())
        );
        assert_eq!(decoded.pixels(), image.pixels());
    }

    #[cfg(feature = "webp")]
    #[test]
    fn webp_quality_100_roundtrip_preserves_geometry() {
        let image = patterned_rgba(13, 9);
        let codec = WebpCodec;
        let encoded = codec
            .encode_with_options(&image, &SaveOptions::default().with_quality(100))
            .expect("webp quality=100 encode should succeed");
        let decoded: InMemoryImage<U8> =
            codec.decode(&encoded).expect("webp decode should succeed");

        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (image.width(), image.height(), image.bands())
        );
        assert!(!decoded.pixels().is_empty());
    }
}
