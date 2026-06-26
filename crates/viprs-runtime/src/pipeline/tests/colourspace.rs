use super::*;

#[test]
fn colourspace_builder_supports_lab_lch_lab() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<F32>::new(1, 1, 3, vec![50.0, 20.0, -30.0]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::Lab)
        .colourspace::<Lch>()
        .unwrap()
        .colourspace::<Lab>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_format, BandFormatId::F32);
    assert_eq!(pipeline.output_bands, 3);
}

#[test]
fn colourspace_builder_supports_srgb_scrgb_and_back() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<U8>::new(1, 1, 3, vec![128, 64, 32]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .colourspace::<ScRgb>()
        .unwrap()
        .colourspace::<SRgb>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_format, BandFormatId::U8);
    assert_eq!(pipeline.output_bands, 3);
}

#[test]
fn colourspace_builder_supports_lab_xyz_and_back() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<F32>::new(1, 1, 3, vec![53.23, 80.1, 67.2]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::Lab)
        .colourspace::<Xyz>()
        .unwrap()
        .colourspace::<Lab>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_format, BandFormatId::F32);
    assert_eq!(pipeline.output_bands, 3);
}

#[test]
fn colourspace_builder_supports_scrgb_xyz_and_back() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<F32>::new(1, 1, 3, vec![0.4, 0.5, 0.6]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::ScRgb)
        .colourspace::<Xyz>()
        .unwrap()
        .colourspace::<ScRgb>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_format, BandFormatId::F32);
    assert_eq!(pipeline.output_bands, 3);
}

#[test]
fn colourspace_builder_supports_srgb_to_bw() {
    use crate::adapters::{sinks::memory::MemorySink, sources::memory::MemorySource};
    use crate::domain::colorspace::Greyscale;

    let mut metadata = ImageMetadata::default();
    metadata.interpretation = Some(Interpretation::Srgb);

    let direct = ImagePipeline::from_source(
        MemorySource::<U8>::new(1, 1, 3, vec![128, 64, 32])
            .unwrap()
            .with_metadata(metadata.clone()),
    )
    .colourspace::<Greyscale>()
    .unwrap()
    .build()
    .unwrap();
    let explicit = ImagePipeline::from_source(
        MemorySource::<U8>::new(1, 1, 3, vec![128, 64, 32])
            .unwrap()
            .with_metadata(metadata),
    )
    .colourspace::<ScRgb>()
    .unwrap()
    .colourspace::<Greyscale>()
    .unwrap()
    .build()
    .unwrap();

    let mut direct_sink = MemorySink::for_pipeline(&direct).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&direct, &mut direct_sink)
        .unwrap();
    let direct_out = bytemuck::cast_slice::<u8, f32>(&direct_sink.into_buffer()).to_vec();

    let mut explicit_sink = MemorySink::for_pipeline(&explicit).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&explicit, &mut explicit_sink)
        .unwrap();
    let explicit_out = bytemuck::cast_slice::<u8, f32>(&explicit_sink.into_buffer()).to_vec();

    assert_eq!(direct.output_bands, 1);
    assert_eq!(direct.output_format, BandFormatId::F32);
    assert_eq!(direct_out.len(), 1);
    assert!((direct_out[0] - explicit_out[0]).abs() < 1e-6);
}

#[test]
fn colourspace_builder_supports_scrgb_to_bw() {
    use crate::adapters::{sinks::memory::MemorySink, sources::memory::MemorySource};
    use crate::domain::colorspace::Greyscale;

    let mut metadata = ImageMetadata::default();
    metadata.interpretation = Some(Interpretation::Scrgb);

    let pipeline = ImagePipeline::from_source(
        MemorySource::<F32>::new(1, 1, 3, vec![0.25, 0.5, 0.75])
            .unwrap()
            .with_metadata(metadata),
    )
    .colourspace::<Greyscale>()
    .unwrap()
    .build()
    .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    let output = bytemuck::cast_slice::<u8, f32>(&sink.into_buffer()).to_vec();

    assert_eq!(pipeline.output_bands, 1);
    assert_eq!(pipeline.output_format, BandFormatId::F32);
    assert_eq!(output.len(), 1);
    assert!((output[0] - 0.4649).abs() < 1e-6);
}

#[test]
fn colourspace_builder_supports_lab_to_bw() {
    use crate::adapters::{sinks::memory::MemorySink, sources::memory::MemorySource};
    use crate::domain::colorspace::Greyscale;

    let mut metadata = ImageMetadata::default();
    metadata.interpretation = Some(Interpretation::Lab);

    let direct = ImagePipeline::from_source(
        MemorySource::<F32>::new(1, 1, 3, vec![53.23, 80.1, 67.2])
            .unwrap()
            .with_metadata(metadata.clone()),
    )
    .colourspace::<Greyscale>()
    .unwrap()
    .build()
    .unwrap();
    let explicit = ImagePipeline::from_source(
        MemorySource::<F32>::new(1, 1, 3, vec![53.23, 80.1, 67.2])
            .unwrap()
            .with_metadata(metadata),
    )
    .colourspace::<SRgb>()
    .unwrap()
    .colourspace::<ScRgb>()
    .unwrap()
    .colourspace::<Greyscale>()
    .unwrap()
    .build()
    .unwrap();

    let mut direct_sink = MemorySink::for_pipeline(&direct).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&direct, &mut direct_sink)
        .unwrap();
    let direct_out = bytemuck::cast_slice::<u8, f32>(&direct_sink.into_buffer()).to_vec();

    let mut explicit_sink = MemorySink::for_pipeline(&explicit).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&explicit, &mut explicit_sink)
        .unwrap();
    let explicit_out = bytemuck::cast_slice::<u8, f32>(&explicit_sink.into_buffer()).to_vec();

    assert_eq!(direct.output_bands, 1);
    assert_eq!(direct.output_format, BandFormatId::F32);
    assert_eq!(direct_out.len(), 1);
    assert!((direct_out[0] - explicit_out[0]).abs() < 1e-6);
}

#[test]
fn colourspace_builder_supports_srgb_cmyk() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<U8>::new(1, 1, 3, vec![128, 64, 32]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .colourspace::<Cmyk>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_format, BandFormatId::U8);
    assert_eq!(pipeline.output_bands, 4);
}

#[test]
fn colourspace_builder_supports_cmyk_xyz_and_back() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<U8>::new(1, 1, 4, vec![0, 255, 255, 0]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::Cmyk)
        .colourspace::<Xyz>()
        .unwrap()
        .colourspace::<Cmyk>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_format, BandFormatId::U8);
    assert_eq!(pipeline.output_bands, 4);
}

#[test]
fn colourspace_builder_supports_srgb_cmyk_and_back() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<U8>::new(1, 1, 3, vec![128, 64, 32]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .colourspace::<Cmyk>()
        .unwrap()
        .colourspace::<SRgb>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_format, BandFormatId::U8);
    assert_eq!(pipeline.output_bands, 3);
}

#[test]
fn colourspace_builder_supports_xyz_oklab_oklch_and_back() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<F32>::new(1, 1, 3, vec![0.95047, 1.0, 1.08883]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::Xyz)
        .colourspace::<Oklab>()
        .unwrap()
        .colourspace::<Oklch>()
        .unwrap()
        .colourspace::<Oklab>()
        .unwrap()
        .colourspace::<Xyz>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_format, BandFormatId::F32);
    assert_eq!(pipeline.output_bands, 3);
}

#[test]
fn colourspace_builder_supports_xyz_yxy_xyz() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<F32>::new(1, 1, 3, vec![0.4, 0.5, 0.6]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::Xyz)
        .colourspace::<Yxy>()
        .unwrap()
        .colourspace::<Xyz>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_format, BandFormatId::F32);
    assert_eq!(pipeline.output_bands, 3);
}

#[test]
fn colourspace_builder_supports_lch_ucs_and_back() {
    use crate::sources::memory::MemorySource;

    let source = MemorySource::<F32>::new(1, 1, 3, vec![50.0, 20.0, 120.0]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::Lch)
        .colourspace::<Ucs>()
        .unwrap()
        .colourspace::<Lch>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_format, BandFormatId::F32);
    assert_eq!(pipeline.output_bands, 3);
}

#[test]
fn colourspace_builder_maps_cmc_interpretation_to_lch_route() {
    use crate::sources::memory::MemorySource;

    let mut metadata = ImageMetadata::default();
    metadata.interpretation = Some(Interpretation::Cmc);

    let source = MemorySource::<F32>::new(1, 1, 3, vec![50.0, 20.0, 120.0])
        .unwrap()
        .with_metadata(metadata);
    let pipeline = ImagePipeline::from_source(source)
        .colourspace::<Lch>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_format, BandFormatId::F32);
    assert_eq!(pipeline.output_bands, 3);
}

#[test]
fn colourspace_builder_preserves_known_white_point_through_xyz_lab_xyz() {
    use crate::adapters::{sinks::memory::MemorySink, sources::memory::MemorySource};

    let source = MemorySource::<F32>::new(1, 1, 3, vec![0.95047, 1.0, 1.08883]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::Xyz)
        .colourspace::<Lab>()
        .unwrap()
        .colourspace::<Xyz>()
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    let output = bytemuck::cast_slice::<u8, f32>(&sink.into_buffer()).to_vec();

    assert!((output[0] - 0.95047).abs() < 5e-4);
    assert!((output[1] - 1.0).abs() < 5e-4);
    assert!((output[2] - 1.08883).abs() < 5e-4);
}

proptest! {
    #[test]
    fn colourspace_builder_preserves_rgba_alpha_through_srgb_lab_srgb(
        red in any::<u8>(),
        green in any::<u8>(),
        blue in any::<u8>(),
        alpha in any::<u8>(),
    ) {
        use crate::adapters::{sinks::memory::MemorySink, sources::memory::MemorySource};

        let source = MemorySource::<U8>::new(1, 1, 4, vec![red, green, blue, alpha]).unwrap();
        let pipeline = ImagePipeline::from_source(source)
            .with_colorspace(ColorspaceId::SRgb)
            .colourspace::<Lab>()
            .unwrap()
            .colourspace::<SRgb>()
            .unwrap()
            .build()
            .unwrap();

        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(1)
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();
        let output = sink.into_buffer();

        prop_assert!((i16::from(output[0]) - i16::from(red)).abs() <= 2);
        prop_assert!((i16::from(output[1]) - i16::from(green)).abs() <= 2);
        prop_assert!((i16::from(output[2]) - i16::from(blue)).abs() <= 2);
        prop_assert_eq!(output[3], alpha);
    }
}

fn patterned_rgba_u8(width: u32, height: u32) -> InMemoryImage<U8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 17 + y * 13 + 3) % 256) as u8);
            pixels.push(((x * 11 + y * 29 + 7) % 256) as u8);
            pixels.push(((x * 5 + y * 19 + 191) % 256) as u8);
            pixels.push(((x * 23 + y * 31 + 255) % 256) as u8);
        }
    }

    InMemoryImage::from_buffer(width, height, 4, pixels)
        .unwrap()
        .with_metadata(ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        })
}

#[test]
fn rgba_u16_thumbnail_builds_and_executes() {
    use crate::domain::{format::U16, ops::resample::thumbnail::ThumbnailTarget};

    let width = 23u32;
    let height = 17u32;
    let bands = 4u32;
    let mut pixels = Vec::with_capacity((width * height * bands) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 257 + y * 1021 + 17) % u16::MAX as u32) as u16);
            pixels.push(((x * 911 + y * 263 + 29) % u16::MAX as u32) as u16);
            pixels.push(((x * 613 + y * 479 + 43) % u16::MAX as u32) as u16);
            pixels.push(((x * 4093 + y * 2081 + 59) % u16::MAX as u32) as u16);
        }
    }

    let image = InMemoryImage::<U16>::from_buffer(width, height, bands, pixels).unwrap();
    let source = MemorySource::<U16>::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .unwrap()
    .with_metadata(image.metadata().clone());

    let pipeline = ImagePipeline::from_source(source)
        .thumbnail_with(Thumbnail::new(
            ThumbnailTarget::Width(11),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .build()
        .unwrap();

    let output = pipeline
        .run_to_image::<U16, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!((output.width(), output.height()), (11, 8));
    assert_eq!(output.bands(), 4);
}

#[test]
fn colourspace_roundtrip_after_affine_and_before_thumbnail_preserves_rgba_alpha_band() {
    let image = patterned_rgba_u8(37, 19);
    let source = MemorySource::<U8>::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .unwrap()
    .with_metadata(image.metadata().clone());
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .affine(
            [37.0 / 29.0, 0.0, 0.0, 19.0 / 13.0],
            0.0,
            0.0,
            29,
            13,
            InterpolationKernel::Lanczos3,
        )
        .unwrap()
        .colourspace::<Lab>()
        .unwrap()
        .colourspace::<SRgb>()
        .unwrap()
        .thumbnail_with(Thumbnail::new(
            ThumbnailTarget::Width(11),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .build()
        .unwrap();

    let output = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!(pipeline.output_bands, 4);
    assert_eq!(output.bands(), 4);
}

#[test]
fn colourspace_roundtrip_after_thumbnail_and_affine_preserves_rgba_alpha_band() {
    let image = patterned_rgba_u8(41, 23);
    let source = MemorySource::<U8>::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .unwrap()
    .with_metadata(image.metadata().clone());
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .thumbnail_with(Thumbnail::new(
            ThumbnailTarget::Width(17),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .affine(
            [17.0 / 13.0, 0.0, 0.0, 10.0 / 7.0],
            0.0,
            0.0,
            13,
            7,
            InterpolationKernel::Lanczos3,
        )
        .unwrap()
        .colourspace::<crate::domain::colorspace::Hsv>()
        .unwrap()
        .colourspace::<SRgb>()
        .unwrap()
        .build()
        .unwrap();

    let output = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!(pipeline.output_bands, 4);
    assert_eq!(output.bands(), 4);
}

#[test]
fn sharpen_builder_preserves_rgba_identity_when_sigma_zero() {
    use crate::adapters::{sinks::memory::MemorySink, sources::memory::MemorySource};

    let source = MemorySource::<U8>::new(1, 1, 4, vec![128, 64, 32, 200]).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .sharpen_with(0.0, 2.0, 10.0, 20.0, 0.0, 3.0)
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    let output = sink.into_buffer();

    assert!((i16::from(output[0]) - 128).abs() <= 2);
    assert!((i16::from(output[1]) - 64).abs() <= 2);
    assert!((i16::from(output[2]) - 32).abs() <= 2);
    assert_eq!(output[3], 200);
}
