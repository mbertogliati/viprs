use super::flows_support::*;
use proptest::proptest;

#[test]
#[cfg(feature = "jpeg")]
fn resize_then_sharpen_handles_odd_dimensions() {
    let image = load_u8_fixture("bench_777x333.jpg");
    let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
        builder
            .resize(Resize::new(0.5, 0.5, InterpolationKernel::Lanczos3))?
            .sharpen(
                SHARPEN_SIGMA,
                SHARPEN_X1,
                SHARPEN_Y2,
                SHARPEN_Y3,
                SHARPEN_M1,
                SHARPEN_M2,
            )
    });

    assert_resize_dimensions(&pipeline, &image, 0.5);
}

#[test]
#[cfg(feature = "jpeg")]
fn invert_then_thumbnail_handles_odd_dimensions() {
    let image = load_u8_fixture("bench_777x333.jpg");
    let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
        builder.invert()?.thumbnail(thumbnail_config(200))
    });

    assert_thumbnail_dimensions(&pipeline, &image, 200);
}

#[test]
#[cfg(feature = "jpeg")]
fn load_jpeg_then_thumbnail_then_encode_jpeg_handles_odd_dimensions() {
    let image = load_u8_fixture("bench_777x333.jpg");
    let (pipeline, output) =
        execute_u8_pipeline_to_image(&image, |builder| builder.thumbnail(thumbnail_config(200)));

    assert_thumbnail_dimensions(&pipeline, &image, 200);
    assert_codec_roundtrip(&JpegCodec, &output, pipeline.width, pipeline.height);
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 4,
        .. ProptestConfig::default()
    })]

    #[test]
    fn thumbnail_then_sharpen_never_panics_for_arbitrary_dimensions(
        width in 1_u32..4096,
        height in 1_u32..4096,
        target_width in 1_u32..1024,
    ) {
        let pixel_count = width as usize * height as usize * 3;
        let pixels: Vec<u8> = (0..pixel_count)
            .map(|index| (index % 251) as u8)
            .collect();

        let image = Image::<U8>::from_buffer(width, height, 3, pixels)
            .expect("failed to create proptest image")
            .with_metadata(ImageMetadata {
                interpretation: Some(Interpretation::Srgb),
                ..ImageMetadata::default()
            });

        let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
            builder
                .thumbnail(thumbnail_config(target_width))?
                .sharpen(
                    SHARPEN_SIGMA,
                    SHARPEN_X1,
                    SHARPEN_Y2,
                    SHARPEN_Y3,
                    SHARPEN_M1,
                    SHARPEN_M2,
                )
        });

        prop_assert!(pipeline.width > 0);
        prop_assert!(pipeline.height > 0);
    }
}
