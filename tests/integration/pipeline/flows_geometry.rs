use super::flows_support::*;

#[test]
#[cfg(feature = "jpeg")]
fn embed_then_extract_area_executes_end_to_end() {
    use viprs::domain::ops::conversion::ExtendMode;

    let image = load_u8_fixture("bench_512x512.jpg");
    let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
        builder
            .embed(
                800,
                600,
                0,
                0,
                image.width(),
                image.height(),
                ExtendMode::Black,
            )?
            .extract_area(100, 100, 400, 300)
    });

    assert_eq!(
        (pipeline.width, pipeline.height),
        (400, 300),
        "embed -> extract_area must produce the extracted dimensions"
    );
}

#[test]
#[cfg(feature = "jpeg")]
fn thumbnail_then_rotate90_then_flip_horizontal_executes_end_to_end() {
    let image = load_u8_fixture("bench_777x333.jpg");
    let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
        builder
            .thumbnail_with(thumbnail_config(400))?
            .rotate90()?
            .flip_horizontal()
    });

    let (thumb_width, thumb_height) = expected_thumbnail_dimensions(&image, 400);
    assert_eq!(
        (pipeline.width, pipeline.height),
        (thumb_height, thumb_width),
        "rotate90 must transpose thumbnail output dimensions before horizontal flip"
    );
}

#[test]
#[cfg(feature = "jpeg")]
fn thumbnail_then_sharpen_then_gauss_blur_then_thumbnail_executes_end_to_end() {
    let image = load_u8_fixture("bench_2048x2048.jpg");
    let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
        builder
            .thumbnail_with(thumbnail_config(400))?
            .sharpen_with(
                SHARPEN_SIGMA,
                SHARPEN_X1,
                SHARPEN_Y2,
                SHARPEN_Y3,
                SHARPEN_M1,
                SHARPEN_M2,
            )?
            .gauss_blur(1.0)?
            .thumbnail_with(thumbnail_config(200))
    });

    assert_eq!(
        (pipeline.width, pipeline.height),
        (200, 200),
        "second thumbnail must reduce the already sharpened/blurred image to 200px width"
    );
}

#[test]
#[cfg(feature = "jpeg")]
fn thumbnail_then_colourspace_lab_then_srgb_keeps_geometry() {
    let image = load_u8_fixture("bench_2048x2048.jpg");
    let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
        builder
            .thumbnail_with(thumbnail_config(400))?
            .colourspace::<Lab>()?
            .colourspace::<SRgb>()
    });

    assert_thumbnail_dimensions(&pipeline, &image, 400);
}

#[test]
#[cfg(feature = "jpeg")]
fn invert_then_thumbnail_keeps_geometry() {
    let image = load_u8_fixture("bench_2048x2048.jpg");
    let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
        builder.invert()?.thumbnail_with(thumbnail_config(400))
    });

    assert_thumbnail_dimensions(&pipeline, &image, 400);
}

#[test]
#[cfg(any(feature = "jpeg", feature = "png", feature = "webp", feature = "tiff"))]
fn load_format_then_thumbnail_then_encode_roundtrips() {
    #[cfg(feature = "jpeg")]
    {
        let image = load_u8_fixture("bench_2048x2048.jpg");
        let (pipeline, output) = execute_u8_pipeline_to_image(&image, |builder| {
            builder.thumbnail_with(thumbnail_config(400))
        });
        assert_thumbnail_dimensions(&pipeline, &image, 400);
        assert_codec_roundtrip(&JpegCodec, &output, pipeline.width, pipeline.height);
    }

    #[cfg(feature = "png")]
    {
        let image = load_u8_fixture("bench_2048x2048.png");
        let (pipeline, output) = execute_u8_pipeline_to_image(&image, |builder| {
            builder.thumbnail_with(thumbnail_config(400))
        });
        assert_thumbnail_dimensions(&pipeline, &image, 400);
        assert_codec_roundtrip(
            &PngCodec::default(),
            &output,
            pipeline.width,
            pipeline.height,
        );
    }

    #[cfg(feature = "webp")]
    {
        let image = load_u8_fixture("bench_2048x2048.webp");
        let (pipeline, output) = execute_u8_pipeline_to_image(&image, |builder| {
            builder.thumbnail_with(thumbnail_config(400))
        });
        assert_thumbnail_dimensions(&pipeline, &image, 400);
        assert_codec_roundtrip(&WebpCodec, &output, pipeline.width, pipeline.height);
    }

    #[cfg(feature = "tiff")]
    {
        let image = load_u8_fixture("bench_2048x2048.tif");
        let (pipeline, output) = execute_u8_pipeline_to_image(&image, |builder| {
            builder.thumbnail_with(thumbnail_config(400))
        });
        assert_thumbnail_dimensions(&pipeline, &image, 400);
        assert_codec_roundtrip(
            &TiffCodec::default(),
            &output,
            pipeline.width,
            pipeline.height,
        );
    }
}

#[test]
#[cfg(feature = "jpeg")]
fn thumbnail_then_sharpen_handles_odd_dimensions() {
    let image = load_u8_fixture("bench_777x333.jpg");
    let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
        builder.thumbnail_with(thumbnail_config(400))?.sharpen_with(
            SHARPEN_SIGMA,
            SHARPEN_X1,
            SHARPEN_Y2,
            SHARPEN_Y3,
            SHARPEN_M1,
            SHARPEN_M2,
        )
    });

    assert_thumbnail_dimensions(&pipeline, &image, 400);
}
