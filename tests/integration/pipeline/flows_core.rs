use super::flows_support::*;

#[test]
#[cfg(feature = "jpeg")]
fn thumbnail_then_sharpen_handles_standard_cdn_sizes() {
    for fixture in [
        "bench_512x512.jpg",
        "bench_2048x2048.jpg",
        "bench_8192x8192.jpg",
    ] {
        let image = load_u8_fixture(fixture);
        for width in [100, 200, 400, 800] {
            let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
                builder
                    .plan_thumbnail(thumbnail_config(width))?
                    .plan_sharpen(
                        SHARPEN_SIGMA,
                        SHARPEN_X1,
                        SHARPEN_Y2,
                        SHARPEN_Y3,
                        SHARPEN_M1,
                        SHARPEN_M2,
                    )
            });
            assert_thumbnail_dimensions(&pipeline, &image, width);
        }
    }
}

#[test]
#[cfg(feature = "jpeg")]
fn thumbnail_then_gauss_blur_handles_requested_sigmas() {
    let image = load_u8_fixture("bench_2048x2048.jpg");
    for width in [400, 800] {
        for sigma in [0.5_f32, 2.0_f32] {
            let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
                builder
                    .plan_thumbnail(thumbnail_config(width))?
                    .plan_gauss_blur(sigma)
            });
            assert_thumbnail_dimensions(&pipeline, &image, width);
        }
    }
}

#[test]
#[cfg(feature = "png")]
fn thumbnail_then_gauss_blur_then_encode_png_executes_end_to_end() {
    let image = load_u8_fixture("bench_2048x2048.png");
    let (pipeline, buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
        builder
            .plan_thumbnail(thumbnail_config(400))?
            .plan_gauss_blur(2.0)
    });

    assert_thumbnail_dimensions(&pipeline, &image, 400);
    let output = output_image_from_buffer(&image, &pipeline, buffer);
    assert_codec_roundtrip(
        &PngCodec::default(),
        &output,
        pipeline.width,
        pipeline.height,
    );
}

#[test]
#[cfg(feature = "jpeg")]
fn thumbnail_then_encode_roundtrips_across_output_codecs() {
    let image = load_u8_fixture("bench_2048x2048.jpg");
    let (pipeline, output) = execute_u8_pipeline_to_image(&image, |builder| {
        builder.plan_thumbnail(thumbnail_config(400))
    });

    assert_thumbnail_dimensions(&pipeline, &image, 400);
    #[cfg(not(any(feature = "jpeg", feature = "webp", feature = "png", feature = "avif")))]
    let _ = &output;

    #[cfg(feature = "jpeg")]
    assert_codec_roundtrip(&JpegCodec, &output, pipeline.width, pipeline.height);
    #[cfg(feature = "webp")]
    assert_codec_roundtrip(&WebpCodec, &output, pipeline.width, pipeline.height);
    #[cfg(feature = "png")]
    assert_codec_roundtrip(
        &PngCodec::default(),
        &output,
        pipeline.width,
        pipeline.height,
    );
    #[cfg(feature = "avif")]
    assert_codec_roundtrip(&AvifCodec, &output, pipeline.width, pipeline.height);
}

#[test]
#[cfg(feature = "jpeg")]
fn resize_then_sharpen_handles_half_and_quarter_scale() {
    let image = load_u8_fixture("bench_2048x2048.jpg");
    for scale in [0.5_f64, 0.25_f64] {
        let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
            builder
                .plan_resize(Resize::new(scale, scale, InterpolationKernel::Lanczos3))?
                .plan_sharpen(
                    SHARPEN_SIGMA,
                    SHARPEN_X1,
                    SHARPEN_Y2,
                    SHARPEN_Y3,
                    SHARPEN_M1,
                    SHARPEN_M2,
                )
        });
        assert_resize_dimensions(&pipeline, &image, scale);
    }
}

#[test]
#[cfg(feature = "jpeg")]
fn resize_then_colourspace_lab_then_srgb_executes_end_to_end() {
    let image = load_u8_fixture("bench_2048x2048.jpg");
    let (pipeline, _buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
        builder
            .plan_resize(Resize::new(0.5, 0.5, InterpolationKernel::Lanczos3))?
            .plan_colourspace::<Lab>()?
            .plan_colourspace::<SRgb>()
    });

    assert_resize_dimensions(&pipeline, &image, 0.5);
}

#[test]
#[cfg(feature = "jpeg")]
fn thumbnail_then_linear_then_encode_jpeg_executes_end_to_end() {
    let image = load_u8_fixture("bench_2048x2048.jpg");
    let (pipeline, buffer) = execute_u8_pipeline_to_buffer(&image, |builder| {
        builder
            .plan_thumbnail(thumbnail_config(200))?
            .plan_linear(1.2, 0.0)
    });

    assert_thumbnail_dimensions(&pipeline, &image, 200);
    let output = output_image_from_buffer(&image, &pipeline, buffer);
    assert_codec_roundtrip(&JpegCodec, &output, pipeline.width, pipeline.height);
}
