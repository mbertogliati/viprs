mod robustez_idempotencia {
    use std::num::NonZeroUsize;

    use viprs::{
        Image, ImageMetadata, Interpretation, U8,
        adapters::{scheduler::rayon_scheduler::RayonScheduler, sources::memory::MemorySource},
        domain::{
            colorspace::{ColorspaceId, Lab, SRgb},
            kernel::InterpolationKernel,
            ops::resample::{Thumbnail, thumbnail::ThumbnailTarget},
        },
    };

    #[cfg(any(feature = "jpeg", feature = "png"))]
    use viprs::domain::codec_options::SaveOptions;
    #[cfg(any(feature = "jpeg", feature = "png"))]
    use viprs::ports::codec::ImageEncoder;
    use viprs::{adapters::codecs::JpegCodec, domain::codec_options::JpegSubsampling};
    use viprs::{adapters::codecs::PngCodec, domain::codec_options::PngFilterStrategy};
    #[cfg(feature = "jpeg")]
    #[cfg(feature = "png")]

    fn patterned_image(width: u32, height: u32, bands: u32) -> Image<U8> {
        let len = width as usize * height as usize * bands as usize;
        let pixels = (0..len)
            .map(|index| ((index * 37 + (index / bands as usize) * 13 + 17) % 251) as u8)
            .collect();

        Image::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(ImageMetadata {
                interpretation: Some(match bands {
                    1 | 2 => Interpretation::BW,
                    _ => Interpretation::Srgb,
                }),
                ..ImageMetadata::default()
            })
    }

    fn source_from_image(image: &Image<U8>) -> MemorySource<U8> {
        MemorySource::new(
            image.width(),
            image.height(),
            image.bands(),
            image.pixels().to_vec(),
        )
        .unwrap()
        .with_metadata(image.metadata().clone())
    }

    fn execute_pipeline<S: viprs_runtime::pipeline::Flush>(
        image: &Image<U8>,
        threads: usize,
        configure: impl FnOnce(
            viprs_runtime::pipeline::PipelineBuilder,
        )
            -> Result<viprs_runtime::pipeline::PipelineBuilder<S>, viprs::BuildError>,
    ) -> Image<U8> {
        let pipeline = configure(viprs_runtime::pipeline::PipelineBuilder::from_source(
            source_from_image(image),
        ))
        .unwrap()
        .build()
        .unwrap();
        let scheduler = RayonScheduler::new(threads).unwrap();
        pipeline.run_to_image::<U8, _>(&scheduler).unwrap()
    }

    fn assert_all_outputs_identical(outputs: &[Vec<u8>], label: &str) {
        let baseline = &outputs[0];
        for (index, output) in outputs.iter().enumerate().skip(1) {
            assert_eq!(
                baseline,
                output,
                "{label}: output changed on repetition {}",
                index + 1
            );
        }
    }

    #[cfg(feature = "jpeg")]
    fn deterministic_jpeg_options() -> SaveOptions {
        SaveOptions {
            quality: Some(85),
            interlace: Some(false),
            jpeg_subsampling: Some(JpegSubsampling::Off),
            strip_metadata: Some(true),
            ..SaveOptions::default()
        }
    }

    #[cfg(feature = "png")]
    fn deterministic_png_options() -> SaveOptions {
        SaveOptions {
            compression_level: Some(6),
            interlace: Some(false),
            png_filter: Some(PngFilterStrategy::None),
            strip_metadata: Some(true),
            ..SaveOptions::default()
        }
    }

    #[test]
    fn same_pipeline_twice_produces_identical_bytes() {
        let image = patterned_image(257, 193, 3);
        let pipeline =
            viprs_runtime::pipeline::PipelineBuilder::from_source(source_from_image(&image))
                .thumbnail(Thumbnail::new(
                    ThumbnailTarget::Width(91),
                    InterpolationKernel::Lanczos3,
                ))
                .unwrap()
                .invert()
                .unwrap()
                .build()
                .unwrap();
        let scheduler = RayonScheduler::new(2).unwrap();

        let first = pipeline.run_to_image::<U8, _>(&scheduler).unwrap();
        let second = pipeline.run_to_image::<U8, _>(&scheduler).unwrap();

        assert_eq!(first.pixels(), second.pixels());
    }

    #[test]
    fn invert_twice_matches_original_u8_bytes() {
        let image = patterned_image(129, 97, 3);

        let output = execute_pipeline(&image, 2, |builder| builder.invert()?.invert());

        assert_eq!(output.pixels(), image.pixels());
    }

    #[test]
    fn shrink_is_deterministic_across_ten_runs() {
        let image = patterned_image(255, 189, 3);
        let outputs: Vec<Vec<u8>> = (0..10)
            .map(|_| {
                execute_pipeline(&image, 2, |builder| builder.shrink(3, 2))
                    .pixels()
                    .to_vec()
            })
            .collect();

        assert_all_outputs_identical(&outputs, "shrink");
    }

    #[test]
    fn thumbnail_is_deterministic_across_five_runs() {
        let image = patterned_image(321, 197, 3);
        let outputs: Vec<Vec<u8>> = (0..5)
            .map(|_| {
                execute_pipeline(&image, 2, |builder| {
                    builder.thumbnail(Thumbnail::new(
                        ThumbnailTarget::FitBox {
                            width: 80,
                            height: 60,
                        },
                        InterpolationKernel::Lanczos3,
                    ))
                })
                .pixels()
                .to_vec()
            })
            .collect();

        assert_all_outputs_identical(&outputs, "thumbnail");
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn jpeg_encoding_is_deterministic() {
        let image = patterned_image(96, 72, 3);
        let codec = JpegCodec;
        let opts = deterministic_jpeg_options();
        let outputs: Vec<Vec<u8>> = (0..3)
            .map(|_| codec.encode_with_options(&image, &opts).unwrap())
            .collect();

        assert_all_outputs_identical(&outputs, "jpeg encode");
    }

    #[test]
    #[cfg(feature = "png")]
    fn png_encoding_is_deterministic() {
        let image = patterned_image(96, 72, 3);
        let codec = PngCodec::default();
        let opts = deterministic_png_options();
        let outputs: Vec<Vec<u8>> = (0..3)
            .map(|_| codec.encode_with_options(&image, &opts).unwrap())
            .collect();

        assert_all_outputs_identical(&outputs, "png encode");
    }

    #[test]
    fn srgb_lab_srgb_round_trip_is_deterministic() {
        let image = patterned_image(111, 83, 3);
        let outputs: Vec<Vec<u8>> = (0..2)
            .map(|_| {
                execute_pipeline(&image, 2, |builder| {
                    builder
                        .with_colorspace(ColorspaceId::SRgb)
                        .colourspace::<Lab>()?
                        .colourspace::<SRgb>()
                })
                .pixels()
                .to_vec()
            })
            .collect();

        assert_all_outputs_identical(&outputs, "colour round-trip");
    }

    #[test]
    fn reduce_is_deterministic_across_five_runs() {
        let image = patterned_image(250, 175, 3);
        let outputs: Vec<Vec<u8>> = (0..5)
            .map(|_| {
                execute_pipeline(&image, 2, |builder| {
                    builder.reduce(2.5, 2.5, InterpolationKernel::Lanczos3)
                })
                .pixels()
                .to_vec()
            })
            .collect();

        assert_all_outputs_identical(&outputs, "reduce");
    }

    #[test]
    fn cache_enabled_pipeline_matches_uncached_output() {
        let image = patterned_image(257, 193, 3);

        let uncached = execute_pipeline(&image, 2, |builder| {
            builder
                .thumbnail(Thumbnail::new(
                    ThumbnailTarget::Width(91),
                    InterpolationKernel::Lanczos3,
                ))?
                .invert()
        });
        let cached = execute_pipeline(&image, 2, |builder| {
            builder
                .thumbnail(Thumbnail::new(
                    ThumbnailTarget::Width(91),
                    InterpolationKernel::Lanczos3,
                ))?
                .invert()?
                .cache_last_op(NonZeroUsize::new(1 << 20).unwrap())
        });

        assert_eq!(uncached.pixels(), cached.pixels());
    }

    #[test]
    fn output_is_independent_of_thread_count() {
        let image = patterned_image(257, 193, 3);

        let single_thread = execute_pipeline(&image, 1, |builder| {
            builder
                .thumbnail(Thumbnail::new(
                    ThumbnailTarget::Width(91),
                    InterpolationKernel::Lanczos3,
                ))?
                .reduce(1.5, 1.5, InterpolationKernel::Lanczos3)?
                .invert()
        });
        let four_threads = execute_pipeline(&image, 4, |builder| {
            builder
                .thumbnail(Thumbnail::new(
                    ThumbnailTarget::Width(91),
                    InterpolationKernel::Lanczos3,
                ))?
                .reduce(1.5, 1.5, InterpolationKernel::Lanczos3)?
                .invert()
        });

        assert_eq!(single_thread.pixels(), four_threads.pixels());
    }
}
