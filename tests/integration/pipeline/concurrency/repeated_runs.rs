mod robustness_determinism {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use viprs::{
      BuildError, CompiledPipeline, InMemoryImage, Interpretation, U8,
      adapters::{
          pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
          sinks::memory::MemorySink, sources::memory::MemorySource,
        },
      domain::{
            kernel::InterpolationKernel,
            ops::resample::{Thumbnail, thumbnail::ThumbnailTarget},
        },
      ports::scheduler::TileScheduler,
    };

    fn project_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_owned()
    }

    fn fixture_path(name: &str) -> PathBuf {
        project_root()
            .join("tests")
            .join("fixtures")
            .join("images")
            .join(name)
    }

    fn load_fixture_image(name: &str, width: u32, height: u32, bands: u32) -> InMemoryImage<U8> {
        let path = fixture_path(name);
        let bytes = fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read fixture {}: {error}", path.display()));
        let expected_len = width as usize * height as usize * bands as usize;
        let raw_pixels = if bytes.len() >= expected_len {
            bytes[..expected_len].to_vec()
        } else {
            bytes.iter().copied().cycle().take(expected_len).collect()
        };

        let mut image =
            InMemoryImage::from_buffer(width, height, bands, raw_pixels).unwrap_or_else(|error| {
                panic!(
                    "failed to build fixture-backed image {}: {error}",
                    path.display()
                )
            });
        if bands >= 3 {
            image = image.with_metadata(viprs::ImageMetadata {
                interpretation: Some(Interpretation::Srgb),
                ..viprs::ImageMetadata::default()
            });
        }

        image
    }

    fn memory_source_from_image(image: &InMemoryImage<U8>) -> MemorySource<U8> {
        MemorySource::new(
            image.width(),
            image.height(),
            image.bands(),
            image.pixels().to_vec(),
        )
        .unwrap_or_else(|error| panic!("failed to create memory source: {error}"))
        .with_metadata(image.metadata().clone())
    }

    fn execute_to_buffer<S: viprs::pipeline::Commit>(
      image: &InMemoryImage<U8>,
      threads: usize,
      configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
    ) -> (CompiledPipeline, Vec<u8>) {
        let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(
            image,
        )))
        .unwrap_or_else(|error| panic!("pipeline stage failed: {error:?}"))
        .build()
        .unwrap_or_else(|error| panic!("pipeline build failed: {error:?}"));

        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(threads)
            .unwrap_or_else(|error| panic!("scheduler construction failed: {error}"))
            .run(&pipeline, &mut sink)
            .unwrap_or_else(|error| panic!("pipeline execution failed: {error}"));

        (pipeline, sink.into_buffer())
    }

    fn thumbnail(width: u32) -> Thumbnail {
        Thumbnail::new(ThumbnailTarget::Width(width), InterpolationKernel::Lanczos3)
    }

    #[test]
    fn invert_pipeline_is_deterministic_across_repeated_runs() {
        let image = load_fixture_image("sample.jpg", 64, 64, 3);

        let (_first_pipeline, first) = execute_to_buffer(&image, 2, |builder| builder.invert());
        let (_second_pipeline, second) = execute_to_buffer(&image, 2, |builder| builder.invert());

        assert_eq!(
            first, second,
            "invert output must stay byte-identical across runs"
        );
    }

    #[test]
    fn thumbnail_pipeline_is_deterministic_across_five_runs() {
        let image = load_fixture_image("bench_777x333.jpg", 192, 192, 3);
        let mut baseline: Option<Vec<u8>> = None;

        for run in 0..5 {
            let (_pipeline, output) =
                execute_to_buffer(&image, 2, |builder| builder.thumbnail(thumbnail(200)));
            if let Some(expected) = &baseline {
                assert_eq!(
                    output.as_slice(),
                    expected.as_slice(),
                    "thumbnail output changed on run {}",
                    run + 1
                );
            } else {
                baseline = Some(output);
            }
        }
    }

    #[test]
    fn double_invert_matches_original_bytes() {
        let image = load_fixture_image("sample.jpg", 64, 64, 3);

        let (_pipeline, output) =
            execute_to_buffer(&image, 2, |builder| builder.invert()?.invert());

        assert_eq!(
            output.as_slice(),
            image.pixels(),
            "invert(invert(image)) must match the original bytes"
        );
    }

    #[test]
    fn pipeline_output_is_identical_with_one_or_four_threads() {
        let image = load_fixture_image("bench_1024x1024.jpg", 256, 256, 3);

        let (_single_pipeline, single_thread) = execute_to_buffer(&image, 1, |builder| {
            builder.thumbnail(thumbnail(256))?.invert()
        });
        let (_multi_pipeline, four_threads) = execute_to_buffer(&image, 4, |builder| {
            builder.thumbnail(thumbnail(256))?.invert()
        });

        assert_eq!(
            single_thread, four_threads,
            "pipeline output must be byte-identical regardless of scheduler thread count"
        );
    }
}
