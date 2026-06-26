mod robustness_recovery {
    use std::path::Path;

    use viprs::{
        DemandHint, Image, Region, U8, ViprsError,
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::memory::MemorySource,
        },
        ports::source::ImageSource,
    };

    const FIXTURE_ROW_WIDTH: u32 = 64;
    const FAILURE_MESSAGE: &str = "synthetic source read failure";

    struct FailingSource {
        width: u32,
        height: u32,
        bands: u32,
    }

    impl FailingSource {
        fn from_image(image: &Image<U8>) -> Self {
            Self {
                width: image.width(),
                height: image.height(),
                bands: image.bands(),
            }
        }
    }

    impl ImageSource for FailingSource {
        type Format = U8;

        fn width(&self) -> u32 {
            self.width
        }

        fn height(&self) -> u32 {
            self.height
        }

        fn bands(&self) -> u32 {
            self.bands
        }

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn read_region(&self, _region: Region, _output: &mut [u8]) -> Result<(), ViprsError> {
            Err(ViprsError::Codec(FAILURE_MESSAGE.into()))
        }
    }

    fn fixture_image_from_buffer(name: &str) -> Image<U8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/images")
            .join(name);
        let mut bytes = std::fs::read(&path)
            .unwrap_or_else(|err| panic!("failed to read fixture {}: {err}", path.display()));
        assert!(
            !bytes.is_empty(),
            "fixture {} must contain bytes for the recovery test",
            path.display()
        );

        let width = FIXTURE_ROW_WIDTH.min(bytes.len() as u32).max(1);
        let height = (bytes.len() as u32 / width).max(1);
        bytes.truncate(width as usize * height as usize);

        Image::from_buffer(width, height, 1, bytes)
            .unwrap_or_else(|err| panic!("failed to rebuild image from {}: {err}", path.display()))
    }

    fn expected_invert_pixels(image: &Image<U8>) -> Vec<u8> {
        image.pixels().iter().map(|pixel| 255u8 - *pixel).collect()
    }

    fn build_valid_pipeline(image: &Image<U8>) -> viprs_runtime::pipeline::CompiledPipeline {
        let source = MemorySource::<U8>::new(
            image.width(),
            image.height(),
            image.bands(),
            image.pixels().to_vec(),
        )
        .expect("fixture image should have a valid in-memory buffer");

        viprs_runtime::pipeline::PipelineBuilder::from_source(source)
            .invert()
            .expect("invert should build for U8 fixture images")
            .build()
            .expect("valid recovery pipeline should compile")
    }

    fn build_failing_pipeline(image: &Image<U8>) -> viprs_runtime::pipeline::CompiledPipeline {
        let source = FailingSource::from_image(image);

        viprs_runtime::pipeline::PipelineBuilder::from_source(source)
            .invert()
            .expect("invert should build for failing U8 sources")
            .build()
            .expect("failing recovery pipeline should compile")
    }

    #[test]
    fn pipeline_recovers_from_repeated_execution_errors() {
        let scheduler = RayonScheduler::new(4).expect("rayon scheduler should initialize");
        let image = fixture_image_from_buffer("sample.png");
        let expected = expected_invert_pixels(&image);

        for _ in 0..10 {
            let failing_pipeline = build_failing_pipeline(&image);
            let result = failing_pipeline.run_to_image::<U8, _>(&scheduler);
            assert!(
                matches!(&result, Err(ViprsError::Codec(message)) if message == FAILURE_MESSAGE),
                "expected source read failure to propagate cleanly, got {result:?}"
            );

            let valid_pipeline = build_valid_pipeline(&image);
            let recovered = valid_pipeline
                .run_to_image::<U8, _>(&scheduler)
                .expect("scheduler should recover after a failed pipeline");

            assert_eq!(recovered.width(), image.width());
            assert_eq!(recovered.height(), image.height());
            assert_eq!(recovered.bands(), image.bands());
            assert_eq!(recovered.pixels(), expected.as_slice());
        }
    }

    #[test]
    fn rayon_thread_pool_remains_usable_after_errors() {
        let scheduler = RayonScheduler::new(4).expect("rayon scheduler should initialize");
        let image = fixture_image_from_buffer("bench_512x512.png");
        assert!(
            image.height() > 16,
            "fixture-derived image should span multiple scheduler tiles"
        );
        let expected = expected_invert_pixels(&image);

        for _ in 0..10 {
            let failing_pipeline = build_failing_pipeline(&image);
            assert!(
                matches!(
                    failing_pipeline.run_to_image::<U8, _>(&scheduler),
                    Err(ViprsError::Codec(message)) if message == FAILURE_MESSAGE
                ),
                "expected repeated source failures before the recovery run"
            );
        }

        let valid_pipeline = build_valid_pipeline(&image);
        let mut sink = MemorySink::for_pipeline(&valid_pipeline).unwrap();
        let profile = scheduler
            .run_with_profile(&valid_pipeline, &mut sink)
            .expect("thread pool should keep running a multi-tile pipeline after errors");
        let recovered = sink
            .into_image::<U8>(
                valid_pipeline.width,
                valid_pipeline.height,
                valid_pipeline.output_bands,
                valid_pipeline.source.metadata(),
            )
            .expect("memory sink should rebuild the output image");

        assert!(
            profile.tile_count > 1,
            "expected a multi-tile run, got {profile:?}"
        );
        assert!(
            profile.total_ns > 0,
            "expected runtime metrics for the recovery run"
        );
        assert_eq!(recovered.pixels(), expected.as_slice());
    }
}
