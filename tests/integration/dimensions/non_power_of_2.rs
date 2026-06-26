use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
};

use viprs::{
    BuildError, CompiledPipeline, InMemoryImage, ImageCodecExt, ImageMetadata, Interpretation, U8,
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        kernel::InterpolationKernel,
        ops::resample::{Resize, Thumbnail, thumbnail::ThumbnailTarget},
    },
    ports::scheduler::TileScheduler,
};

const NON_POWER_OF_2_FIXTURES: [(&str, u32, u32); 4] = [
    ("bench_640x480.jpg", 640, 480),
    ("bench_1920x1080.jpg", 1920, 1080),
    ("bench_777x333.jpg", 777, 333),
    ("bench_1001x999.jpg", 1001, 999),
];

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

#[cfg(feature = "jpeg")]
fn load_u8_fixture(name: &str) -> InMemoryImage<U8> {
    let path = fixture_path(name);
    InMemoryImage::<U8>::load(&path).unwrap_or_else(|error| {
        panic!("failed to load JPEG fixture {}: {error}", path.display());
    })
}

#[cfg(feature = "jpeg")]
fn output_metadata(image: &InMemoryImage<U8>) -> ImageMetadata {
    let mut metadata = image.metadata().clone();
    if metadata.interpretation.is_none() && image.bands() >= 3 {
        metadata.interpretation = Some(Interpretation::Srgb);
    }
    metadata
}

#[cfg(feature = "jpeg")]
fn memory_source_from_image(image: &InMemoryImage<U8>) -> MemorySource<U8> {
    MemorySource::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .unwrap_or_else(|error| panic!("failed to create memory source: {error}"))
    .with_metadata(output_metadata(image))
}

#[cfg(feature = "jpeg")]
fn execute_to_image<S: viprs::pipeline::Commit>(
    image: &InMemoryImage<U8>,
    op_name: &str,
    configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
) -> (CompiledPipeline, InMemoryImage<U8>) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(
            image,
        )))
        .unwrap_or_else(|error| panic!("{op_name} stage failed: {error:?}"))
        .build()
        .unwrap_or_else(|error| panic!("{op_name} build failed: {error:?}"));
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(2)
            .unwrap_or_else(|error| panic!("scheduler construction failed: {error}"))
            .run(&pipeline, &mut sink)
            .unwrap_or_else(|error| panic!("{op_name} execution failed: {error}"));
        let output = sink
            .into_image::<U8>(
                pipeline.width,
                pipeline.height,
                pipeline.output_bands,
                output_metadata(image),
            )
            .unwrap_or_else(|error| panic!("failed to materialize {op_name} output: {error}"));
        (pipeline, output)
    }));

    assert!(
        result.is_ok(),
        "{op_name} panicked for {}x{} input",
        image.width(),
        image.height()
    );

    result.unwrap()
}

#[cfg(feature = "jpeg")]
fn thumbnail_config() -> Thumbnail {
    Thumbnail::new(ThumbnailTarget::Width(200), InterpolationKernel::Lanczos3)
}

#[cfg(feature = "jpeg")]
fn affine_target_dims(width: u32, height: u32) -> (u32, u32) {
    (
        (width.saturating_mul(2)).div_ceil(3),
        (height.saturating_mul(3)).div_ceil(5),
    )
}

#[test]
#[cfg(feature = "jpeg")]
fn thumbnail_smoke_handles_non_power_of_2_fixtures() {
    for (name, expected_width, expected_height) in NON_POWER_OF_2_FIXTURES {
        let image = load_u8_fixture(name);
        assert_eq!(
            (image.width(), image.height()),
            (expected_width, expected_height),
            "fixture {name} must have the expected source dimensions"
        );

        let thumbnail = thumbnail_config();
        let expected = thumbnail.into_pipeline_nodes(image.width(), image.height(), image.bands());
        let (pipeline, output) =
            execute_to_image(&image, "thumbnail", |builder| builder.thumbnail(thumbnail));

        assert_eq!(
            (pipeline.width, pipeline.height),
            (expected.output_width, expected.output_height),
            "thumbnail pipeline dimensions must match the plan for {name}"
        );
        assert_eq!(
            (output.width(), output.height()),
            (expected.output_width, expected.output_height),
            "thumbnail output image must match the planned dimensions for {name}"
        );
    }
}

#[test]
#[cfg(feature = "jpeg")]
fn resize_smoke_handles_non_power_of_2_fixtures() {
    for (name, expected_width, expected_height) in NON_POWER_OF_2_FIXTURES {
        let image = load_u8_fixture(name);
        assert_eq!(
            (image.width(), image.height()),
            (expected_width, expected_height),
            "fixture {name} must have the expected source dimensions"
        );

        let resize = Resize::new(0.5, 0.5, InterpolationKernel::Lanczos3);
        let expected = resize.into_pipeline_nodes(image.width(), image.height());
        let (pipeline, output) =
            execute_to_image(&image, "resize", |builder| builder.resize(resize));

        assert_eq!(
            (pipeline.width, pipeline.height),
            (expected.output_width, expected.output_height),
            "resize pipeline dimensions must match the plan for {name}"
        );
        assert_eq!(
            (output.width(), output.height()),
            (expected.output_width, expected.output_height),
            "resize output image must match the planned dimensions for {name}"
        );
    }
}

#[test]
#[cfg(feature = "jpeg")]
fn affine_smoke_handles_non_power_of_2_fixtures() {
    for (name, expected_width, expected_height) in NON_POWER_OF_2_FIXTURES {
        let image = load_u8_fixture(name);
        assert_eq!(
            (image.width(), image.height()),
            (expected_width, expected_height),
            "fixture {name} must have the expected source dimensions"
        );

        let (output_width, output_height) = affine_target_dims(image.width(), image.height());
        let matrix = [
            image.width() as f64 / output_width as f64,
            0.0,
            0.0,
            image.height() as f64 / output_height as f64,
        ];
        let (pipeline, output) = execute_to_image(&image, "affine", |builder| {
            builder.affine(
                matrix,
                0.0,
                0.0,
                output_width,
                output_height,
                InterpolationKernel::Bilinear,
            )
        });

        assert_eq!(
            (pipeline.width, pipeline.height),
            (output_width, output_height),
            "affine pipeline dimensions must match the requested canvas for {name}"
        );
        assert_eq!(
            (output.width(), output.height()),
            (output_width, output_height),
            "affine output image must match the requested canvas for {name}"
        );
    }
}
