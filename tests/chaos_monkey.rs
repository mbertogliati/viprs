#![allow(missing_docs)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use bytemuck::Pod;
use viprs::{
    BuildError, CompiledPipeline, Image, U8,
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::{kernel::InterpolationKernel, ops::resample::Resize},
    ports::scheduler::TileScheduler,
};

fn zero_band_image() -> Image<U8> {
    Image::from_buffer(1, 1, 0, Vec::<u8>::new()).expect("zero-band image should construct")
}

fn memory_source_from_image<F>(image: &Image<F>) -> MemorySource<F>
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
    .expect("memory source should construct")
    .with_metadata(image.metadata().clone())
}

fn execute_pipeline<F>(
    image: &Image<F>,
    configure: impl FnOnce(
        viprs_runtime::pipeline::internal::PipelineBuilder,
    ) -> Result<CompiledPipeline, BuildError>,
) -> Result<Vec<u8>, String>
where
    F: viprs::BandFormat,
    F::Sample: Pod,
{
    let pipeline = configure(
        viprs_runtime::pipeline::internal::PipelineBuilder::from_source(memory_source_from_image(
            image,
        )),
    )
    .map_err(|err| format!("build failed: {err:?}"))?;
    let mut sink =
        MemorySink::for_pipeline(&pipeline).map_err(|err| format!("sink failed: {err:?}"))?;
    RayonScheduler::new(2)
        .map_err(|err| format!("scheduler failed: {err}"))?
        .run(&pipeline, &mut sink)
        .map_err(|err| format!("run failed: {err:?}"))?;
    Ok(sink.into_buffer())
}

fn assert_zero_band_pipeline_is_rejected(
    op_name: &str,
    configure: impl FnOnce(
        viprs_runtime::pipeline::internal::PipelineBuilder,
    ) -> Result<CompiledPipeline, BuildError>,
) {
    let image = zero_band_image();
    let result = catch_unwind(AssertUnwindSafe(|| execute_pipeline(&image, configure)));
    match result {
        Err(_) => panic!("zero-band {op_name} panicked instead of returning a typed error"),
        Ok(Ok(_)) => panic!("zero-band {op_name} unexpectedly succeeded"),
        Ok(Err(message)) => assert!(
            message.contains("Build")
                || message.contains("SourceHint")
                || message.contains("Unsupported")
                || message.contains("Invalid"),
            "zero-band {op_name} failed without a typed build/runtime error: {message}"
        ),
    }
}

#[test]
#[ignore = "BUG B-675: zero-band invert pipeline succeeds instead of rejecting invalid input"]
fn zero_band_invert_is_rejected() {
    assert_zero_band_pipeline_is_rejected("invert", |builder| builder.invert()?.build());
}

#[test]
#[ignore = "BUG B-677: zero-band flip_horizontal pipeline succeeds instead of rejecting invalid input"]
fn zero_band_flip_horizontal_is_rejected() {
    assert_zero_band_pipeline_is_rejected("flip_horizontal", |builder| {
        builder.flip_horizontal()?.build()
    });
}

#[test]
#[ignore = "BUG B-678: zero-band rotate90 pipeline succeeds instead of rejecting invalid input"]
fn zero_band_rotate90_is_rejected() {
    assert_zero_band_pipeline_is_rejected("rotate90", |builder| builder.rotate90()?.build());
}

#[test]
#[ignore = "BUG B-679: zero-band resize panics in zoom instead of returning a typed error"]
fn zero_band_resize_is_rejected() {
    assert_zero_band_pipeline_is_rejected("resize", |builder| {
        builder
            .resize(Resize::new(1.5, 1.5, InterpolationKernel::Lanczos3))?
            .build()
    });
}
