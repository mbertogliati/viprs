use bytemuck::Pod;
use viprs::{
    BandFormat, BuildError, CompiledPipeline, F32, Image, ImageMetadata, Interpretation, U8, U16,
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    ports::scheduler::TileScheduler,
};

pub fn rgb_metadata() -> ImageMetadata {
    ImageMetadata {
        interpretation: Some(Interpretation::Srgb),
        ..ImageMetadata::default()
    }
}

pub fn make_u8_image(width: u32, height: u32, bands: u32, pixels: Vec<u8>) -> Image<U8> {
    Image::from_buffer(width, height, bands, pixels)
        .unwrap()
        .with_metadata(if bands >= 3 {
            rgb_metadata()
        } else {
            ImageMetadata::default()
        })
}

pub fn make_u16_image(width: u32, height: u32, bands: u32, pixels: Vec<u16>) -> Image<U16> {
    Image::from_buffer(width, height, bands, pixels).unwrap()
}

pub fn make_f32_image(
    width: u32,
    height: u32,
    bands: u32,
    pixels: Vec<f32>,
    metadata: ImageMetadata,
) -> Image<F32> {
    Image::from_buffer(width, height, bands, pixels)
        .unwrap()
        .with_metadata(metadata)
}

pub fn memory_source_from_image<F>(image: &Image<F>) -> MemorySource<F>
where
    F: BandFormat,
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

pub fn execute_to_image<F, S: viprs_runtime::pipeline::Flush>(
    image: &Image<F>,
    configure: impl FnOnce(viprs_runtime::pipeline::PipelineBuilder) -> Result<viprs_runtime::pipeline::PipelineBuilder<S>, BuildError>,
) -> Result<(CompiledPipeline, Image<F>), String>
where
    F: BandFormat,
    F::Sample: Pod,
{
    let pipeline = configure(viprs_runtime::pipeline::PipelineBuilder::from_source(memory_source_from_image(
        image,
    )))
    .map_err(|error| format!("stage failed: {error:?}"))?
    .build()
    .map_err(|error| format!("build failed: {error:?}"))?;

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .map_err(|error| format!("scheduler construction failed: {error}"))?
        .run(&pipeline, &mut sink)
        .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

    let output = sink
        .into_image::<F>(
            pipeline.width,
            pipeline.height,
            pipeline.output_bands,
            image.metadata().clone(),
        )
        .map_err(|error| format!("failed to materialize output: {error:?}"))?;

    Ok((pipeline, output))
}

pub fn execute_to_buffer<F, S: viprs_runtime::pipeline::Flush>(
    image: &Image<F>,
    configure: impl FnOnce(viprs_runtime::pipeline::PipelineBuilder) -> Result<viprs_runtime::pipeline::PipelineBuilder<S>, BuildError>,
) -> Result<(CompiledPipeline, Vec<u8>), String>
where
    F: BandFormat,
    F::Sample: Pod,
{
    let pipeline = configure(viprs_runtime::pipeline::PipelineBuilder::from_source(memory_source_from_image(
        image,
    )))
    .map_err(|error| format!("stage failed: {error:?}"))?
    .build()
    .map_err(|error| format!("build failed: {error:?}"))?;

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .map_err(|error| format!("scheduler construction failed: {error}"))?
        .run(&pipeline, &mut sink)
        .map_err(|error| format!("pipeline execution failed: {error:?}"))?;

    Ok((pipeline, sink.into_buffer()))
}
