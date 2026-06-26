pub(crate) use proptest::prelude::*;
use std::path::{Path, PathBuf};
pub(crate) use viprs::{
    BandFormatId, BuildError, CompiledPipeline, ImageCodecExt, ImageMetadata, InMemoryImage,
    Interpretation, U8,
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{Lab, SRgb},
        kernel::InterpolationKernel,
        ops::resample::{Resize, Thumbnail, thumbnail::ThumbnailTarget},
    },
    ports::{
        codec::{ImageDecoder, ImageEncoder},
        scheduler::TileScheduler,
    },
};

#[cfg(feature = "avif")]
pub(crate) use viprs::adapters::codecs::AvifCodec;
#[cfg(feature = "jpeg")]
pub(crate) use viprs::adapters::codecs::JpegCodec;
#[cfg(feature = "png")]
pub(crate) use viprs::adapters::codecs::PngCodec;
#[cfg(feature = "tiff")]
pub(crate) use viprs::adapters::codecs::TiffCodec;
#[cfg(feature = "webp")]
pub(crate) use viprs::adapters::codecs::WebpCodec;

pub(crate) const SHARPEN_SIGMA: f32 = 0.0;
pub(crate) const SHARPEN_X1: f32 = 2.0;
pub(crate) const SHARPEN_Y2: f32 = 10.0;
pub(crate) const SHARPEN_Y3: f32 = 20.0;
pub(crate) const SHARPEN_M1: f32 = 0.0;
pub(crate) const SHARPEN_M2: f32 = 3.0;

pub(crate) fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_owned()
}

pub(crate) fn fixture_path(name: &str) -> PathBuf {
    project_root()
        .join("tests")
        .join("fixtures")
        .join("images")
        .join(name)
}

pub(crate) fn load_u8_fixture(name: &str) -> InMemoryImage<U8> {
    let path = fixture_path(name);
    InMemoryImage::<U8>::load(&path).unwrap_or_else(|error| {
        panic!("failed to load U8 fixture {}: {error}", path.display());
    })
}

pub(crate) fn memory_source_from_image(image: &InMemoryImage<U8>) -> MemorySource<U8> {
    let mut metadata = image.metadata().clone();
    if metadata.interpretation.is_none() && image.bands() >= 3 {
        metadata.interpretation = Some(Interpretation::Srgb);
    }

    MemorySource::<U8>::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .expect("failed to create memory source")
    .with_metadata(metadata)
}

pub(crate) fn execute_u8_pipeline<S: viprs::pipeline::Commit>(
    image: &InMemoryImage<U8>,
    configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
) -> (CompiledPipeline, MemorySink) {
    let pipeline = configure(ImagePipeline::from_source(memory_source_from_image(image)))
        .expect("pipeline stage failed")
        .build()
        .expect("pipeline build failed");

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .expect("scheduler construction failed")
        .run(&pipeline, &mut sink)
        .expect("pipeline execution failed");

    (pipeline, sink)
}

pub(crate) fn execute_u8_pipeline_to_buffer<S: viprs::pipeline::Commit>(
    image: &InMemoryImage<U8>,
    configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
) -> (CompiledPipeline, Vec<u8>) {
    let (pipeline, sink) = execute_u8_pipeline(image, configure);
    let buffer = sink.into_buffer();
    assert_valid_pipeline_buffer(&pipeline, &buffer);
    (pipeline, buffer)
}

#[cfg(any(
    feature = "jpeg",
    feature = "png",
    feature = "webp",
    feature = "tiff",
    feature = "avif"
))]
pub(crate) fn output_image_from_buffer(
    source_image: &InMemoryImage<U8>,
    pipeline: &CompiledPipeline,
    buffer: Vec<u8>,
) -> InMemoryImage<U8> {
    assert_eq!(
        pipeline.output_format,
        BandFormatId::U8,
        "encode roundtrips require U8 output, got {:?}",
        pipeline.output_format
    );

    let output = InMemoryImage::<U8>::from_buffer(
        pipeline.width,
        pipeline.height,
        pipeline.output_bands,
        buffer,
    )
    .expect("failed to materialize pipeline output buffer")
    .with_metadata(output_metadata(source_image.metadata().clone()));

    assert_eq!(output.width(), pipeline.width);
    assert_eq!(output.height(), pipeline.height);
    assert_eq!(output.bands(), pipeline.output_bands);
    assert!(
        output.pixels().iter().any(|&value| value != 0),
        "pipeline output unexpectedly contains only zeros"
    );

    output
}

#[cfg(any(
    feature = "jpeg",
    feature = "png",
    feature = "webp",
    feature = "tiff",
    feature = "avif"
))]
pub(crate) fn execute_u8_pipeline_to_image<S: viprs::pipeline::Commit>(
    image: &InMemoryImage<U8>,
    configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline<S>, BuildError>,
) -> (CompiledPipeline, InMemoryImage<U8>) {
    let (pipeline, buffer) = execute_u8_pipeline_to_buffer(image, configure);
    let output = output_image_from_buffer(image, &pipeline, buffer);
    (pipeline, output)
}

#[cfg(any(
    feature = "jpeg",
    feature = "png",
    feature = "webp",
    feature = "tiff",
    feature = "avif"
))]
pub(crate) fn output_metadata(mut metadata: ImageMetadata) -> ImageMetadata {
    if metadata.interpretation.is_none() {
        metadata.interpretation = Some(Interpretation::Srgb);
    }
    metadata
}

pub(crate) fn thumbnail_config(width: u32) -> Thumbnail {
    Thumbnail::new(ThumbnailTarget::Width(width), InterpolationKernel::Lanczos3)
}

pub(crate) fn expected_thumbnail_dimensions(image: &InMemoryImage<U8>, width: u32) -> (u32, u32) {
    let plan =
        thumbnail_config(width).into_pipeline_nodes(image.width(), image.height(), image.bands());
    (plan.output_width, plan.output_height)
}

pub(crate) fn expected_resize_dimensions(image: &InMemoryImage<U8>, scale: f64) -> (u32, u32) {
    let scaled = |input: u32| ((input as f64 * scale).round() as u32).max(1);
    (scaled(image.width()), scaled(image.height()))
}

pub(crate) fn bytes_per_sample(format: BandFormatId) -> usize {
    match format {
        BandFormatId::U8 => 1,
        BandFormatId::U16 | BandFormatId::I16 => 2,
        BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
        BandFormatId::F64 => 8,
    }
}

pub(crate) fn assert_valid_pipeline_buffer(pipeline: &CompiledPipeline, buffer: &[u8]) {
    let expected_len = pipeline.width as usize
        * pipeline.height as usize
        * pipeline.output_bands as usize
        * bytes_per_sample(pipeline.output_format);
    assert_eq!(
        buffer.len(),
        expected_len,
        "buffer length mismatch for {}x{}x{} {:?}",
        pipeline.width,
        pipeline.height,
        pipeline.output_bands,
        pipeline.output_format
    );
    assert!(
        buffer.iter().any(|&byte| byte != 0),
        "pipeline output unexpectedly contains only zeros"
    );
}

pub(crate) fn assert_thumbnail_dimensions(
    pipeline: &CompiledPipeline,
    image: &InMemoryImage<U8>,
    width: u32,
) {
    let expected = expected_thumbnail_dimensions(image, width);
    assert_eq!(
        (pipeline.width, pipeline.height),
        expected,
        "thumbnail dimensions mismatch for {}x{} -> width {}",
        image.width(),
        image.height(),
        width
    );
}

pub(crate) fn assert_resize_dimensions(
    pipeline: &CompiledPipeline,
    image: &InMemoryImage<U8>,
    scale: f64,
) {
    let expected = expected_resize_dimensions(image, scale);
    assert_eq!(
        (pipeline.width, pipeline.height),
        expected,
        "resize dimensions mismatch for {}x{} -> scale {}",
        image.width(),
        image.height(),
        scale
    );
}

#[cfg(any(
    feature = "jpeg",
    feature = "png",
    feature = "webp",
    feature = "tiff",
    feature = "avif"
))]
pub(crate) fn assert_codec_roundtrip<C: ImageEncoder + ImageDecoder>(
    codec: &C,
    image: &InMemoryImage<U8>,
    expected_width: u32,
    expected_height: u32,
) {
    let encoded = codec.encode(image).expect("encode failed");
    assert!(
        !encoded.is_empty(),
        "{} encoder produced an empty buffer",
        ImageEncoder::format_name(codec)
    );

    let decoded = codec.decode::<U8>(&encoded).expect("decode failed");
    assert_eq!(decoded.width(), expected_width);
    assert_eq!(decoded.height(), expected_height);
    assert!(
        !decoded.pixels().is_empty(),
        "{} decoder returned an empty pixel buffer",
        ImageDecoder::format_name(codec)
    );
}
