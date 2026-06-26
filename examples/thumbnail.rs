#![allow(missing_docs)]

#[cfg(feature = "jpeg")]
use std::fs;

#[cfg(feature = "jpeg")]
use viprs::{
    ViprsError,
    adapters::{
        codecs::JpegCodec, pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
    },
    domain::{
        codec_options::{LoadOptions, SaveOptions},
        format::U8,
        ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
    },
    kernel::InterpolationKernel,
    ports::codec::{ImageDecoder, ImageEncoder},
    sources::decoder_source::DecoderSource,
};

#[cfg(feature = "jpeg")]
fn usage(program: &str) -> ! {
    eprintln!("Usage: {program} <input.jpg> <output.jpg> [width]");
    eprintln!("Example: cargo run --example thumbnail --features jpeg -- in.jpg out.jpg 400");
    std::process::exit(1);
}

#[cfg(feature = "jpeg")]
fn parse_width(value: Option<String>) -> Result<u32, ViprsError> {
    match value {
        Some(raw) => raw.parse::<u32>().map_err(|err| {
            ViprsError::Codec(format!("thumbnail width must be a positive integer: {err}"))
        }),
        None => Ok(400),
    }
}

#[cfg(feature = "jpeg")]
fn main() -> Result<(), ViprsError> {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "thumbnail".to_owned());
    let input = args.next().unwrap_or_else(|| usage(&program));
    let output = args.next().unwrap_or_else(|| usage(&program));
    let target_width = parse_width(args.next())?;

    let source = DecoderSource::<_, U8>::probed_path_with_options(
        JpegCodec,
        &input,
        LoadOptions::default(),
    )?;
    let pipeline = ImagePipeline::from_source(source)
        .thumbnail_with(Thumbnail::new(
            ThumbnailTarget::Width(target_width),
            InterpolationKernel::Lanczos3,
        ))?
        .build()?;
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads())?;
    let image = pipeline.run_to_image::<U8, _>(&scheduler)?;
    let encoded = JpegCodec.encode_with_options(&image, &SaveOptions::default())?;
    fs::write(&output, encoded)?;

    println!("Saved thumbnail to {output}");
    Ok(())
}

#[cfg(not(feature = "jpeg"))]
fn main() {
    eprintln!("This example requires --features jpeg");
    std::process::exit(1);
}
