//! Ergonomic high-level image API built on top of sources, pipelines, and codecs.
//!
//! This façade solves the common "decode, transform, encode" workflow with a
//! fluent chain that hides pipeline construction details while still using the
//! same compiled execution machinery as lower-level adapters.

use std::{
    io::{Read, Write},
    mem::size_of,
    path::Path,
};

use bytemuck::Pod;
#[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
use std::{fs, sync::Arc};

#[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
use crate::domain::codec_options::SaveOptions;
use crate::{
    adapters::{
        codecs::ForeignRegistry,
        pipeline::{CompiledPipeline, PipelineBuilder, PipelineOp},
        scheduler::rayon_scheduler::RayonScheduler,
        sources::memory::MemorySource,
    },
    domain::{
        codec_options::LoadOptions,
        error::{BuildError, ViprsError},
        format::{BandFormat, BandFormatId, F32, F64, I16, I32, U8, U16, U32},
        image::Image,
        kernel::InterpolationKernel,
        limits::{DecodeLimits, ResourceLimits},
        ops::conversion::SmartcropOp,
        ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
    },
};

#[cfg(feature = "jpeg")]
use crate::adapters::codecs::JpegCodec;
#[cfg(feature = "png")]
use crate::adapters::codecs::PngCodec;
#[cfg(feature = "webp")]
use crate::adapters::codecs::WebpCodec;
#[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
use crate::ports::codec::ImageEncoder;

#[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
use crate::adapters::sources::decoder_source::DecoderSource;

const JPEG_HEADER: [u8; 3] = [0xFF, 0xD8, 0xFF];
const PNG_HEADER: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
#[cfg(feature = "webp")]
const WEBP_RIFF_HEADER: [u8; 4] = *b"RIFF";
#[cfg(feature = "webp")]
const WEBP_MAGIC: [u8; 4] = *b"WEBP";
#[cfg(feature = "png")]
const PNG_IHDR_BIT_DEPTH_OFFSET: usize = 24;
const DEFAULT_SHARPEN_SIGMA: f32 = 0.5;
const DEFAULT_SHARPEN_X1: f32 = 2.0;
const DEFAULT_SHARPEN_Y2: f32 = 10.0;
const DEFAULT_SHARPEN_Y3: f32 = 20.0;
const DEFAULT_SHARPEN_M1: f32 = 0.0;
const DEFAULT_SHARPEN_M2: f32 = 3.0;

macro_rules! with_output_image {
    ($pipeline:expr, $scheduler:expr, |$image:ident| $body:expr) => {{
        match $pipeline.output_format {
            BandFormatId::U8 => {
                let $image = $pipeline.run_to_image::<U8, _>($scheduler)?;
                $body
            }
            BandFormatId::U16 => {
                let $image = $pipeline.run_to_image::<U16, _>($scheduler)?;
                $body
            }
            BandFormatId::I16 => {
                let $image = $pipeline.run_to_image::<I16, _>($scheduler)?;
                $body
            }
            BandFormatId::U32 => {
                let $image = $pipeline.run_to_image::<U32, _>($scheduler)?;
                $body
            }
            BandFormatId::I32 => {
                let $image = $pipeline.run_to_image::<I32, _>($scheduler)?;
                $body
            }
            BandFormatId::F32 => {
                let $image = $pipeline.run_to_image::<F32, _>($scheduler)?;
                $body
            }
            BandFormatId::F64 => {
                let $image = $pipeline.run_to_image::<F64, _>($scheduler)?;
                $body
            }
        }
    }};
}

mod load;
#[cfg(feature = "icc")]
pub use load::ImageApiThumbnailOptions;
pub use load::{ImageApi, ImageApiLoader};

mod encode;
mod transform;

#[cfg(test)]
mod tests;
