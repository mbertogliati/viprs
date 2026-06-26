//! First-class public image pipeline vocabulary.
//!
//! The API in this module names the public pipeline concepts while delegating
//! execution to the existing compiled pipeline engine.

mod config;
mod format;
mod input;
mod ops;
mod output;
mod pipeline;
mod sink;

pub use crate::domain::{
    colorspace::{Colorspace, ColorspaceId},
    format::BandFormatId,
    image::DemandHint,
    kernel::InterpolationKernel,
    ops::{
        conversion::{Angle, Angle45, ExtendMode, Gravity},
        resample::{Resize, Thumbnail},
    },
};
pub use config::ProcessingConfig;
pub use format::Format;
pub use input::Input;
pub use output::{PipelineOutput, RawOutputPipeline};
pub use pipeline::ImagePipeline;
pub use sink::Sink;
