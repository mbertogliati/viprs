mod core;
mod metadata;
mod region;
#[cfg(test)]
mod tests;

pub use crate::op::DemandHint;
pub use core::Image;
pub use metadata::{
    AnimationFrame, AnimationLoopCount, FrameDisposal, ImageMetadata, Interpretation,
    MetadataOverrides, UhdrGainMap, UhdrGainMapMetadata,
};
pub use region::clamp_i64_to_i32;
pub use region::{Region, Tile, TileMut};
