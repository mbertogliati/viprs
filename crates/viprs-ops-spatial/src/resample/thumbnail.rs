//! Thumbnail: composite high-quality downscale.
//!
//! `Thumbnail` is a **composite** operation, not a primitive. It decomposes into:
#![allow(clippy::unused_self)]
// REASON: planning helpers stay instance-bound alongside the builder-style thumbnail API.
#![allow(clippy::wrong_self_convention)]
// REASON: `into_*` naming is kept for parity with the existing public thumbnail planning API.
//!
//!   1. `shrink` — integer box filter for large reduction factors (avoids aliasing)
//!   2. `ReduceH` + `ReduceV` — fractional remainder with the configured kernel
//!   3. `Affine` — final adjustment if the target size requires sub-pixel accuracy
//!   4. `premultiply` / `unpremultiply` — correct alpha-aware resampling while preserving alpha
//!
//! `Thumbnail` is NOT a primitive because:
//! - The shrink-on-load hint to the codec is a `ImageDecoder` concern, not an op.
//! - The composition of steps is fixed; no pixel path benefit from merging them
//!   into a single tile loop (each step has a different halo / tile geometry).
//! - `TileCache` must be available before the `Affine` step can operate on a
//!   sequential source.
//!
//! `Thumbnail::into_pipeline_nodes()` computes this decomposition as a pure plan.
//! The adapter layer (`PipelineBuilder::thumbnail`) materialises the plan into
//! concrete `DynOperation` nodes and may pass `shrink_factor` back to the
//! source when shrink-on-load is available, without importing adapter types here.

use viprs_core::{error::BuildError, kernel::InterpolationKernel};

const EPSILON: f64 = 1e-9;

#[inline]
const fn affine_kernel_for_resize_kernel(kernel: InterpolationKernel) -> InterpolationKernel {
    match kernel {
        InterpolationKernel::Nearest => InterpolationKernel::Nearest,
        InterpolationKernel::Bilinear => InterpolationKernel::Bilinear,
        _ => InterpolationKernel::Bicubic,
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
// REASON: thumbnail nodes store floating-point parameters, so `Eq` is not representable.
#[derive(Debug, Clone, PartialEq)]
/// Enumerates the available thumbnail node values.
pub enum ThumbnailNode {
    /// Uses the `Premultiply` variant of `ThumbnailNode`.
    Premultiply,
    /// Uses the `ShrinkH` variant of `ThumbnailNode`.
    ShrinkH {
        /// Stores the `factor` value for this item.
        factor: u32,
    },
    /// Uses the `ShrinkV` variant of `ThumbnailNode`.
    ShrinkV {
        /// Stores the `factor` value for this item.
        factor: u32,
    },
    /// Uses the `ReduceH` variant of `ThumbnailNode`.
    ReduceH {
        /// Stores the `factor` value for this item.
        factor: f64,
        /// Stores the `kernel` value for this item.
        kernel: InterpolationKernel,
    },
    /// Uses the `ReduceV` variant of `ThumbnailNode`.
    ReduceV {
        /// Stores the `factor` value for this item.
        factor: f64,
        /// Stores the `kernel` value for this item.
        kernel: InterpolationKernel,
    },
    /// Uses the `Affine` variant of `ThumbnailNode`.
    Affine {
        /// Matrix associated with this condition.
        matrix: [f64; 4],
        /// Stores the `tx` value for this item.
        tx: f64,
        /// Stores the `ty` value for this item.
        ty: f64,
        /// Output width associated with this condition.
        output_width: u32,
        /// Output height associated with this condition.
        output_height: u32,
        /// Stores the `kernel` value for this item.
        kernel: InterpolationKernel,
    },
    /// Uses the `ExtractArea` variant of `ThumbnailNode`.
    ExtractArea {
        /// Horizontal factor associated with this condition.
        x: u32,
        /// Vertical factor associated with this condition.
        y: u32,
        /// Width associated with this item.
        width: u32,
        /// Height associated with this item.
        height: u32,
    },
    /// Uses the `Unpremultiply` variant of `ThumbnailNode`.
    Unpremultiply,
    /// Uses the `Flatten` variant of `ThumbnailNode`.
    Flatten {
        /// Stores the `background` value for this item.
        background: [f32; 4],
    },
}

#[derive(Debug, Clone, PartialEq)]
/// Represents a thumbnail pipeline nodes.
pub struct ThumbnailPipelineNodes {
    /// Optional shrink-on-load factor pushed into the source adapter.
    pub shrink_factor: Option<u32>,
    /// Ordered node plan the adapter layer should materialize.
    pub nodes: Vec<ThumbnailNode>,
    /// Output width associated with this condition.
    pub output_width: u32,
    /// Output height associated with this condition.
    pub output_height: u32,
}

/// Target size specification for `Thumbnail`.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbnailTarget {
    /// Fit within `width × height` while preserving aspect ratio without upscaling.
    FitBox {
        /// Bounding-box width to fit inside.
        width: u32,
        /// Bounding-box height to fit inside.
        height: u32,
    },
    /// Force exact output dimensions (may distort aspect ratio).
    ForceExact {
        /// Exact output width to produce.
        width: u32,
        /// Exact output height to produce.
        height: u32,
    },
    /// Constrain only width without upscaling; height is derived from aspect ratio.
    Width(u32),
    /// Constrain only height without upscaling; width is derived from aspect ratio.
    Height(u32),
}

/// Composite thumbnail operation.
///
/// Decomposed into `shrink + ReduceH + ReduceV + Affine + flatten + premultiply`
/// at pipeline construction time. The decomposition is performed by
/// `into_pipeline_nodes()`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::resample::thumbnail::Thumbnail;
///
/// let op = Thumbnail::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[allow(dead_code)]
pub struct Thumbnail {
    target: ThumbnailTarget,
    kernel: InterpolationKernel,
    crop: bool,
    /// Background colour reserved for thumbnail edge handling (RGBA, f32 range [0,1]).
    background: [f32; 4],
}

impl Thumbnail {
    /// Create a `Thumbnail` configuration.
    #[must_use]
    pub const fn new(target: ThumbnailTarget, kernel: InterpolationKernel) -> Self {
        Self {
            target,
            kernel,
            crop: false,
            background: [1.0, 1.0, 1.0, 1.0], // opaque white
        }
    }

    /// Override the background colour used for thumbnail edge handling.
    ///
    /// `background` is RGBA with channels in `[0.0, 1.0]`.
    #[must_use]
    pub const fn with_background(mut self, background: [f32; 4]) -> Self {
        self.background = background;
        self
    }

    /// Enable center-crop-to-fit after resizing to a covering box.
    #[must_use]
    pub const fn with_crop(mut self, crop: bool) -> Self {
        self.crop = crop;
        self
    }

    /// Returns or performs validate input.
    pub const fn validate_input(&self, bands: u32) -> Result<(), BuildError> {
        if bands == 0 {
            return Err(BuildError::InvalidThumbnailParameters {
                message: "band count must be greater than zero",
            });
        }
        Ok(())
    }

    #[must_use]
    /// Returns or performs into pipeline nodes.
    pub fn into_pipeline_nodes(
        &self,
        input_width: u32,
        input_height: u32,
        bands: u32,
    ) -> ThumbnailPipelineNodes {
        self.into_pipeline_nodes_internal(input_width, input_height, bands, true)
    }

    /// Build the thumbnail plan while forcing the explicit shrink/reduce stages
    /// instead of relying on any loader shrink-on-load hint.
    ///
    /// This is used by tooling that needs to inspect the in-memory pipeline shape
    /// exactly as `PipelineBuilder::thumbnail` will materialize it for sources
    /// that do not support loader-side pre-shrink.
    #[must_use]
    pub fn into_pipeline_nodes_without_shrink_hint(
        &self,
        input_width: u32,
        input_height: u32,
        bands: u32,
    ) -> ThumbnailPipelineNodes {
        self.into_pipeline_nodes_internal(input_width, input_height, bands, false)
    }

    fn into_pipeline_nodes_internal(
        &self,
        input_width: u32,
        input_height: u32,
        bands: u32,
        allow_shrink_hint: bool,
    ) -> ThumbnailPipelineNodes {
        if input_width == 0 || input_height == 0 {
            return ThumbnailPipelineNodes {
                shrink_factor: None,
                nodes: Vec::new(),
                output_width: 0,
                output_height: 0,
            };
        }

        let allow_upscale = matches!(self.target, ThumbnailTarget::ForceExact { .. });
        let single_pixel_input = input_width == 1 && input_height == 1;
        let (mut target_width, mut target_height) = if self.crop {
            match self.target {
                ThumbnailTarget::FitBox { width, height }
                | ThumbnailTarget::ForceExact { width, height } => (width.max(1), height.max(1)),
                _ => self.target_dimensions(input_width, input_height),
            }
        } else {
            self.target_dimensions(input_width, input_height)
        };
        if allow_upscale && !single_pixel_input {
            target_width = target_width.max(1);
            target_height = target_height.max(1);
        } else {
            target_width = target_width.min(input_width).max(1);
            target_height = target_height.min(input_height).max(1);
        }
        let has_alpha = matches!(bands, 2 | 4);
        let (mut geometry_width, mut geometry_height) = if self.crop {
            let hscale = f64::from(target_width.max(1)) / f64::from(input_width);
            let vscale = f64::from(target_height.max(1)) / f64::from(input_height);
            let cover = hscale.max(vscale);
            (
                (f64::from(input_width) * cover)
                    .round()
                    .max(f64::from(target_width)) as u32,
                (f64::from(input_height) * cover)
                    .round()
                    .max(f64::from(target_height)) as u32,
            )
        } else {
            (target_width, target_height)
        };
        if allow_upscale && !single_pixel_input {
            geometry_width = geometry_width.max(1);
            geometry_height = geometry_height.max(1);
        } else {
            geometry_width = geometry_width.min(input_width).max(1);
            geometry_height = geometry_height.min(input_height).max(1);
        }

        if input_width == 1 || input_height == 1 {
            if single_pixel_input {
                target_width = 1;
                target_height = 1;
            }
            let mut geometry_nodes = Vec::new();
            if target_width != input_width || target_height != input_height {
                let inv_scale_x = f64::from(input_width) / f64::from(target_width);
                let inv_scale_y = f64::from(input_height) / f64::from(target_height);
                let tx = if self.kernel == InterpolationKernel::Nearest {
                    0.0
                } else {
                    0.5 * (1.0 - inv_scale_x)
                };
                let ty = if self.kernel == InterpolationKernel::Nearest {
                    0.0
                } else {
                    0.5 * (1.0 - inv_scale_y)
                };

                geometry_nodes.push(ThumbnailNode::Affine {
                    matrix: [inv_scale_x, 0.0, 0.0, inv_scale_y],
                    tx,
                    ty,
                    output_width: target_width,
                    output_height: target_height,
                    kernel: affine_kernel_for_resize_kernel(self.kernel),
                });
            }

            return ThumbnailPipelineNodes {
                shrink_factor: None,
                nodes: self.wrap_geometry_nodes(has_alpha, geometry_nodes),
                output_width: target_width,
                output_height: target_height,
            };
        }

        let hshrink = f64::from(input_width) / f64::from(geometry_width);
        let vshrink = f64::from(input_height) / f64::from(geometry_height);
        let shrink_factor = allow_shrink_hint
            .then(|| shrink_on_load_factor(hshrink.min(vshrink)))
            .flatten();
        let effective_input_width =
            shrink_factor.map_or(input_width, |factor| (input_width / factor).max(1));
        let effective_input_height =
            shrink_factor.map_or(input_height, |factor| (input_height / factor).max(1));

        let hshrink = f64::from(effective_input_width) / f64::from(geometry_width);
        let vshrink = f64::from(effective_input_height) / f64::from(geometry_height);

        let int_hshrink = choose_integer_shrink(effective_input_width, geometry_width, hshrink);
        let int_vshrink = choose_integer_shrink(effective_input_height, geometry_height, vshrink);

        let after_shrink_width = effective_input_width / int_hshrink.max(1);
        let after_shrink_height = effective_input_height / int_vshrink.max(1);

        let residual_h = if hshrink > 1.0 {
            hshrink / f64::from(int_hshrink)
        } else {
            1.0
        };
        let residual_v = if vshrink > 1.0 {
            vshrink / f64::from(int_vshrink)
        } else {
            1.0
        };

        let after_reduce_width = if residual_h > 1.0 + EPSILON {
            (f64::from(after_shrink_width) / residual_h)
                .round()
                .max(1.0) as u32
        } else {
            after_shrink_width
        };
        let after_reduce_height = if residual_v > 1.0 + EPSILON {
            (f64::from(after_shrink_height) / residual_v)
                .round()
                .max(1.0) as u32
        } else {
            after_shrink_height
        };

        let mut geometry_nodes = Vec::new();

        if int_hshrink > 1 {
            geometry_nodes.push(ThumbnailNode::ShrinkH {
                factor: int_hshrink,
            });
        }
        if int_vshrink > 1 {
            geometry_nodes.push(ThumbnailNode::ShrinkV {
                factor: int_vshrink,
            });
        }
        if residual_h > 1.0 + EPSILON {
            geometry_nodes.push(ThumbnailNode::ReduceH {
                factor: residual_h,
                kernel: self.kernel,
            });
        }
        if residual_v > 1.0 + EPSILON {
            geometry_nodes.push(ThumbnailNode::ReduceV {
                factor: residual_v,
                kernel: self.kernel,
            });
        }

        let affine_scale_x = f64::from(geometry_width) / f64::from(after_reduce_width);
        let affine_scale_y = f64::from(geometry_height) / f64::from(after_reduce_height);
        if (affine_scale_x - 1.0).abs() > EPSILON || (affine_scale_y - 1.0).abs() > EPSILON {
            let inv_scale_x = 1.0 / affine_scale_x;
            let inv_scale_y = 1.0 / affine_scale_y;
            let tx = if self.kernel == InterpolationKernel::Nearest {
                0.0
            } else {
                0.5 * (1.0 - inv_scale_x)
            };
            let ty = if self.kernel == InterpolationKernel::Nearest {
                0.0
            } else {
                0.5 * (1.0 - inv_scale_y)
            };

            geometry_nodes.push(ThumbnailNode::Affine {
                matrix: [inv_scale_x, 0.0, 0.0, inv_scale_y],
                tx,
                ty,
                output_width: geometry_width,
                output_height: geometry_height,
                kernel: affine_kernel_for_resize_kernel(self.kernel),
            });
        }

        if self.crop && (geometry_width != target_width || geometry_height != target_height) {
            geometry_nodes.push(ThumbnailNode::ExtractArea {
                x: (geometry_width - target_width) / 2,
                y: (geometry_height - target_height) / 2,
                width: target_width,
                height: target_height,
            });
        }

        ThumbnailPipelineNodes {
            shrink_factor,
            nodes: self.wrap_geometry_nodes(has_alpha, geometry_nodes),
            output_width: target_width,
            output_height: target_height,
        }
    }

    fn wrap_geometry_nodes(
        &self,
        has_alpha: bool,
        geometry_nodes: Vec<ThumbnailNode>,
    ) -> Vec<ThumbnailNode> {
        let mut nodes = Vec::new();
        if has_alpha && !geometry_nodes.is_empty() {
            nodes.push(ThumbnailNode::Premultiply);
        }
        nodes.extend(geometry_nodes);
        if has_alpha
            && nodes
                .iter()
                .any(|node| !matches!(node, ThumbnailNode::Premultiply))
        {
            nodes.push(ThumbnailNode::Unpremultiply);
        }
        nodes
    }

    fn target_dimensions(&self, input_width: u32, input_height: u32) -> (u32, u32) {
        match self.target {
            ThumbnailTarget::FitBox { width, height } => {
                let hscale = f64::from(width.max(1)) / f64::from(input_width);
                let vscale = f64::from(height.max(1)) / f64::from(input_height);
                let scale = hscale.min(vscale).min(1.0);
                (
                    (f64::from(input_width) * scale).round().max(1.0) as u32,
                    (f64::from(input_height) * scale).round().max(1.0) as u32,
                )
            }
            ThumbnailTarget::ForceExact { width, height } => (width.max(1), height.max(1)),
            ThumbnailTarget::Width(width) => {
                let width = width.max(1).min(input_width);
                let height = ((f64::from(input_height) * f64::from(width)) / f64::from(input_width))
                    .round()
                    .max(1.0) as u32;
                (width, height)
            }
            ThumbnailTarget::Height(height) => {
                let height = height.max(1).min(input_height);
                let width = ((f64::from(input_width) * f64::from(height)) / f64::from(input_height))
                    .round()
                    .max(1.0) as u32;
                (width, height)
            }
        }
    }
}

#[inline]
fn choose_integer_shrink(input_len: u32, target_len: u32, total_shrink: f64) -> u32 {
    if total_shrink <= 1.0 + EPSILON {
        return 1;
    }

    let mut factor = total_shrink.floor().max(1.0) as u32;
    while factor > 1 {
        let after_shrink_len = input_len / factor;
        let residual = total_shrink / f64::from(factor);
        let after_reduce_len = if residual > 1.0 + EPSILON {
            (f64::from(after_shrink_len) / residual).round().max(1.0) as u32
        } else {
            after_shrink_len
        };

        if after_reduce_len >= target_len {
            break;
        }
        factor -= 1;
    }

    factor
}

#[inline]
fn shrink_on_load_factor(common_shrink: f64) -> Option<u32> {
    if common_shrink >= 16.0 {
        Some(16)
    } else if common_shrink >= 8.0 {
        Some(4)
    } else if common_shrink >= 4.0 {
        Some(2)
    } else {
        None
    }
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use crate::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        ports::scheduler::TileScheduler,
    };
    use proptest::prelude::*;
    use viprs_core::{error::BuildError, format::U8};

    fn run_thumbnail_pipeline(
        input_width: u32,
        input_height: u32,
        target_width: u32,
        target_height: u32,
    ) -> (u32, u32, Vec<u8>) {
        let pixels = vec![128u8; input_width as usize * input_height as usize * 3];
        run_thumbnail_pipeline_with_pixels(
            input_width,
            input_height,
            3,
            pixels,
            target_width,
            target_height,
            InterpolationKernel::Lanczos3,
        )
    }

    fn run_thumbnail_pipeline_with_pixels(
        input_width: u32,
        input_height: u32,
        bands: u32,
        pixels: Vec<u8>,
        target_width: u32,
        target_height: u32,
        kernel: InterpolationKernel,
    ) -> (u32, u32, Vec<u8>) {
        let source = MemorySource::<U8>::new(input_width, input_height, bands, pixels).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .thumbnail(Thumbnail::new(
                ThumbnailTarget::FitBox {
                    width: target_width,
                    height: target_height,
                },
                kernel,
            ))
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(RayonScheduler::default_threads())
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();

        (pipeline.width, pipeline.height, sink.into_buffer())
    }

    fn stepped_gradient_pixels(width: u32, height: u32) -> Vec<u8> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let ramp = x * 5 + y * 3;
                    let plateau = if x >= width / 3 && x < (width * 2) / 3 {
                        72
                    } else {
                        0
                    };
                    let stripes = if (x / 2 + y / 3) % 2 == 0 { 24 } else { 0 };
                    (ramp + plateau + stripes).min(u32::from(u8::MAX)) as u8
                })
            })
            .collect()
    }

    #[test]
    fn non_integral_thumbnail_uses_shrink_and_reduce_when_geometry_lands_exactly() {
        let thumbnail = Thumbnail::new(ThumbnailTarget::Width(99), InterpolationKernel::Lanczos3);
        let plan = thumbnail.into_pipeline_nodes(4000, 3000, 3);

        assert_eq!(plan.shrink_factor, Some(16));
        assert_eq!(plan.output_width, 99);
        assert_eq!(plan.output_height, 74);
        assert!(matches!(
            plan.nodes[0],
            ThumbnailNode::ShrinkH { factor: 2 }
        ));
        assert!(matches!(
            plan.nodes[1],
            ThumbnailNode::ShrinkV { factor: 2 }
        ));
        assert!(matches!(plan.nodes[2], ThumbnailNode::ReduceH { .. }));
        assert!(matches!(plan.nodes[3], ThumbnailNode::ReduceV { .. }));
        assert_eq!(plan.nodes.len(), 4);
    }

    #[test]
    fn crop_mode_adds_center_extract_area() {
        let plan = Thumbnail::new(
            ThumbnailTarget::FitBox {
                width: 120,
                height: 120,
            },
            InterpolationKernel::Lanczos3,
        )
        .with_crop(true)
        .into_pipeline_nodes(400, 200, 3);

        assert_eq!(plan.shrink_factor, None);
        assert_eq!(plan.output_width, 120);
        assert_eq!(plan.output_height, 120);
        assert!(plan.nodes.iter().any(|node| matches!(
            node,
            ThumbnailNode::ExtractArea {
                width: 120,
                height: 120,
                ..
            }
        )));
    }

    #[test]
    fn shrink_on_load_leaves_at_least_two_for_final_resize() {
        assert_eq!(shrink_on_load_factor(3.99), None);
        assert_eq!(shrink_on_load_factor(4.0), Some(2));
        assert_eq!(shrink_on_load_factor(7.99), Some(2));
        assert_eq!(shrink_on_load_factor(8.0), Some(4));
        assert_eq!(shrink_on_load_factor(15.99), Some(4));
        assert_eq!(shrink_on_load_factor(16.0), Some(16));
    }

    #[test]
    fn integer_shrink_avoids_affine_upsample_for_2048_to_400_thumbnail() {
        let plan = Thumbnail::new(ThumbnailTarget::Width(400), InterpolationKernel::Lanczos3)
            .into_pipeline_nodes_without_shrink_hint(2048, 2048, 3);

        assert!(matches!(
            plan.nodes.first(),
            Some(ThumbnailNode::ShrinkH { factor: 4 })
        ));
        assert!(matches!(
            plan.nodes.get(1),
            Some(ThumbnailNode::ShrinkV { factor: 4 })
        ));
        assert!(
            !plan
                .nodes
                .iter()
                .any(|node| matches!(node, ThumbnailNode::Affine { .. }))
        );
    }

    #[test]
    fn integer_shrink_never_undershoots_requested_width() {
        for target_width in 1..=2048 {
            let total_shrink = 2048.0 / target_width as f64;
            let factor = choose_integer_shrink(2048, target_width, total_shrink);
            let after_shrink = 2048 / factor;
            let residual = total_shrink / factor as f64;
            let after_reduce = if residual > 1.0 + EPSILON {
                ((after_shrink as f64) / residual).round().max(1.0) as u32
            } else {
                after_shrink
            };

            assert!(
                after_reduce >= target_width,
                "target={target_width} factor={factor} after_reduce={after_reduce}"
            );
        }
    }

    #[test]
    fn rgba_thumbnail_plan_preserves_alpha_band() {
        let plan = Thumbnail::new(ThumbnailTarget::Width(11), InterpolationKernel::Lanczos3)
            .into_pipeline_nodes_without_shrink_hint(37, 19, 4);

        assert!(matches!(
            plan.nodes.first(),
            Some(ThumbnailNode::Premultiply)
        ));
        assert!(matches!(
            plan.nodes.last(),
            Some(ThumbnailNode::Unpremultiply)
        ));
        assert!(
            !plan
                .nodes
                .iter()
                .any(|node| matches!(node, ThumbnailNode::Flatten { .. }))
        );
    }

    #[test]
    fn test_thumbnail_pipeline_width_limited() {
        let (width, height, pixels) = run_thumbnail_pipeline(1000, 500, 100, 100);

        assert_eq!(width, 100);
        assert_eq!(height, 50);
        assert!(pixels.iter().all(|&sample| sample == 128));
    }

    #[test]
    fn test_thumbnail_pipeline_height_limited() {
        let (width, height, pixels) = run_thumbnail_pipeline(500, 1000, 100, 100);

        assert_eq!(width, 50);
        assert_eq!(height, 100);
        assert!(pixels.iter().all(|&sample| sample == 128));
    }

    #[test]
    fn thumbnail_does_not_upscale_single_pixel_input() {
        let (width, height, pixels) = run_thumbnail_pipeline_with_pixels(
            1,
            1,
            3,
            vec![10, 20, 30],
            100,
            100,
            InterpolationKernel::Lanczos3,
        );

        assert_eq!((width, height), (1, 1));
        assert_eq!(pixels, vec![10, 20, 30]);
    }

    #[test]
    fn thumbnail_does_not_upscale_non_single_pixel_inputs() {
        let (width, height, pixels) = run_thumbnail_pipeline_with_pixels(
            32,
            32,
            3,
            vec![10; 32 * 32 * 3],
            64,
            64,
            InterpolationKernel::Lanczos3,
        );

        assert_eq!((width, height), (32, 32));
        assert_eq!(pixels.len(), 32 * 32 * 3);
    }

    #[test]
    fn force_exact_upscales_non_single_pixel_inputs() {
        let source = MemorySource::<U8>::new(32, 32, 3, vec![10; 32 * 32 * 3]).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .thumbnail(Thumbnail::new(
                ThumbnailTarget::ForceExact {
                    width: 64,
                    height: 64,
                },
                InterpolationKernel::Lanczos3,
            ))
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(RayonScheduler::default_threads())
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();

        assert_eq!((pipeline.width, pipeline.height), (64, 64));
        assert_eq!(sink.into_buffer().len(), 64 * 64 * 3);
    }

    #[test]
    fn force_exact_does_not_upscale_single_pixel_input() {
        let source = MemorySource::<U8>::new(1, 1, 3, vec![10, 20, 30]).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .thumbnail(Thumbnail::new(
                ThumbnailTarget::ForceExact {
                    width: 64,
                    height: 64,
                },
                InterpolationKernel::Lanczos3,
            ))
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(RayonScheduler::default_threads())
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();

        assert_eq!((pipeline.width, pipeline.height), (1, 1));
        assert_eq!(sink.into_buffer(), vec![10, 20, 30]);
    }

    #[test]
    fn benchmark_widths_400_and_800_do_not_panic() {
        for target_width in [400, 800] {
            let (width, height, pixels) =
                run_thumbnail_pipeline(1024, 1024, target_width, target_width);

            assert_eq!(width, target_width);
            assert_eq!(height, target_width);
            assert_eq!(pixels.len(), width as usize * height as usize * 3);
        }
    }

    #[test]
    fn thumbnail_rejects_zero_band_images_with_typed_build_error() {
        let result =
            PipelineBuilder::from_source(MemorySource::<U8>::new(4, 4, 0, Vec::new()).unwrap())
                .thumbnail(Thumbnail::new(
                    ThumbnailTarget::Width(2),
                    InterpolationKernel::Lanczos3,
                ));

        assert!(matches!(
            result,
            Err(BuildError::InvalidThumbnailParameters {
                message: "band count must be greater than zero",
            })
        ));
    }

    #[test]
    fn thumbnail_non_constant_input_exposes_lanczos3_reduce_path() {
        let width = 34;
        let height = 34;
        let target = 14;
        let plan = Thumbnail::new(
            ThumbnailTarget::FitBox {
                width: target,
                height: target,
            },
            InterpolationKernel::Lanczos3,
        )
        .into_pipeline_nodes_without_shrink_hint(width, height, 1);

        assert!(
            plan.nodes
                .iter()
                .any(|node| matches!(node, ThumbnailNode::ReduceH { .. }))
        );
        assert!(
            plan.nodes
                .iter()
                .any(|node| matches!(node, ThumbnailNode::ReduceV { .. }))
        );
        let (output_width, output_height, pixels) = run_thumbnail_pipeline_with_pixels(
            width,
            height,
            1,
            stepped_gradient_pixels(width, height),
            target,
            target,
            InterpolationKernel::Lanczos3,
        );
        let (_, _, bilinear_pixels) = run_thumbnail_pipeline_with_pixels(
            width,
            height,
            1,
            stepped_gradient_pixels(width, height),
            target,
            target,
            InterpolationKernel::Bilinear,
        );

        assert_eq!((output_width, output_height), (target, target));
        assert_ne!(pixels, bilinear_pixels);
        assert_eq!(
            pixels,
            vec![
                24, 29, 43, 55, 98, 156, 163, 175, 192, 129, 141, 152, 162, 181, 22, 40, 51, 59,
                111, 158, 172, 184, 194, 142, 144, 160, 173, 180, 34, 46, 58, 68, 116, 168, 178,
                190, 203, 147, 153, 166, 178, 191, 43, 51, 65, 76, 122, 176, 185, 197, 212, 153,
                162, 174, 185, 201, 43, 62, 73, 80, 134, 179, 194, 206, 214, 165, 165, 182, 195,
                201, 61, 65, 79, 92, 134, 193, 199, 211, 228, 165, 178, 188, 198, 218, 58, 76, 88,
                95, 148, 194, 208, 221, 230, 179, 180, 197, 210, 216, 71, 82, 94, 105, 152, 205,
                214, 227, 239, 183, 190, 203, 215, 228, 79, 88, 102, 112, 159, 212, 222, 233, 247,
                190, 198, 211, 222, 237, 80, 99, 109, 117, 170, 216, 230, 241, 249, 201, 202, 218,
                232, 237, 97, 102, 116, 128, 171, 229, 234, 248, 250, 200, 214, 225, 236, 248, 94,
                113, 124, 130, 185, 229, 247, 247, 255, 216, 216, 234, 242, 250, 108, 118, 130,
                141, 188, 242, 245, 252, 254, 216, 227, 238, 249, 252, 115, 125, 138, 148, 195,
                247, 249, 255, 255, 223, 235, 245, 251, 253,
            ]
        );
    }

    #[test]
    fn thumbnail_pipeline_over_512_runs_under_rayon_scheduler() {
        use crate::{
            adapters::{
                pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
                sinks::memory::MemorySink, sources::memory::MemorySource,
            },
            ports::scheduler::TileScheduler,
        };

        let width = 1024;
        let height = 1024;
        let pixels = vec![64u8; width * height];
        let source = MemorySource::<U8>::new(width as u32, height as u32, 1, pixels).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .thumbnail(Thumbnail::new(
                ThumbnailTarget::Width(400),
                InterpolationKernel::Lanczos3,
            ))
            .unwrap()
            .build()
            .unwrap();

        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(2)
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();
        assert_eq!(pipeline.width, 400);
        assert_eq!(pipeline.height, 400);
        assert_eq!(sink.into_buffer().len(), 400 * 400);
    }

    #[test]
    fn thumbnail_maps_lanczos_affine_tail_to_bicubic() {
        let plan = Thumbnail::new(ThumbnailTarget::Height(400), InterpolationKernel::Lanczos3)
            .into_pipeline_nodes_without_shrink_hint(1, 2048, 3);

        assert!(plan.nodes.iter().any(|node| matches!(
            node,
            ThumbnailNode::Affine {
                kernel: InterpolationKernel::Bicubic,
                output_width: 1,
                output_height: 400,
                ..
            }
        )));
    }

    proptest! {
        #[test]
        fn scale_factor_one_keeps_dimensions(
            width in 1u32..=1024,
            height in 1u32..=1024,
            bands in prop_oneof![Just(1u32), Just(3u32), Just(4u32)],
        ) {
            let thumbnail = Thumbnail::new(
                ThumbnailTarget::ForceExact { width, height },
                InterpolationKernel::Lanczos3,
            );
            let plan = thumbnail.into_pipeline_nodes(width, height, bands);

            prop_assert_eq!(plan.shrink_factor, None);
            prop_assert_eq!(plan.output_width, width);
            prop_assert_eq!(plan.output_height, height);
        }
    }
}
