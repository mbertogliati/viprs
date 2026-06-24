//! High-level resize decomposition with libvips-style axis planning.
//!
//! `ResizeOp` is a composite operation: it plans a resize as a sequence of
//! `ShrinkH`/`ShrinkV`, `ReduceH`/`ReduceV`, `Zoom`, and `Affine` nodes.
//! The plan is materialised by `PipelineBuilder::resize()`.

use viprs_core::kernel::InterpolationKernel;

const EPSILON: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq)]
/// Enumerates the available resize node values.
pub enum ResizeNode {
    /// Uses the `ShrinkH` variant of `ResizeNode`.
    ShrinkH {
        /// Stores the `factor` value for this item.
        factor: u32,
    },
    /// Uses the `ShrinkV` variant of `ResizeNode`.
    ShrinkV {
        /// Stores the `factor` value for this item.
        factor: u32,
    },
    /// Uses the `ReduceH` variant of `ResizeNode`.
    ReduceH {
        /// Stores the `factor` value for this item.
        factor: f64,
        /// Stores the `kernel` value for this item.
        kernel: InterpolationKernel,
    },
    /// Uses the `ReduceV` variant of `ResizeNode`.
    ReduceV {
        /// Stores the `factor` value for this item.
        factor: f64,
        /// Stores the `kernel` value for this item.
        kernel: InterpolationKernel,
    },
    /// Uses the `Zoom` variant of `ResizeNode`.
    Zoom {
        /// Stores the `xfac` value for this item.
        xfac: u32,
        /// Stores the `yfac` value for this item.
        yfac: u32,
    },
    /// Uses the `Affine` variant of `ResizeNode`.
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
}

#[derive(Debug, Clone, PartialEq)]
/// Represents a resize pipeline nodes.
pub struct ResizePipelineNodes {
    /// Stores the `nodes` value for this item.
    pub nodes: Vec<ResizeNode>,
    /// Output width associated with this condition.
    pub output_width: u32,
    /// Output height associated with this condition.
    pub output_height: u32,
}

/// High-level resize configuration.
///
/// The resize is expressed as horizontal and vertical scale factors:
/// `output_width = round(input_width * hscale)` and
/// `output_height = round(input_height * vscale)`, clamped to at least 1 pixel
/// per axis. Downscale uses integer shrink plus fractional reduce; upscale uses
/// integer zoom plus an affine residual.
///
/// # Examples
/// ```ignore
/// use viprs_ops_spatial::resample::resize::ResizeOp;
///
/// let op = ResizeOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResizeOp {
    /// Stores the `hscale` value for this item.
    pub hscale: f64,
    /// Stores the `vscale` value for this item.
    pub vscale: f64,
    /// Stores the `kernel` value for this item.
    pub kernel: InterpolationKernel,
    /// Stores the `gap` value for this item.
    pub gap: f64,
}

/// Type alias for resize.
pub type Resize = ResizeOp;

impl ResizeOp {
    /// Associated constant for default kernel.
    pub const DEFAULT_KERNEL: InterpolationKernel = InterpolationKernel::Lanczos3;
    /// Associated constant for default gap.
    pub const DEFAULT_GAP: f64 = 2.0;

    /// Create a `ResizeOp` configuration.
    #[must_use]
    pub fn new(hscale: f64, vscale: f64, kernel: InterpolationKernel) -> Self {
        Self::new_with_gap(hscale, vscale, kernel, None)
    }

    /// Create a `ResizeOp` configuration with an explicit optional gap value.
    ///
    /// When `gap` is `None`, libvips default `2.0` is used.
    #[must_use]
    pub fn new_with_gap(
        hscale: f64,
        vscale: f64,
        kernel: InterpolationKernel,
        gap: Option<f64>,
    ) -> Self {
        debug_assert!(
            hscale.is_finite() && hscale > 0.0,
            "ResizeOp: hscale must be finite and > 0"
        );
        debug_assert!(
            vscale.is_finite() && vscale > 0.0,
            "ResizeOp: vscale must be finite and > 0"
        );
        if let Some(gap) = gap {
            debug_assert!(
                gap.is_finite() && gap >= 1.0,
                "ResizeOp: gap must be finite and >= 1"
            );
        }

        Self {
            hscale: sanitized_scale(hscale),
            vscale: sanitized_scale(vscale),
            kernel,
            gap: sanitized_gap(gap.unwrap_or(Self::DEFAULT_GAP)),
        }
    }

    /// Create a `ResizeOp` using libvips' default resize kernel.
    #[must_use]
    pub fn new_default(hscale: f64, vscale: f64) -> Self {
        Self::new_with_gap(hscale, vscale, Self::DEFAULT_KERNEL, None)
    }

    /// Return a copy of this resize configuration with a custom reducing gap.
    #[must_use]
    pub fn with_gap(mut self, gap: f64) -> Self {
        self.gap = sanitized_gap(gap);
        self
    }

    #[must_use]
    /// Returns or performs into pipeline nodes.
    pub fn into_pipeline_nodes(&self, input_width: u32, input_height: u32) -> ResizePipelineNodes {
        if input_width == 0 || input_height == 0 {
            return ResizePipelineNodes {
                nodes: Vec::new(),
                output_width: 0,
                output_height: 0,
            };
        }

        let target_width = scaled_len(input_width, self.hscale);
        let target_height = scaled_len(input_height, self.vscale);
        let effective_kernel = mapped_resize_kernel(self.kernel);

        if effective_kernel == InterpolationKernel::Lbb {
            // LBB parity in libvips is only available through the nonlinear 2-D
            // affine interpolator, not the 1-D reduce family.
            let mut nodes = Vec::new();

            if target_width != input_width || target_height != input_height {
                let affine_scale_x = f64::from(target_width) / f64::from(input_width);
                let affine_scale_y = f64::from(target_height) / f64::from(input_height);
                nodes.push(ResizeNode::Affine {
                    matrix: [1.0 / affine_scale_x, 0.0, 0.0, 1.0 / affine_scale_y],
                    tx: affine_offset(affine_scale_x, effective_kernel),
                    ty: affine_offset(affine_scale_y, effective_kernel),
                    output_width: target_width,
                    output_height: target_height,
                    kernel: effective_kernel,
                });
            }

            return ResizePipelineNodes {
                nodes,
                output_width: target_width,
                output_height: target_height,
            };
        }

        let mut nodes = Vec::new();
        let mut current_width = input_width;
        let mut current_height = input_height;

        let h_shrink = if target_width < current_width {
            pre_shrink_factor(current_width, target_width, effective_kernel, self.gap)
        } else {
            1
        };
        let v_shrink = if target_height < current_height {
            pre_shrink_factor(current_height, target_height, effective_kernel, self.gap)
        } else {
            1
        };

        if v_shrink > 1 {
            nodes.push(ResizeNode::ShrinkV { factor: v_shrink });
            current_height /= v_shrink;
        }
        if h_shrink > 1 {
            nodes.push(ResizeNode::ShrinkH { factor: h_shrink });
            current_width /= h_shrink;
        }

        if target_height < current_height {
            nodes.push(ResizeNode::ReduceV {
                factor: f64::from(current_height) / f64::from(target_height),
                kernel: effective_kernel,
            });
            current_height = target_height;
        }
        if target_width < current_width {
            nodes.push(ResizeNode::ReduceH {
                factor: f64::from(current_width) / f64::from(target_width),
                kernel: effective_kernel,
            });
            current_width = target_width;
        }

        // VSQBS parity depends on its 4×4 affine stencil; integer zoom would
        // bypass the interpolator entirely and diverge from libvips.
        let can_use_zoom = effective_kernel != InterpolationKernel::Vsqbs;
        let zoom_x = if can_use_zoom && target_width > current_width {
            target_width / current_width
        } else {
            1
        };
        let zoom_y = if can_use_zoom && target_height > current_height {
            target_height / current_height
        } else {
            1
        };

        if zoom_x > 1 || zoom_y > 1 {
            nodes.push(ResizeNode::Zoom {
                xfac: zoom_x.max(1),
                yfac: zoom_y.max(1),
            });
            current_width = current_width.saturating_mul(zoom_x.max(1));
            current_height = current_height.saturating_mul(zoom_y.max(1));
        }

        let affine_scale_x = f64::from(target_width) / f64::from(current_width);
        let affine_scale_y = f64::from(target_height) / f64::from(current_height);

        if needs_affine(affine_scale_x) || needs_affine(affine_scale_y) {
            let inv_scale_x = 1.0 / affine_scale_x;
            let inv_scale_y = 1.0 / affine_scale_y;
            nodes.push(ResizeNode::Affine {
                matrix: [inv_scale_x, 0.0, 0.0, inv_scale_y],
                tx: affine_offset(affine_scale_x, effective_kernel),
                ty: affine_offset(affine_scale_y, effective_kernel),
                output_width: target_width,
                output_height: target_height,
                kernel: effective_kernel,
            });
        }

        ResizePipelineNodes {
            nodes,
            output_width: target_width,
            output_height: target_height,
        }
    }
}

#[inline]
const fn mapped_resize_kernel(kernel: InterpolationKernel) -> InterpolationKernel {
    match kernel {
        // libvips resize.c::vips_resize_interpolate() maps non-kernel
        // interpolators (like nohalo) to bicubic.
        InterpolationKernel::Nohalo => InterpolationKernel::Bicubic,
        _ => kernel,
    }
}

#[inline]
fn sanitized_scale(scale: f64) -> f64 {
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}

#[inline]
fn sanitized_gap(gap: f64) -> f64 {
    if gap.is_finite() && gap >= 1.0 {
        gap
    } else {
        ResizeOp::DEFAULT_GAP
    }
}

#[inline]
fn scaled_len(input_len: u32, scale: f64) -> u32 {
    ((f64::from(input_len) * scale).round().max(1.0)) as u32
}

#[inline]
fn pre_shrink_factor(
    input_len: u32,
    target_len: u32,
    kernel: InterpolationKernel,
    gap: f64,
) -> u32 {
    if target_len == 0 || input_len <= target_len || kernel == InterpolationKernel::Nearest {
        return 1;
    }

    ((f64::from(input_len) / f64::from(target_len) / gap)
        .floor()
        .max(1.0)) as u32
}

#[inline]
fn needs_affine(scale: f64) -> bool {
    (scale - 1.0).abs() > EPSILON
}

#[inline]
fn affine_offset(scale: f64, kernel: InterpolationKernel) -> f64 {
    if kernel == InterpolationKernel::Nearest {
        0.0
    } else {
        0.5 * (1.0 - 1.0 / scale)
    }
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::kernel::InterpolationKernel;

    fn simulated_dimensions(
        input_width: u32,
        input_height: u32,
        plan: &ResizePipelineNodes,
    ) -> (u32, u32) {
        let mut width = input_width;
        let mut height = input_height;

        for node in &plan.nodes {
            match node {
                ResizeNode::ShrinkH { factor } => width /= factor,
                ResizeNode::ShrinkV { factor } => height /= factor,
                ResizeNode::ReduceH { factor, .. } => {
                    width = (width as f64 / factor).round() as u32
                }
                ResizeNode::ReduceV { factor, .. } => {
                    height = (height as f64 / factor).round() as u32
                }
                ResizeNode::Zoom { xfac, yfac } => {
                    width *= xfac;
                    height *= yfac;
                }
                ResizeNode::Affine {
                    output_width,
                    output_height,
                    ..
                } => {
                    width = *output_width;
                    height = *output_height;
                }
            }
        }

        (width, height)
    }

    #[test]
    fn resize_default_kernel_matches_libvips_lanczos3() {
        let resize = Resize::new_default(0.75, 0.5);
        assert_eq!(resize.kernel, InterpolationKernel::Lanczos3);
        assert_eq!(resize.gap, Resize::DEFAULT_GAP);
    }

    #[test]
    fn downscale_4x_uses_shrink2_reduce2_with_default_gap() {
        let plan =
            Resize::new(0.25, 0.25, InterpolationKernel::Lanczos3).into_pipeline_nodes(128, 128);
        assert!(matches!(
            plan.nodes.as_slice(),
            [
                ResizeNode::ShrinkV { factor: 2 },
                ResizeNode::ShrinkH { factor: 2 },
                ResizeNode::ReduceV { factor: factor_v, kernel: InterpolationKernel::Lanczos3 },
                ResizeNode::ReduceH { factor, kernel: InterpolationKernel::Lanczos3 },
            ] if (*factor - 2.0).abs() < EPSILON && (*factor_v - 2.0).abs() < EPSILON
        ));
    }

    #[test]
    fn downscale_4x_with_gap1_uses_full_integer_shrink() {
        let plan = Resize::new(0.25, 0.25, InterpolationKernel::Lanczos3)
            .with_gap(1.0)
            .into_pipeline_nodes(128, 128);
        assert!(matches!(
            plan.nodes.as_slice(),
            [
                ResizeNode::ShrinkV { factor: 4 },
                ResizeNode::ShrinkH { factor: 4 }
            ]
        ));
    }

    #[test]
    fn nearest_downscale_4x_skips_box_prefilter_shrink() {
        let plan =
            Resize::new(0.25, 0.25, InterpolationKernel::Nearest).into_pipeline_nodes(128, 128);
        assert!(matches!(
            plan.nodes.as_slice(),
            [
                ResizeNode::ReduceV { factor: factor_v, kernel: InterpolationKernel::Nearest },
                ResizeNode::ReduceH { factor, kernel: InterpolationKernel::Nearest },
            ] if (*factor - 4.0).abs() < EPSILON && (*factor_v - 4.0).abs() < EPSILON
        ));
    }

    #[test]
    fn mixed_resize_uses_downscale_and_upscale_stages() {
        let plan =
            Resize::new(0.5, 2.5, InterpolationKernel::Bilinear).into_pipeline_nodes(100, 100);
        assert!(matches!(
            plan.nodes[0],
            ResizeNode::ReduceH {
                factor,
                kernel: InterpolationKernel::Bilinear,
            } if (factor - 2.0).abs() < EPSILON
        ));
        assert!(matches!(
            plan.nodes[1],
            ResizeNode::Zoom { xfac: 1, yfac: 2 }
        ));
        assert!(matches!(plan.nodes[2], ResizeNode::Affine { .. }));
        assert_eq!(simulated_dimensions(100, 100, &plan), (50, 250));
    }

    #[test]
    fn lbb_resize_uses_affine_stage_for_downscale() {
        let plan = Resize::new(0.6, 0.6, InterpolationKernel::Lbb).into_pipeline_nodes(100, 100);
        assert!(matches!(
            plan.nodes.as_slice(),
            [ResizeNode::Affine {
                kernel: InterpolationKernel::Lbb,
                ..
            }]
        ));
        assert_eq!(simulated_dimensions(100, 100, &plan), (60, 60));
    }

    #[test]
    fn lbb_resize_2x_still_uses_affine_stage() {
        let plan = Resize::new(2.0, 2.0, InterpolationKernel::Lbb).into_pipeline_nodes(25, 25);
        assert!(matches!(
            plan.nodes.as_slice(),
            [ResizeNode::Affine {
                matrix,
                output_width: 50,
                output_height: 50,
                kernel: InterpolationKernel::Lbb,
                ..
            }] if *matrix == [0.5, 0.0, 0.0, 0.5]
        ));
    }

    #[test]
    fn nearest_affine_offset_is_zero() {
        let plan =
            Resize::new(1.5, 1.5, InterpolationKernel::Nearest).into_pipeline_nodes(100, 100);
        match plan.nodes.as_slice() {
            [ResizeNode::Affine { tx, ty, .. }] => {
                assert_eq!(*tx, 0.0);
                assert_eq!(*ty, 0.0);
            }
            other => panic!("unexpected plan: {other:?}"),
        }
    }

    #[test]
    fn vsqbs_resize_2x_uses_affine_instead_of_zoom() {
        let plan = Resize::new(2.0, 2.0, InterpolationKernel::Vsqbs).into_pipeline_nodes(2, 2);

        assert!(matches!(
            plan.nodes.as_slice(),
            [ResizeNode::Affine {
                matrix,
                tx,
                ty,
                output_width: 4,
                output_height: 4,
                kernel: InterpolationKernel::Vsqbs,
            }] if *matrix == [0.5, 0.0, 0.0, 0.5] && (*tx - 0.25).abs() < EPSILON && (*ty - 0.25).abs() < EPSILON
        ));
    }

    proptest! {
        #[test]
        fn plan_dimensions_match_requested_scales(
            width in 1u32..=256,
            height in 1u32..=256,
            hscale in 0.05f64..4.0,
            vscale in 0.05f64..4.0,
        ) {
            let resize = Resize::new(hscale, vscale, InterpolationKernel::Lanczos3);
            let plan = resize.into_pipeline_nodes(width, height);

            prop_assert_eq!(plan.output_width, ((width as f64 * hscale).round().max(1.0)) as u32);
            prop_assert_eq!(plan.output_height, ((height as f64 * vscale).round().max(1.0)) as u32);
            prop_assert_eq!(
                simulated_dimensions(width, height, &plan),
                (plan.output_width, plan.output_height)
            );
        }
    }
}
