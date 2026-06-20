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
/// use viprs::domain::ops::resample::resize::ResizeOp;
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
    use viprs_core::{format::U8, kernel::InterpolationKernel};
    use viprs_ports::scheduler::TileScheduler;
    use viprs_runtime::{
        domain::ops::resample::Resize as RuntimeResize, pipeline::PipelineBuilder,
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };

    fn run_resize_pipeline_with_pixels(
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        hscale: f64,
        vscale: f64,
        kernel: InterpolationKernel,
    ) -> (u32, u32, Vec<u8>) {
        let source = MemorySource::<U8>::new(width, height, 1, pixels).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .resize(RuntimeResize::new(hscale, vscale, kernel))
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(1)
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();

        (pipeline.width, pipeline.height, sink.into_buffer())
    }

    fn run_resize_pipeline(
        width: u32,
        height: u32,
        hscale: f64,
        vscale: f64,
        kernel: InterpolationKernel,
    ) -> (u32, u32, Vec<u8>) {
        run_resize_pipeline_with_pixels(
            width,
            height,
            vec![128u8; width as usize * height as usize],
            hscale,
            vscale,
            kernel,
        )
    }

    fn gradient_pixels(width: u32, height: u32) -> Vec<u8> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| (((x * 9) + (y * 5)).min(u32::from(u8::MAX))) as u8)
            })
            .collect()
    }

    fn plane_gradient_pixels(width: u32, height: u32, x_step: u8, y_step: u8) -> Vec<u8> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let value = (u32::from(x_step) * x) + (u32::from(y_step) * y);
                    value.min(u32::from(u8::MAX)) as u8
                })
            })
            .collect()
    }

    #[cfg(feature = "lock_instrumentation")]
    fn resize_half_profile_for_size(
        size: u32,
    ) -> (
        ResizePipelineNodes,
        viprs_runtime::scheduler::rayon_scheduler::PipelineRunProfile,
    ) {
        let pixels = gradient_pixels(size, size);
        let resize = RuntimeResize::new(0.5, 0.5, InterpolationKernel::Lanczos3);
        let plan = resize.into_pipeline_nodes(size, size);
        let source = MemorySource::<U8>::new(size, size, 1, pixels).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .resize(resize)
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
        let profile = scheduler.run_with_profile(&pipeline, &mut sink).unwrap();
        (plan, profile)
    }

    #[cfg(feature = "lock_instrumentation")]
    fn resize_node_name(node: &ResizeNode) -> &'static str {
        match node {
            ResizeNode::ShrinkH { .. } => "ShrinkH",
            ResizeNode::ShrinkV { .. } => "ShrinkV",
            ResizeNode::ReduceH { .. } => "ReduceH",
            ResizeNode::ReduceV { .. } => "ReduceV",
            ResizeNode::Zoom { .. } => "Zoom",
            ResizeNode::Affine { .. } => "Affine",
        }
    }

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
    fn identity_scale_returns_empty_plan_and_same_pixels() {
        let plan =
            Resize::new(1.0, 1.0, InterpolationKernel::Lanczos3).into_pipeline_nodes(100, 100);
        assert!(plan.nodes.is_empty());
        assert_eq!(plan.output_width, 100);
        assert_eq!(plan.output_height, 100);

        let pixels = gradient_pixels(100, 100);
        let (width, height, buffer) = run_resize_pipeline_with_pixels(
            100,
            100,
            pixels.clone(),
            1.0,
            1.0,
            InterpolationKernel::Lanczos3,
        );
        assert_eq!(width, 100);
        assert_eq!(height, 100);
        assert_eq!(buffer.len(), 100 * 100);
        assert_eq!(buffer, pixels);
    }

    #[test]
    fn resize_default_kernel_matches_libvips_lanczos3() {
        let resize = Resize::new_default(0.75, 0.5);
        assert_eq!(resize.kernel, InterpolationKernel::Lanczos3);
        assert_eq!(resize.gap, Resize::DEFAULT_GAP);
    }

    #[test]
    fn nohalo_resize_request_maps_to_bicubic_plan_and_pixels() {
        let plan_nohalo =
            Resize::new(0.5, 0.5, InterpolationKernel::Nohalo).into_pipeline_nodes(16, 16);
        let plan_bicubic =
            Resize::new(0.5, 0.5, InterpolationKernel::Bicubic).into_pipeline_nodes(16, 16);
        assert_eq!(plan_nohalo, plan_bicubic);

        let pixels = gradient_pixels(16, 16);
        let (_, _, nohalo_output) = run_resize_pipeline_with_pixels(
            16,
            16,
            pixels.clone(),
            0.5,
            0.5,
            InterpolationKernel::Nohalo,
        );
        let (_, _, bicubic_output) =
            run_resize_pipeline_with_pixels(16, 16, pixels, 0.5, 0.5, InterpolationKernel::Bicubic);
        assert_eq!(nohalo_output, bicubic_output);
    }

    #[test]
    fn downscale_2x_uses_fractional_reduce_with_default_gap() {
        let plan =
            Resize::new(0.5, 0.5, InterpolationKernel::Lanczos3).into_pipeline_nodes(100, 100);
        assert!(matches!(
            plan.nodes.as_slice(),
            [
                ResizeNode::ReduceV { factor: factor_v, kernel: InterpolationKernel::Lanczos3 },
                ResizeNode::ReduceH { factor, kernel: InterpolationKernel::Lanczos3 },
            ] if (*factor - 2.0).abs() < EPSILON && (*factor_v - 2.0).abs() < EPSILON
        ));
        assert_eq!(plan.output_width, 50);
        assert_eq!(plan.output_height, 50);

        let (width, height, _) =
            run_resize_pipeline(100, 100, 0.5, 0.5, InterpolationKernel::Lanczos3);
        assert_eq!(width, 50);
        assert_eq!(height, 50);
    }

    #[test]
    fn uniform_colour_downscale_keeps_pixels_within_rounding_error() {
        let (width, height, output) = run_resize_pipeline_with_pixels(
            8,
            8,
            vec![200_u8; 8 * 8],
            0.5,
            0.5,
            InterpolationKernel::Lanczos3,
        );

        assert_eq!((width, height), (4, 4));
        assert!(
            output.iter().all(|&pixel| pixel.abs_diff(200) <= 1),
            "unexpected output: {output:?}"
        );
    }

    #[test]
    fn checkerboard_downsample_2x2_to_1x1_yields_mid_gray() {
        let (width, height, output) = run_resize_pipeline_with_pixels(
            2,
            2,
            vec![0_u8, 255, 0, 255],
            0.5,
            0.5,
            InterpolationKernel::Lanczos3,
        );

        assert_eq!((width, height), (1, 1));
        assert!(
            matches!(output.as_slice(), [127 | 128]),
            "unexpected output: {output:?}"
        );
    }

    #[test]
    fn plane_gradient_downscale_keeps_predictable_center_value() {
        let pixels = plane_gradient_pixels(6, 6, 20, 10);
        let (width, height, output) =
            run_resize_pipeline_with_pixels(6, 6, pixels, 0.5, 0.5, InterpolationKernel::Lanczos3);

        assert_eq!((width, height), (3, 3));
        assert_eq!(output[4], 75, "unexpected output: {output:?}");
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
    fn checkerboard_downscale_4x4_to_1x1_yields_mid_gray() {
        let pixels = vec![
            0_u8, 255, 0, 255, 255, 0, 255, 0, 0, 255, 0, 255, 255, 0, 255, 0,
        ];
        let (width, height, output) = run_resize_pipeline_with_pixels(
            4,
            4,
            pixels,
            0.25,
            0.25,
            InterpolationKernel::Lanczos3,
        );

        assert_eq!((width, height), (1, 1));
        assert!(
            matches!(output.as_slice(), [127 | 128]),
            "unexpected output: {output:?}"
        );
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
    fn enlarge_2x_reaches_double_size() {
        let plan =
            Resize::new(2.0, 2.0, InterpolationKernel::Nearest).into_pipeline_nodes(100, 100);
        assert_eq!(plan.nodes, vec![ResizeNode::Zoom { xfac: 2, yfac: 2 }]);

        let (width, height, _) =
            run_resize_pipeline(100, 100, 2.0, 2.0, InterpolationKernel::Nearest);
        assert_eq!(width, 200);
        assert_eq!(height, 200);
    }

    #[test]
    fn non_square_resize_scales_axes_independently() {
        let plan =
            Resize::new(0.5, 1.0, InterpolationKernel::Lanczos3).into_pipeline_nodes(100, 200);
        assert!(matches!(
            plan.nodes.as_slice(),
            [ResizeNode::ReduceH { factor, kernel: InterpolationKernel::Lanczos3 }]
                if (*factor - 2.0).abs() < EPSILON
        ));
        assert_eq!(plan.output_width, 50);
        assert_eq!(plan.output_height, 200);

        let (width, height, _) =
            run_resize_pipeline(100, 200, 0.5, 1.0, InterpolationKernel::Lanczos3);
        assert_eq!(width, 50);
        assert_eq!(height, 200);
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

    #[test]
    fn vsqbs_resize_pipeline_matches_reference_2x2_to_4x4_upscale() {
        let (width, height, buffer) = run_resize_pipeline_with_pixels(
            2,
            2,
            vec![0u8, 64, 128, 255],
            2.0,
            2.0,
            InterpolationKernel::Vsqbs,
        );

        assert_eq!((width, height), (4, 4));
        assert_eq!(
            buffer,
            vec![
                57, 96, 115, 118, 124, 169, 198, 201, 159, 214, 245, 249, 164, 219, 251, 255
            ]
        );
    }

    #[test]
    fn downsample_then_upsample_keeps_gradient_error_bounded() {
        let source = gradient_pixels(64, 48);
        let (_, _, downsampled) = run_resize_pipeline_with_pixels(
            64,
            48,
            source.clone(),
            0.5,
            0.5,
            InterpolationKernel::Lanczos3,
        );
        let (_, _, restored) = run_resize_pipeline_with_pixels(
            32,
            24,
            downsampled,
            2.0,
            2.0,
            InterpolationKernel::Lanczos3,
        );

        let max_error = source
            .iter()
            .zip(restored.iter())
            .map(|(expected, got)| expected.abs_diff(*got))
            .max()
            .unwrap_or(0);

        assert!(
            max_error <= 8,
            "round-trip max error too large: {max_error}"
        );
    }

    #[cfg(feature = "lock_instrumentation")]
    #[test]
    fn resize_half_profile() {
        for size in [2048_u32, 8192_u32] {
            let (plan, profile) = resize_half_profile_for_size(size);
            assert_eq!(plan.nodes.len(), profile.nodes.len());

            let node_ns = profile
                .nodes
                .iter()
                .map(|node| node.process_ns)
                .sum::<u128>();
            let tile_overhead_ns = profile
                .tile_execute_ns
                .saturating_sub(profile.source_read_ns + node_ns);
            let scheduler_overhead_ns = profile
                .total_ns
                .saturating_sub(profile.tile_execute_ns + profile.sink_write_ns);

            eprintln!(
                "resize_half_profile size={size} total_ms={:.3} tile_execute_ms={:.3} source_read_ms={:.3} sink_write_ms={:.3} tile_overhead_ms={:.3} scheduler_overhead_ms={:.3} tiles={} locks={}",
                profile.total_ns as f64 / 1_000_000.0,
                profile.tile_execute_ns as f64 / 1_000_000.0,
                profile.source_read_ns as f64 / 1_000_000.0,
                profile.sink_write_ns as f64 / 1_000_000.0,
                tile_overhead_ns as f64 / 1_000_000.0,
                scheduler_overhead_ns as f64 / 1_000_000.0,
                profile.tile_count,
                profile.lock_stats.total_lock_acquisitions,
            );
            for (index, (node, node_profile)) in plan.nodes.iter().zip(&profile.nodes).enumerate() {
                eprintln!(
                    "  node[{index}]={} exec_count={} cache_hits={} process_ms={:.3}",
                    resize_node_name(node),
                    node_profile.exec_count,
                    node_profile.cache_hits,
                    node_profile.process_ns as f64 / 1_000_000.0,
                );
            }
        }
    }

    proptest! {
        #[test]
        fn identity_scale_preserves_exact_pixels(
            width in 1u32..=32,
            height in 1u32..=32,
            pixels in prop::collection::vec(any::<u8>(), 1..=1024),
        ) {
            let expected_len = (width * height) as usize;
            prop_assume!(pixels.len() >= expected_len);
            let input = pixels[..expected_len].to_vec();

            let (out_w, out_h, output) = run_resize_pipeline_with_pixels(
                width,
                height,
                input.clone(),
                1.0,
                1.0,
                InterpolationKernel::Lanczos3,
            );

            prop_assert_eq!((out_w, out_h), (width, height));
            prop_assert_eq!(output, input);
        }

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
