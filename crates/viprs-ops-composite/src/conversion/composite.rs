use std::{any::Any, marker::PhantomData};

use bytemuck::cast_slice;

use viprs_core::{
    error::{CompositeError, ViprsError},
    format::{BandFormat, BandFormatId, F32},
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enumerates the available blend mode values.
pub enum BlendMode {
    /// Uses the `Clear` variant of `BlendMode`.
    Clear,
    /// Uses the `Source` variant of `BlendMode`.
    Source,
    /// Uses the `Over` variant of `BlendMode`.
    Over,
    /// Uses the `In` variant of `BlendMode`.
    In,
    /// Uses the `Out` variant of `BlendMode`.
    Out,
    /// Uses the `Atop` variant of `BlendMode`.
    Atop,
    /// Uses the `Dest` variant of `BlendMode`.
    Dest,
    /// Uses the `DestOver` variant of `BlendMode`.
    DestOver,
    /// Uses the `DestIn` variant of `BlendMode`.
    DestIn,
    /// Uses the `DestOut` variant of `BlendMode`.
    DestOut,
    /// Uses the `DestAtop` variant of `BlendMode`.
    DestAtop,
    /// Uses the `Xor` variant of `BlendMode`.
    Xor,
    /// Uses the `Add` variant of `BlendMode`.
    Add,
    /// Uses the `Saturate` variant of `BlendMode`.
    Saturate,
    /// Uses the `Multiply` variant of `BlendMode`.
    Multiply,
    /// Uses the `Screen` variant of `BlendMode`.
    Screen,
    /// Uses the `Overlay` variant of `BlendMode`.
    Overlay,
    /// Uses the `Darken` variant of `BlendMode`.
    Darken,
    /// Uses the `Lighten` variant of `BlendMode`.
    Lighten,
    /// Uses the `ColourDodge` variant of `BlendMode`.
    ColourDodge,
    /// Uses the `ColourBurn` variant of `BlendMode`.
    ColourBurn,
    /// Uses the `HardLight` variant of `BlendMode`.
    HardLight,
    /// Uses the `SoftLight` variant of `BlendMode`.
    SoftLight,
    /// Uses the `Difference` variant of `BlendMode`.
    Difference,
    /// Uses the `Exclusion` variant of `BlendMode`.
    Exclusion,
    /// Uses the `Hue` variant of `BlendMode`.
    Hue,
    /// Uses the `Saturation` variant of `BlendMode`.
    Saturation,
    /// Uses the `Colour` variant of `BlendMode`.
    Colour,
    /// Uses the `Luminosity` variant of `BlendMode`.
    Luminosity,
}

/// Two-input Porter-Duff / PDF compositing op.
///
/// Slot 0 is the base image, slot 1 is the overlay image. Both inputs must have
/// the same format, geometry, and band count, with alpha in the last band.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::composite::CompositeOp;
///
/// let op = CompositeOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct CompositeOp<F: BandFormat> {
    /// Stores the `mode` value for this item.
    pub mode: BlendMode,
    /// Stores the `premultiplied` value for this item.
    pub premultiplied: bool,
    /// Number of bands associated with this item.
    pub bands: u32,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> CompositeOp<F> {
    /// Creates a new `CompositeOp`.
    pub fn new(mode: BlendMode, premultiplied: bool, bands: u32) -> Result<Self, ViprsError> {
        validate_composite_bands(mode, bands)?;

        Ok(Self {
            mode,
            premultiplied,
            bands,
            _phantom: PhantomData,
        })
    }
}

const fn blend_mode_name(mode: BlendMode) -> &'static str {
    match mode {
        BlendMode::Clear => "clear",
        BlendMode::Source => "source",
        BlendMode::Over => "over",
        BlendMode::In => "in",
        BlendMode::Out => "out",
        BlendMode::Atop => "atop",
        BlendMode::Dest => "dest",
        BlendMode::DestOver => "dest-over",
        BlendMode::DestIn => "dest-in",
        BlendMode::DestOut => "dest-out",
        BlendMode::DestAtop => "dest-atop",
        BlendMode::Xor => "xor",
        BlendMode::Add => "add",
        BlendMode::Saturate => "saturate",
        BlendMode::Multiply => "multiply",
        BlendMode::Screen => "screen",
        BlendMode::Overlay => "overlay",
        BlendMode::Darken => "darken",
        BlendMode::Lighten => "lighten",
        BlendMode::ColourDodge => "colour-dodge",
        BlendMode::ColourBurn => "colour-burn",
        BlendMode::HardLight => "hard-light",
        BlendMode::SoftLight => "soft-light",
        BlendMode::Difference => "difference",
        BlendMode::Exclusion => "exclusion",
        BlendMode::Hue => "hue",
        BlendMode::Saturation => "saturation",
        BlendMode::Colour => "colour",
        BlendMode::Luminosity => "luminosity",
    }
}

fn validate_composite_bands(mode: BlendMode, bands: u32) -> Result<(), ViprsError> {
    if is_non_separable_mode(mode) && bands != 4 {
        return Err(CompositeError::NonSeparableModeRequiresRgba {
            mode: blend_mode_name(mode),
            bands,
        }
        .into());
    }

    Ok(())
}

fn alpha_out(mode: BlendMode, alpha_src: f32, alpha_dst: f32) -> f32 {
    match mode {
        BlendMode::Clear => 0.0,
        BlendMode::Source | BlendMode::DestAtop => alpha_src,
        BlendMode::Over
        | BlendMode::Multiply
        | BlendMode::Screen
        | BlendMode::Overlay
        | BlendMode::Darken
        | BlendMode::Lighten
        | BlendMode::ColourDodge
        | BlendMode::ColourBurn
        | BlendMode::HardLight
        | BlendMode::SoftLight
        | BlendMode::Difference
        | BlendMode::Exclusion
        | BlendMode::Hue
        | BlendMode::Saturation
        | BlendMode::Colour
        | BlendMode::Luminosity => alpha_dst.mul_add(1.0 - alpha_src, alpha_src),
        BlendMode::In | BlendMode::DestIn => alpha_src * alpha_dst,
        BlendMode::Out => alpha_src * (1.0 - alpha_dst),
        BlendMode::Atop | BlendMode::Dest => alpha_dst,
        BlendMode::DestOver => alpha_src.mul_add(1.0 - alpha_dst, alpha_dst),
        BlendMode::DestOut => alpha_dst * (1.0 - alpha_src),
        BlendMode::Xor => (2.0 * alpha_src).mul_add(-alpha_dst, alpha_src + alpha_dst),
        BlendMode::Add | BlendMode::Saturate => (alpha_src + alpha_dst).min(1.0),
    }
}

const fn is_non_separable_mode(mode: BlendMode) -> bool {
    matches!(
        mode,
        BlendMode::Hue | BlendMode::Saturation | BlendMode::Colour | BlendMode::Luminosity
    )
}

fn composite_lum(rgb: [f32; 3]) -> f32 {
    0.11f32.mul_add(rgb[2], 0.59f32.mul_add(rgb[1], 0.3 * rgb[0]))
}

fn composite_sat(rgb: [f32; 3]) -> f32 {
    rgb[0].max(rgb[1]).max(rgb[2]) - rgb[0].min(rgb[1]).min(rgb[2])
}

fn composite_clip_colour(rgb: &mut [f32; 3]) {
    let lum = composite_lum(*rgb);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let max = rgb[0].max(rgb[1]).max(rgb[2]);

    if min < 0.0 {
        for channel in rgb.iter_mut() {
            *channel = lum + (*channel - lum) * lum / (lum - min);
        }
    }

    if max > 1.0 {
        for channel in rgb.iter_mut() {
            *channel = lum + (*channel - lum) * (1.0 - lum) / (max - lum);
        }
    }
}

fn composite_set_lum(rgb: &mut [f32; 3], lum: f32) {
    let delta = lum - composite_lum(*rgb);
    for channel in rgb.iter_mut() {
        *channel += delta;
    }
    composite_clip_colour(rgb);
}

fn composite_set_sat(rgb: &mut [f32; 3], sat: f32) {
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let max = rgb[0].max(rgb[1]).max(rgb[2]);

    if max > min {
        for channel in rgb.iter_mut() {
            if *channel == max {
                *channel = sat;
            } else if *channel == min {
                *channel = 0.0;
            } else {
                *channel = (*channel - min) * sat / (max - min);
            }
        }
    } else {
        *rgb = [0.0; 3];
    }
}

fn unpremultiply_sample(sample: f32, alpha: f32, premultiplied: bool) -> f32 {
    if premultiplied {
        if alpha > 0.0 { sample / alpha } else { 0.0 }
    } else {
        sample
    }
}

fn composite_non_separable(
    mode: BlendMode,
    src_pre: [f32; 3],
    dst_pre: [f32; 3],
    src: [f32; 3],
    dst: [f32; 3],
    alpha_src: f32,
    alpha_dst: f32,
) -> [f32; 3] {
    let mut blended = match mode {
        BlendMode::Hue | BlendMode::Colour => src,
        _ => dst,
    };

    match mode {
        BlendMode::Hue => {
            composite_set_sat(&mut blended, composite_sat(dst));
            composite_set_lum(&mut blended, composite_lum(dst));
        }
        BlendMode::Saturation => {
            composite_set_sat(&mut blended, composite_sat(src));
            composite_set_lum(&mut blended, composite_lum(dst));
        }
        BlendMode::Colour => {
            composite_set_lum(&mut blended, composite_lum(dst));
        }
        BlendMode::Luminosity => {
            composite_set_lum(&mut blended, composite_lum(src));
        }
        _ => {}
    }

    let t1 = 1.0 - alpha_dst;
    let t2 = 1.0 - alpha_src;
    let t3 = alpha_src * alpha_dst;

    [
        t3.mul_add(blended[0], dst_pre[0].mul_add(t2, src_pre[0] * t1)),
        t3.mul_add(blended[1], dst_pre[1].mul_add(t2, src_pre[1] * t1)),
        t3.mul_add(blended[2], dst_pre[2].mul_add(t2, src_pre[2] * t1)),
    ]
}

fn pdf_blend(mode: BlendMode, src: f32, dst: f32) -> f32 {
    match mode {
        BlendMode::Multiply => src * dst,
        BlendMode::Screen => src.mul_add(-dst, src + dst),
        BlendMode::Overlay => {
            if dst <= 0.5 {
                2.0 * src * dst
            } else {
                (2.0 * (1.0 - src)).mul_add(-(1.0 - dst), 1.0)
            }
        }
        BlendMode::Darken => src.min(dst),
        BlendMode::Lighten => src.max(dst),
        BlendMode::ColourDodge => {
            if src < 1.0 {
                (dst / (1.0 - src)).min(1.0)
            } else {
                1.0
            }
        }
        BlendMode::ColourBurn => {
            if src > 0.0 {
                1.0 - ((1.0 - dst) / src).min(1.0)
            } else {
                0.0
            }
        }
        BlendMode::HardLight => {
            if src <= 0.5 {
                2.0 * src * dst
            } else {
                (2.0 * (1.0 - src)).mul_add(-(1.0 - dst), 1.0)
            }
        }
        BlendMode::SoftLight => {
            let g = if dst <= 0.25 {
                16.0f32.mul_add(dst, -12.0).mul_add(dst, 4.0) * dst
            } else {
                dst.sqrt()
            };

            if src <= 0.5 {
                (2.0f32.mul_add(-src, 1.0) * dst).mul_add(-(1.0 - dst), dst)
            } else {
                2.0f32.mul_add(src, -1.0).mul_add(g - dst, dst)
            }
        }
        BlendMode::Difference => (dst - src).abs(),
        BlendMode::Exclusion => (2.0 * src).mul_add(-dst, src + dst),
        _ => src,
    }
}

#[allow(clippy::unreachable)]
// REASON: non-separable blend modes are dispatched through the tuple path before this helper runs.
fn composite_channel(
    mode: BlendMode,
    src_pre: f32,
    dst_pre: f32,
    src: f32,
    dst: f32,
    alpha_src: f32,
    alpha_dst: f32,
) -> f32 {
    match mode {
        BlendMode::Clear => 0.0,
        BlendMode::Source => src_pre,
        BlendMode::Over => dst_pre.mul_add(1.0 - alpha_src, src_pre),
        BlendMode::In => src_pre * alpha_dst,
        BlendMode::Out => src_pre * (1.0 - alpha_dst),
        BlendMode::Atop => dst_pre.mul_add(1.0 - alpha_src, src_pre * alpha_dst),
        BlendMode::Dest => dst_pre,
        BlendMode::DestOver => src_pre.mul_add(1.0 - alpha_dst, dst_pre),
        BlendMode::DestIn => dst_pre * alpha_src,
        BlendMode::DestOut => dst_pre * (1.0 - alpha_src),
        BlendMode::DestAtop => dst_pre.mul_add(alpha_src, src_pre * (1.0 - alpha_dst)),
        BlendMode::Xor => dst_pre.mul_add(1.0 - alpha_src, src_pre * (1.0 - alpha_dst)),
        BlendMode::Add => src_pre + dst_pre,
        BlendMode::Saturate => src_pre.mul_add(alpha_src.min(1.0 - alpha_dst), dst_pre),
        BlendMode::Multiply
        | BlendMode::Screen
        | BlendMode::Overlay
        | BlendMode::Darken
        | BlendMode::Lighten
        | BlendMode::ColourDodge
        | BlendMode::ColourBurn
        | BlendMode::HardLight
        | BlendMode::SoftLight
        | BlendMode::Difference
        | BlendMode::Exclusion => (alpha_src * alpha_dst).mul_add(
            pdf_blend(mode, src, dst),
            dst_pre.mul_add(1.0 - alpha_src, src_pre * (1.0 - alpha_dst)),
        ),
        BlendMode::Hue | BlendMode::Saturation | BlendMode::Colour | BlendMode::Luminosity => {
            unreachable!("non-separable composite modes are handled as RGB tuples")
        }
    }
}

#[inline(always)]
fn composite_over_rgba_unpremultiplied(base: &[f32], overlay: &[f32], out: &mut [f32]) {
    debug_assert_eq!(base.len(), overlay.len());
    debug_assert_eq!(base.len(), out.len());
    debug_assert_eq!(base.len() % 4, 0);

    for offset in (0..base.len()).step_by(4) {
        let alpha_dst = base[offset + 3].clamp(0.0, 1.0);
        let alpha_src = overlay[offset + 3].clamp(0.0, 1.0);
        let inv_alpha_src = 1.0 - alpha_src;
        let alpha_result = alpha_src + alpha_dst * inv_alpha_src;

        if alpha_result > 0.0 {
            let src_scale = alpha_src / alpha_result;
            let dst_scale = alpha_dst * inv_alpha_src / alpha_result;
            out[offset] = base[offset].mul_add(dst_scale, overlay[offset] * src_scale);
            out[offset + 1] = base[offset + 1].mul_add(dst_scale, overlay[offset + 1] * src_scale);
            out[offset + 2] = base[offset + 2].mul_add(dst_scale, overlay[offset + 2] * src_scale);
        } else {
            out[offset] = 0.0;
            out[offset + 1] = 0.0;
            out[offset + 2] = 0.0;
        }

        out[offset + 3] = alpha_result;
    }
}

#[inline(always)]
fn composite_over_rgba_premultiplied(base: &[f32], overlay: &[f32], out: &mut [f32]) {
    debug_assert_eq!(base.len(), overlay.len());
    debug_assert_eq!(base.len(), out.len());
    debug_assert_eq!(base.len() % 4, 0);

    for offset in (0..base.len()).step_by(4) {
        let alpha_dst = base[offset + 3].clamp(0.0, 1.0);
        let alpha_src = overlay[offset + 3].clamp(0.0, 1.0);
        let inv_alpha_src = 1.0 - alpha_src;

        out[offset] = base[offset].mul_add(inv_alpha_src, overlay[offset]);
        out[offset + 1] = base[offset + 1].mul_add(inv_alpha_src, overlay[offset + 1]);
        out[offset + 2] = base[offset + 2].mul_add(inv_alpha_src, overlay[offset + 2]);
        out[offset + 3] = alpha_src + alpha_dst * inv_alpha_src;
    }
}

impl DynOperation for CompositeOp<F32> {
    fn input_format(&self) -> BandFormatId {
        BandFormatId::F32
    }

    fn output_format(&self) -> BandFormatId {
        BandFormatId::F32
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn input_slot_count(&self) -> usize {
        2
    }

    fn required_input_region_slot(&self, output: &Region, _slot: usize) -> Region {
        *output
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        _input: &[u8],
        _output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        debug_assert!(
            false,
            "CompositeOp: dyn_process_region called on a 2-input node — use dyn_process_region_multi"
        );
    }

    #[inline]
    fn dyn_process_region_multi(
        &self,
        _state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        input_regions: &[Region],
        output_region: Region,
    ) {
        debug_assert_eq!(inputs.len(), 2, "CompositeOp expects exactly 2 inputs");
        debug_assert_eq!(
            input_regions.len(),
            2,
            "CompositeOp expects exactly 2 input regions"
        );

        let Some(&base_bytes) = inputs.first() else {
            return;
        };
        let Some(&overlay_bytes) = inputs.get(1) else {
            return;
        };

        let base = cast_slice::<u8, f32>(base_bytes);
        let overlay = cast_slice::<u8, f32>(overlay_bytes);
        let out = bytemuck::cast_slice_mut::<u8, f32>(output);
        let bands = self.bands as usize;

        debug_assert!(bands >= 2, "CompositeOp requires an alpha band");
        debug_assert_eq!(
            input_regions[0], output_region,
            "CompositeOp currently requires aligned base tiles"
        );
        debug_assert_eq!(
            input_regions[1], output_region,
            "CompositeOp currently requires aligned overlay tiles"
        );

        let pixel_count = output_region.pixel_count();
        let alpha_band = bands - 1;
        debug_assert_eq!(base.len(), pixel_count * bands);
        debug_assert_eq!(overlay.len(), pixel_count * bands);
        debug_assert_eq!(out.len(), pixel_count * bands);
        debug_assert!(
            !is_non_separable_mode(self.mode) || alpha_band == 3,
            "non-separable composite modes require RGB plus alpha"
        );

        if self.mode == BlendMode::Over && bands == 4 {
            if self.premultiplied {
                composite_over_rgba_premultiplied(base, overlay, out);
            } else {
                composite_over_rgba_unpremultiplied(base, overlay, out);
            }
            return;
        }

        for px in 0..pixel_count {
            let offset = px * bands;
            let alpha_dst = base[offset + alpha_band].clamp(0.0, 1.0);
            let alpha_src = overlay[offset + alpha_band].clamp(0.0, 1.0);
            let alpha_result = alpha_out(self.mode, alpha_src, alpha_dst);

            if is_non_separable_mode(self.mode) && alpha_band == 3 {
                let mut src_pre = [0.0; 3];
                let mut dst_pre = [0.0; 3];
                let mut src = [0.0; 3];
                let mut dst = [0.0; 3];

                for channel in 0..3 {
                    let dst_sample = base[offset + channel];
                    let src_sample = overlay[offset + channel];
                    dst_pre[channel] = if self.premultiplied {
                        dst_sample
                    } else {
                        dst_sample * alpha_dst
                    };
                    src_pre[channel] = if self.premultiplied {
                        src_sample
                    } else {
                        src_sample * alpha_src
                    };
                    dst[channel] = unpremultiply_sample(dst_pre[channel], alpha_dst, true);
                    src[channel] = unpremultiply_sample(src_pre[channel], alpha_src, true);
                }

                let out_pre = composite_non_separable(
                    self.mode, src_pre, dst_pre, src, dst, alpha_src, alpha_dst,
                );
                for channel in 0..3 {
                    out[offset + channel] = if self.premultiplied {
                        out_pre[channel]
                    } else if alpha_result > 0.0 {
                        out_pre[channel] / alpha_result
                    } else {
                        0.0
                    };
                }
                out[offset + alpha_band] = alpha_result;
                continue;
            }

            for channel in 0..alpha_band {
                let dst_sample = base[offset + channel];
                let src_sample = overlay[offset + channel];
                let dst_pre = if self.premultiplied {
                    dst_sample
                } else {
                    dst_sample * alpha_dst
                };
                let src_pre = if self.premultiplied {
                    src_sample
                } else {
                    src_sample * alpha_src
                };
                let dst = unpremultiply_sample(dst_pre, alpha_dst, true);
                let src = unpremultiply_sample(src_pre, alpha_src, true);
                let out_pre =
                    composite_channel(self.mode, src_pre, dst_pre, src, dst, alpha_src, alpha_dst);

                out[offset + channel] = if self.premultiplied {
                    out_pre
                } else if alpha_result > 0.0 {
                    out_pre / alpha_result
                } else {
                    0.0
                };
            }

            out[offset + alpha_band] = alpha_result;
        }
    }
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use crate::conversion::copy::CopyOp;
    use proptest::prelude::*;
    use proptest::proptest;
    use std::{fs, path::Path, process::Command};
    use viprs_core::op::OperationBridge;
    use viprs_ports::{scheduler::TileScheduler, source::DynImageSource};
    use viprs_runtime::{
        pipeline::PipelineArena, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    };

    fn run_composite(
        mode: BlendMode,
        premultiplied: bool,
        bands: u32,
        base: &[f32],
        overlay: &[f32],
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        let op = CompositeOp::<F32>::new(mode, premultiplied, bands).unwrap();
        let region = Region::new(0, 0, width, height);
        let mut output = vec![0.0f32; base.len()];
        let base_bytes = bytemuck::cast_slice(base);
        let overlay_bytes = bytemuck::cast_slice(overlay);
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            &[base_bytes, overlay_bytes],
            bytemuck::cast_slice_mut(&mut output),
            &[region, region],
            region,
        );
        output
    }

    #[test]
    fn over_with_fully_opaque_overlay_replaces_base() {
        let base = vec![0.2, 0.3, 0.4, 0.8];
        let overlay = vec![0.9, 0.1, 0.2, 1.0];
        let output = run_composite(BlendMode::Over, false, 4, &base, &overlay, 1, 1);
        assert_eq!(output, overlay);
    }

    #[test]
    fn over_with_fully_transparent_overlay_keeps_base() {
        let base = vec![0.2, 0.3, 0.4, 0.8];
        let overlay = vec![0.9, 0.1, 0.2, 0.0];
        let output = run_composite(BlendMode::Over, false, 4, &base, &overlay, 1, 1);
        for (actual, expected) in output.iter().zip(base.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn over_with_half_alpha_blends_expected_pixel() {
        let base = vec![0.2, 0.4, 0.6, 0.5];
        let overlay = vec![0.8, 0.2, 0.4, 0.5];
        let output = run_composite(BlendMode::Over, false, 4, &base, &overlay, 1, 1);
        let alpha = 0.5f32.mul_add(1.0 - 0.5, 0.5);
        assert!((output[3] - alpha).abs() < 1e-6);
        assert!((output[0] - 0.6).abs() < 1e-6);
        assert!((output[1] - 0.266_666_68).abs() < 1e-6);
        assert!((output[2] - 0.466_666_67).abs() < 1e-6);
    }

    #[test]
    fn multiply_white_over_white_stays_white() {
        let base = vec![1.0, 1.0, 1.0, 1.0];
        let overlay = vec![1.0, 1.0, 1.0, 1.0];
        let output = run_composite(BlendMode::Multiply, false, 4, &base, &overlay, 1, 1);
        assert_eq!(output, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn multiply_black_over_white_is_black() {
        let base = vec![1.0, 1.0, 1.0, 1.0];
        let overlay = vec![0.0, 0.0, 0.0, 1.0];
        let output = run_composite(BlendMode::Multiply, false, 4, &base, &overlay, 1, 1);
        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn composite_runs_end_to_end_through_compiled_pipeline() {
        let pixels = vec![
            0.1, 0.2, 0.3, 1.0, //
            0.4, 0.5, 0.6, 1.0,
        ];
        let source = MemorySource::<F32>::new(2, 1, 4, pixels.clone()).unwrap();

        let mut arena = PipelineArena::with_source(Box::new(source) as Box<dyn DynImageSource>);
        let base = arena.add_node(Box::new(OperationBridge::new_pixel_local(
            CopyOp::<F32>::default(),
            4,
        )));
        let overlay = arena.add_node(Box::new(OperationBridge::new_pixel_local(
            CopyOp::<F32>::default(),
            4,
        )));
        let composite = arena.add_node(Box::new(
            CompositeOp::<F32>::new(BlendMode::Over, false, 4).unwrap(),
        ));

        arena.connect(base, overlay).unwrap();
        arena.connect(base, composite).unwrap();
        arena.connect_to_slot(overlay, composite, 1).unwrap();

        let pipeline = arena.compile().unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(1)
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();
        let raw = sink.into_buffer();
        let result = bytemuck::cast_slice::<u8, f32>(&raw).to_vec();

        assert_eq!(result, pixels);
    }

    #[test]
    fn metadata_reports_two_input_rgba_configuration() {
        let op = CompositeOp::<F32>::new(BlendMode::Screen, true, 4).unwrap();
        let region = Region::new(3, 4, 5, 6);

        assert_eq!(op.input_format(), BandFormatId::F32);
        assert_eq!(op.output_format(), BandFormatId::F32);
        assert_eq!(op.bands(), 4);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.input_slot_count(), 2);
        assert_eq!(op.required_input_region_slot(&region, 0), region);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(5, 6), NodeSpec::identity(5, 6));
    }

    #[test]
    fn alpha_out_and_pdf_blend_cover_all_modes() {
        let src_alpha = 0.25;
        let dst_alpha = 0.5;
        let src = 0.2;
        let dst = 0.8;
        let modes = [
            BlendMode::Clear,
            BlendMode::Source,
            BlendMode::Over,
            BlendMode::In,
            BlendMode::Out,
            BlendMode::Atop,
            BlendMode::Dest,
            BlendMode::DestOver,
            BlendMode::DestIn,
            BlendMode::DestOut,
            BlendMode::DestAtop,
            BlendMode::Xor,
            BlendMode::Add,
            BlendMode::Saturate,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
            BlendMode::ColourDodge,
            BlendMode::ColourBurn,
            BlendMode::HardLight,
            BlendMode::SoftLight,
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::Hue,
            BlendMode::Saturation,
            BlendMode::Colour,
            BlendMode::Luminosity,
        ];

        for mode in modes {
            let alpha = alpha_out(mode, src_alpha, dst_alpha);
            assert!((0.0..=1.0).contains(&alpha), "{mode:?} produced {alpha}");
            if is_non_separable_mode(mode) {
                let blended = composite_non_separable(
                    mode,
                    [src * src_alpha; 3],
                    [dst * dst_alpha; 3],
                    [src; 3],
                    [dst; 3],
                    src_alpha,
                    dst_alpha,
                );
                assert!(blended.iter().all(|channel| channel.is_finite()));
            } else {
                if matches!(
                    mode,
                    BlendMode::Multiply
                        | BlendMode::Screen
                        | BlendMode::Overlay
                        | BlendMode::Darken
                        | BlendMode::Lighten
                        | BlendMode::ColourDodge
                        | BlendMode::ColourBurn
                        | BlendMode::HardLight
                        | BlendMode::SoftLight
                        | BlendMode::Difference
                        | BlendMode::Exclusion
                ) {
                    let blended = pdf_blend(mode, src, dst);
                    assert!(blended.is_finite(), "{mode:?} produced non-finite blend");
                }

                let channel = composite_channel(
                    mode,
                    src * src_alpha,
                    dst * dst_alpha,
                    src,
                    dst,
                    src_alpha,
                    dst_alpha,
                );
                assert!(channel.is_finite(), "{mode:?} produced non-finite channel");
            }
        }
    }

    #[test]
    fn premultiplied_over_keeps_premultiplied_channels() {
        let base = vec![0.1, 0.2, 0.3, 0.5];
        let overlay = vec![0.4, 0.1, 0.2, 0.5];
        let output = run_composite(BlendMode::Over, true, 4, &base, &overlay, 1, 1);

        assert!((output[0] - 0.45).abs() < 1e-6);
        assert!((output[1] - 0.2).abs() < 1e-6);
        assert!((output[2] - 0.35).abs() < 1e-6);
        assert!((output[3] - 0.75).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn source_with_opaque_overlay_is_identity(
            base_rgb in prop::collection::vec(0.0f32..=1.0, 3),
            overlay_rgb in prop::collection::vec(0.0f32..=1.0, 3),
            base_alpha in 0.0f32..=1.0,
        ) {
            let base = vec![base_rgb[0], base_rgb[1], base_rgb[2], base_alpha];
            let overlay = vec![overlay_rgb[0], overlay_rgb[1], overlay_rgb[2], 1.0];

            let output = run_composite(BlendMode::Source, false, 4, &base, &overlay, 1, 1);

            prop_assert_eq!(output, overlay);
        }

        #[test]
        fn hue_modes_are_identity_when_pixels_match(
            pixel in prop::collection::vec(0.0f32..=1.0, 3),
        ) {
            let rgba = vec![pixel[0], pixel[1], pixel[2], 1.0];

            for mode in [
                BlendMode::Hue,
                BlendMode::Saturation,
                BlendMode::Colour,
                BlendMode::Luminosity,
            ] {
                let output = run_composite(mode, false, 4, &rgba, &rgba, 1, 1);
                for (actual, expected) in output.iter().zip(rgba.iter()) {
                    prop_assert!((actual - expected).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn clear_zeroes_pixel_at_boundary_values() {
        let base = vec![0.0, 1.0, 0.5, 1.0];
        let overlay = vec![1.0, 0.0, 1.0, 1.0];

        let output = run_composite(BlendMode::Clear, false, 4, &base, &overlay, 1, 1);

        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn source_with_transparent_overlay_outputs_transparent_black() {
        let base = vec![0.3, 0.6, 0.9, 1.0];
        let overlay = vec![1.0, 0.0, 0.5, 0.0];

        let output = run_composite(BlendMode::Source, false, 4, &base, &overlay, 1, 1);

        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn hue_with_greyscale_base_preserves_base() {
        let base = vec![0.4, 0.4, 0.4, 1.0];
        let overlay = vec![0.9, 0.2, 0.1, 1.0];

        let output = run_composite(BlendMode::Hue, false, 4, &base, &overlay, 1, 1);

        assert_eq!(output, base);
    }

    fn write_pam_rgba(path: &std::path::Path, pixel: [u8; 4]) {
        let header = b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n";
        let mut bytes = header.to_vec();
        bytes.extend_from_slice(&pixel);
        fs::write(path, bytes).unwrap();
    }

    fn assert_mode_matches_vips_for_single_rgba_pixel_when_available(
        mode: BlendMode,
        base: [u8; 4],
        overlay: [u8; 4],
    ) {
        if std::env::var_os("VIPRS_RUN_VIPS_COMPOSITE_GOLDENS").is_none() {
            return;
        }

        let vips = Path::new("/opt/homebrew/bin/vips");
        if !vips.exists() {
            return;
        }

        let workdir = Path::new("target/composite-golden");
        fs::create_dir_all(workdir).unwrap();
        let base_path = workdir.join("base.pam");
        let overlay_path = workdir.join("overlay.pam");
        let out_v_path = workdir.join("out.v");
        let out_raw_path = workdir.join("out.raw");

        write_pam_rgba(&base_path, base);
        write_pam_rgba(&overlay_path, overlay);

        let output = Command::new(vips)
            .args([
                "composite2",
                base_path.to_str().unwrap(),
                overlay_path.to_str().unwrap(),
                out_v_path.to_str().unwrap(),
                blend_mode_name(mode),
            ])
            .output()
            .unwrap();
        if !output.status.success() {
            return;
        }

        let rawsave = Command::new(vips)
            .args([
                "rawsave",
                out_v_path.to_str().unwrap(),
                out_raw_path.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            rawsave.status.success(),
            "vips rawsave failed: {}",
            String::from_utf8_lossy(&rawsave.stderr)
        );

        let expected: [u8; 4] = fs::read(&out_raw_path).unwrap()[..4].try_into().unwrap();
        let actual = run_composite(
            mode,
            false,
            4,
            &base.iter().map(|v| *v as f32 / 255.0).collect::<Vec<_>>(),
            &overlay
                .iter()
                .map(|v| *v as f32 / 255.0)
                .collect::<Vec<_>>(),
            1,
            1,
        )
        .into_iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect::<Vec<_>>();

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (*actual as i16 - *expected as i16).abs() <= 1,
                "mode {mode:?} channel mismatch: actual={actual}, expected={expected}"
            );
        }

        let _ = fs::remove_file(base_path);
        let _ = fs::remove_file(overlay_path);
        let _ = fs::remove_file(out_v_path);
        let _ = fs::remove_file(out_raw_path);
    }

    #[test]
    fn over_matches_vips_for_single_rgba_pixel_when_available() {
        let base = [64u8, 128, 192, 128];
        let overlay = [255u8, 0, 0, 128];
        assert_mode_matches_vips_for_single_rgba_pixel_when_available(
            BlendMode::Over,
            base,
            overlay,
        );
    }

    #[test]
    fn new_modes_match_vips_for_single_rgba_pixel_when_available() {
        let base = [64u8, 128, 192, 200];
        let overlay = [220u8, 32, 96, 180];

        for mode in [
            BlendMode::Clear,
            BlendMode::Source,
            BlendMode::Hue,
            BlendMode::Saturation,
            BlendMode::Colour,
            BlendMode::Luminosity,
        ] {
            assert_mode_matches_vips_for_single_rgba_pixel_when_available(mode, base, overlay);
        }
    }

    #[test]
    fn non_separable_modes_reject_greyscale_alpha_layouts() {
        for mode in [
            BlendMode::Hue,
            BlendMode::Saturation,
            BlendMode::Colour,
            BlendMode::Luminosity,
        ] {
            match CompositeOp::<F32>::new(mode, false, 2) {
                Err(ViprsError::Composite(CompositeError::NonSeparableModeRequiresRgba {
                    mode: actual_mode,
                    bands,
                })) => {
                    assert_eq!(actual_mode, blend_mode_name(mode));
                    assert_eq!(bands, 2);
                }
                Err(other) => panic!("unexpected error for {mode:?}: {other}"),
                Ok(_) => panic!("non-separable modes must reject greyscale+alpha input"),
            }
        }
    }

    #[test]
    fn composite_chaos_non_separable_modes_accept_nan_rgba_without_panicking() {
        let base = vec![f32::NAN, 0.25, 0.75, 1.0];
        let overlay = vec![0.5, f32::NAN, 0.1, 1.0];

        for mode in [
            BlendMode::Hue,
            BlendMode::Saturation,
            BlendMode::Colour,
            BlendMode::Luminosity,
        ] {
            let output = run_composite(mode, false, 4, &base, &overlay, 1, 1);
            assert_eq!(output.len(), 4);
            assert_eq!(output[3], 1.0, "alpha should stay opaque for {mode:?}");
        }
    }
}
