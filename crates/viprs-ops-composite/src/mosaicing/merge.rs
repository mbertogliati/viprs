use std::{any::Any, marker::PhantomData};

use bytemuck::Pod;

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
    shared_ops::sample_conv::{FromF64, ToF64},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enumerates the available merge direction values.
pub enum MergeDirection {
    /// Uses the `Horizontal` variant of `MergeDirection`.
    Horizontal,
    /// Uses the `Vertical` variant of `MergeDirection`.
    Vertical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeAxis {
    Horizontal,
    Vertical,
}

impl From<MergeDirection> for MergeAxis {
    fn from(direction: MergeDirection) -> Self {
        match direction {
            MergeDirection::Horizontal => Self::Horizontal,
            MergeDirection::Vertical => Self::Vertical,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MergeLayout {
    ref_left: i32,
    ref_top: i32,
    sec_left: i32,
    sec_top: i32,
    output_width: u32,
    output_height: u32,
    overlap_left: i32,
    overlap_top: i32,
    overlap_width: u32,
    overlap_height: u32,
}

impl MergeLayout {
    const fn overlap_contains(self, x: i32, y: i32) -> bool {
        x >= self.overlap_left
            && y >= self.overlap_top
            && x < self.overlap_left + self.overlap_width as i32
            && y < self.overlap_top + self.overlap_height as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlendClass {
    Left,
    Blend,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SeamRange {
    first: i32,
    last: i32,
}

impl SeamRange {
    const fn classify(self, pos: i32) -> BlendClass {
        if pos < self.first {
            BlendClass::Left
        } else if pos >= self.last {
            BlendClass::Right
        } else {
            BlendClass::Blend
        }
    }

    const fn width(self) -> i32 {
        self.last - self.first
    }

    fn weight_ref(self, pos: i32) -> f64 {
        let width = f64::from(self.width().max(1));
        let phase = (f64::from(pos - self.first) / width).clamp(0.0, 1.0);
        1.0 - phase
    }
}

struct MergeCommon<F: BandFormat> {
    axis: MergeAxis,
    ref_width: u32,
    ref_height: u32,
    sec_width: u32,
    sec_height: u32,
    dx: i32,
    dy: i32,
    blend_width: u32,
    bands: u32,
    _format: PhantomData<F>,
}
impl<F: BandFormat> MergeCommon<F> {
    fn layout(&self) -> MergeLayout {
        let ref_left = 0;
        let ref_top = 0;
        let sec_left = -self.dx;
        let sec_top = -self.dy;

        let union_left = ref_left.min(sec_left);
        let union_top = ref_top.min(sec_top);
        let union_right = (ref_left + self.ref_width as i32).max(sec_left + self.sec_width as i32);
        let union_bottom = (ref_top + self.ref_height as i32).max(sec_top + self.sec_height as i32);

        let ref_left = ref_left - union_left;
        let ref_top = ref_top - union_top;
        let sec_left = sec_left - union_left;
        let sec_top = sec_top - union_top;

        let overlap_left = ref_left.max(sec_left);
        let overlap_top = ref_top.max(sec_top);
        let overlap_right =
            (ref_left + self.ref_width as i32).min(sec_left + self.sec_width as i32);
        let overlap_bottom =
            (ref_top + self.ref_height as i32).min(sec_top + self.sec_height as i32);

        MergeLayout {
            ref_left,
            ref_top,
            sec_left,
            sec_top,
            output_width: (union_right - union_left) as u32,
            output_height: (union_bottom - union_top) as u32,
            overlap_left,
            overlap_top,
            overlap_width: overlap_right.saturating_sub(overlap_left) as u32,
            overlap_height: overlap_bottom.saturating_sub(overlap_top) as u32,
        }
    }

    fn output_width(&self) -> u32 {
        self.layout().output_width
    }

    fn output_height(&self) -> u32 {
        self.layout().output_height
    }

    fn placement(&self, slot: usize) -> (i32, i32, u32, u32) {
        let layout = self.layout();
        match slot {
            0 => (
                layout.ref_left,
                layout.ref_top,
                self.ref_width,
                self.ref_height,
            ),
            1 => (
                layout.sec_left,
                layout.sec_top,
                self.sec_width,
                self.sec_height,
            ),
            _ => (0, 0, 0, 0),
        }
    }

    #[inline]
    fn process_region_typed(
        &self,
        ref_input: &[F::Sample],
        sec_input: &[F::Sample],
        output: &mut [F::Sample],
        input_regions: &[Region],
        output_region: Region,
    ) where
        F::Sample: Copy + Pod + ToF64 + FromF64,
    {
        let layout = self.layout();
        let bands = self.bands as usize;
        match self.axis {
            MergeAxis::Horizontal => {
                let out_width = output_region.width as usize;
                for row in 0..output_region.height as usize {
                    let y = output_region.y + row as i32;
                    let seam = seam_range_for_row(
                        ref_input,
                        sec_input,
                        input_regions,
                        layout,
                        y,
                        bands,
                        self.blend_width,
                    );
                    for col in 0..output_region.width as usize {
                        let x = output_region.x + col as i32;
                        let out_base = (row * out_width + col) * bands;
                        self.process_output_pixel(
                            ref_input,
                            sec_input,
                            output,
                            input_regions,
                            layout,
                            seam,
                            x,
                            y,
                            out_base,
                            bands,
                        );
                    }
                }
            }
            MergeAxis::Vertical => {
                let out_width = output_region.width as usize;
                for col in 0..output_region.width as usize {
                    let x = output_region.x + col as i32;
                    let seam = seam_range_for_column(
                        ref_input,
                        sec_input,
                        input_regions,
                        layout,
                        x,
                        bands,
                        self.blend_width,
                    );
                    for row in 0..output_region.height as usize {
                        let y = output_region.y + row as i32;
                        let out_base = (row * out_width + col) * bands;
                        self.process_output_pixel(
                            ref_input,
                            sec_input,
                            output,
                            input_regions,
                            layout,
                            seam,
                            x,
                            y,
                            out_base,
                            bands,
                        );
                    }
                }
            }
        }
    }

    #[inline]
    fn process_output_pixel(
        &self,
        ref_input: &[F::Sample],
        sec_input: &[F::Sample],
        output: &mut [F::Sample],
        input_regions: &[Region],
        layout: MergeLayout,
        seam: Option<SeamRange>,
        x: i32,
        y: i32,
        out_base: usize,
        bands: usize,
    ) where
        F::Sample: Copy + Pod + ToF64 + FromF64,
    {
        let ref_pixel = pixel_at(
            ref_input,
            input_regions[0],
            x - layout.ref_left,
            y - layout.ref_top,
            bands,
        );
        let sec_pixel = pixel_at(
            sec_input,
            input_regions[1],
            x - layout.sec_left,
            y - layout.sec_top,
            bands,
        );

        let ref_zero = ref_pixel.is_none_or(all_zero);
        let sec_zero = sec_pixel.is_none_or(all_zero);

        match (ref_pixel, sec_pixel, ref_zero, sec_zero) {
            (Some(ref_px), Some(sec_px), false, false) if layout.overlap_contains(x, y) => {
                let blend_class = seam.map_or(BlendClass::Right, |range| {
                    range.classify(match self.axis {
                        MergeAxis::Horizontal => x,
                        MergeAxis::Vertical => y,
                    })
                });
                match blend_class {
                    BlendClass::Left => {
                        output[out_base..out_base + bands].copy_from_slice(ref_px);
                    }
                    BlendClass::Right => {
                        output[out_base..out_base + bands].copy_from_slice(sec_px);
                    }
                    BlendClass::Blend => {
                        let ref_weight = seam.map_or(0.0, |range| {
                            range.weight_ref(match self.axis {
                                MergeAxis::Horizontal => x,
                                MergeAxis::Vertical => y,
                            })
                        });
                        let sec_weight = 1.0 - ref_weight;
                        for band in 0..bands {
                            let blended = sec_px[band]
                                .to_f64()
                                .mul_add(sec_weight, ref_px[band].to_f64() * ref_weight);
                            output[out_base + band] = F::Sample::from_f64(blended);
                        }
                    }
                }
            }
            (Some(ref_px), _, false, _) => {
                output[out_base..out_base + bands].copy_from_slice(ref_px);
            }
            (_, Some(sec_px), _, false) => {
                output[out_base..out_base + bands].copy_from_slice(sec_px);
            }
            _ => {
                for band in 0..bands {
                    output[out_base + band] = F::Sample::from_f64(0.0);
                }
            }
        }
    }
}

/// Applies the `merge` mosaicing operation to related images. Use it when matching, aligning,
/// or merging overlapping image content.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::mosaicing::merge::MergeOp;
///
/// let op = MergeOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct MergeOp<F: BandFormat> {
    direction: MergeDirection,
    inner: MergeCommon<F>,
}

impl<F: BandFormat> MergeOp<F> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    /// Creates a new `MergeOp`.
    pub fn new(
        direction: MergeDirection,
        ref_width: u32,
        ref_height: u32,
        sec_width: u32,
        sec_height: u32,
        dx: i32,
        dy: i32,
        blend_width: u32,
        bands: u32,
    ) -> Self {
        Self {
            direction,
            inner: MergeCommon {
                axis: direction.into(),
                ref_width,
                ref_height,
                sec_width,
                sec_height,
                dx,
                dy,
                blend_width,
                bands,
                _format: PhantomData,
            },
        }
    }

    #[must_use]
    /// Returns or performs output width.
    pub fn output_width(&self) -> u32 {
        self.inner.output_width()
    }

    #[must_use]
    /// Returns or performs output height.
    pub fn output_height(&self) -> u32 {
        self.inner.output_height()
    }
}

/// Horizontal merge: places the reference image on the left and the secondary
/// image at offset `(-dx, -dy)`, feathering over the horizontal overlap.
pub struct MergeH<F: BandFormat> {
    inner: MergeCommon<F>,
}

impl<F: BandFormat> MergeH<F> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    /// Creates a new `MergeH`.
    pub const fn new(
        ref_width: u32,
        ref_height: u32,
        sec_width: u32,
        sec_height: u32,
        dx: i32,
        dy: i32,
        blend_width: u32,
        bands: u32,
    ) -> Self {
        Self {
            inner: MergeCommon {
                axis: MergeAxis::Horizontal,
                ref_width,
                ref_height,
                sec_width,
                sec_height,
                dx,
                dy,
                blend_width,
                bands,
                _format: PhantomData,
            },
        }
    }

    #[must_use]
    /// Returns or performs output width.
    pub fn output_width(&self) -> u32 {
        self.inner.output_width()
    }

    #[must_use]
    /// Returns or performs output height.
    pub fn output_height(&self) -> u32 {
        self.inner.output_height()
    }
}

/// Vertical merge: places the reference image above and the secondary image at
/// offset `(-dx, -dy)`, feathering over the vertical overlap.
pub struct MergeV<F: BandFormat> {
    inner: MergeCommon<F>,
}

impl<F: BandFormat> MergeV<F> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    /// Creates a new `MergeV`.
    pub const fn new(
        ref_width: u32,
        ref_height: u32,
        sec_width: u32,
        sec_height: u32,
        dx: i32,
        dy: i32,
        blend_width: u32,
        bands: u32,
    ) -> Self {
        Self {
            inner: MergeCommon {
                axis: MergeAxis::Vertical,
                ref_width,
                ref_height,
                sec_width,
                sec_height,
                dx,
                dy,
                blend_width,
                bands,
                _format: PhantomData,
            },
        }
    }

    #[must_use]
    /// Returns or performs output width.
    pub fn output_width(&self) -> u32 {
        self.inner.output_width()
    }

    #[must_use]
    /// Returns or performs output height.
    pub fn output_height(&self) -> u32 {
        self.inner.output_height()
    }
}

impl<F> DynOperation for MergeH<F>
where
    F: BandFormat + Send + Sync,
    F::Sample: Copy + Pod + ToF64 + FromF64 + Send,
{
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        self.inner.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn input_slot_count(&self) -> usize {
        2
    }

    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        let (px, py, pw, ph) = self.inner.placement(slot);
        intersect_and_translate(*output, px, py, pw, ph)
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        Self::output_width(self)
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        Self::output_height(self)
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
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        debug_assert!(
            false,
            "MergeH: dyn_process_region called on a 2-input node — use dyn_process_region_multi"
        );
        let len = output.len().min(input.len());
        output[..len].copy_from_slice(&input[..len]);
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
        process_multi(&self.inner, inputs, output, input_regions, output_region);
    }
}

impl<F> DynOperation for MergeV<F>
where
    F: BandFormat + Send + Sync,
    F::Sample: Copy + Pod + ToF64 + FromF64 + Send,
{
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        self.inner.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn input_slot_count(&self) -> usize {
        2
    }

    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        let (px, py, pw, ph) = self.inner.placement(slot);
        intersect_and_translate(*output, px, py, pw, ph)
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        Self::output_width(self)
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        Self::output_height(self)
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
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        debug_assert!(
            false,
            "MergeV: dyn_process_region called on a 2-input node — use dyn_process_region_multi"
        );
        let len = output.len().min(input.len());
        output[..len].copy_from_slice(&input[..len]);
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
        process_multi(&self.inner, inputs, output, input_regions, output_region);
    }
}

impl<F> DynOperation for MergeOp<F>
where
    F: BandFormat + Send + Sync,
    F::Sample: Copy + Pod + ToF64 + FromF64 + Send,
{
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        self.inner.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn input_slot_count(&self) -> usize {
        2
    }

    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        let (px, py, pw, ph) = self.inner.placement(slot);
        intersect_and_translate(*output, px, py, pw, ph)
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        Self::output_width(self)
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        Self::output_height(self)
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
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        debug_assert!(
            false,
            "MergeOp: dyn_process_region called on a 2-input node — use dyn_process_region_multi"
        );
        let len = output.len().min(input.len());
        output[..len].copy_from_slice(&input[..len]);
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
        let _ = self.direction;
        process_multi(&self.inner, inputs, output, input_regions, output_region);
    }
}

#[inline]
fn process_multi<F>(
    inner: &MergeCommon<F>,
    inputs: &[&[u8]],
    output: &mut [u8],
    input_regions: &[Region],
    output_region: Region,
) where
    F: BandFormat,
    F::Sample: Copy + Pod + ToF64 + FromF64,
{
    debug_assert_eq!(inputs.len(), 2, "Merge: expected exactly 2 input slices");
    debug_assert_eq!(
        input_regions.len(),
        2,
        "Merge: expected exactly 2 input regions"
    );

    let (Some(&ref_bytes), Some(&sec_bytes)) = (inputs.first(), inputs.get(1)) else {
        debug_assert!(false, "Merge: missing input slices");
        return;
    };

    let Ok(ref_input) = bytemuck::try_cast_slice(ref_bytes) else {
        debug_assert!(false, "Merge: cast failed on input[0]");
        return;
    };
    let Ok(sec_input) = bytemuck::try_cast_slice(sec_bytes) else {
        debug_assert!(false, "Merge: cast failed on input[1]");
        return;
    };
    let Ok(output) = bytemuck::try_cast_slice_mut(output) else {
        debug_assert!(false, "Merge: cast failed on output");
        return;
    };

    inner.process_region_typed(ref_input, sec_input, output, input_regions, output_region);
}

#[inline(always)]
const fn sample_offset(x: i32, y: i32, region: Region, bands: usize) -> Option<usize> {
    if region.is_empty()
        || x < region.x
        || y < region.y
        || x >= region.x + region.width as i32
        || y >= region.y + region.height as i32
    {
        return None;
    }

    let local_x = (x - region.x) as usize;
    let local_y = (y - region.y) as usize;
    Some((local_y * region.width as usize + local_x) * bands)
}

#[inline(always)]
fn all_zero<T>(pixel: &[T]) -> bool
where
    T: ToF64,
{
    pixel.iter().copied().all(|sample| sample.to_f64() == 0.0)
}

fn seam_range_for_row<T: ToF64 + Copy>(
    ref_input: &[T],
    sec_input: &[T],
    input_regions: &[Region],
    layout: MergeLayout,
    y: i32,
    bands: usize,
    blend_width: u32,
) -> Option<SeamRange> {
    if y < layout.overlap_top || y >= layout.overlap_top + layout.overlap_height as i32 {
        return None;
    }

    let sec_first =
        (layout.overlap_left..layout.overlap_left + layout.overlap_width as i32).find(|&x| {
            pixel_at(
                sec_input,
                input_regions[1],
                x - layout.sec_left,
                y - layout.sec_top,
                bands,
            )
            .is_some_and(|pixel| !all_zero(pixel))
        })?;
    let ref_last = (layout.overlap_left..layout.overlap_left + layout.overlap_width as i32)
        .rev()
        .find(|&x| {
            pixel_at(
                ref_input,
                input_regions[0],
                x - layout.ref_left,
                y - layout.ref_top,
                bands,
            )
            .is_some_and(|pixel| !all_zero(pixel))
        })?;

    clip_seam_range(
        SeamRange {
            first: sec_first,
            last: ref_last,
        },
        blend_width,
    )
}

fn seam_range_for_column<T: ToF64 + Copy>(
    ref_input: &[T],
    sec_input: &[T],
    input_regions: &[Region],
    layout: MergeLayout,
    x: i32,
    bands: usize,
    blend_width: u32,
) -> Option<SeamRange> {
    if x < layout.overlap_left || x >= layout.overlap_left + layout.overlap_width as i32 {
        return None;
    }

    let sec_first =
        (layout.overlap_top..layout.overlap_top + layout.overlap_height as i32).find(|&y| {
            pixel_at(
                sec_input,
                input_regions[1],
                x - layout.sec_left,
                y - layout.sec_top,
                bands,
            )
            .is_some_and(|pixel| !all_zero(pixel))
        })?;
    let ref_last = (layout.overlap_top..layout.overlap_top + layout.overlap_height as i32)
        .rev()
        .find(|&y| {
            pixel_at(
                ref_input,
                input_regions[0],
                x - layout.ref_left,
                y - layout.ref_top,
                bands,
            )
            .is_some_and(|pixel| !all_zero(pixel))
        })?;

    clip_seam_range(
        SeamRange {
            first: sec_first,
            last: ref_last,
        },
        blend_width,
    )
}

const fn clip_seam_range(mut seam: SeamRange, blend_width: u32) -> Option<SeamRange> {
    if seam.last <= seam.first {
        return None;
    }

    let max_width = blend_width as i32;
    if seam.width() > max_width {
        let shrink_by = seam.width() - max_width;
        seam.first += shrink_by / 2;
        seam.last -= shrink_by / 2;
    }

    Some(seam)
}

#[inline(always)]
fn pixel_at<T>(input: &[T], region: Region, x: i32, y: i32, bands: usize) -> Option<&[T]> {
    sample_offset(x, y, region, bands).map(|idx| &input[idx..idx + bands])
}

fn intersect_and_translate(output: Region, px: i32, py: i32, pw: u32, ph: u32) -> Region {
    let left = output.x.max(px);
    let top = output.y.max(py);
    let right = (output.x + output.width as i32).min(px + pw as i32);
    let bottom = (output.y + output.height as i32).min(py + ph as i32);

    if right <= left || bottom <= top {
        return Region::new(0, 0, 0, 0);
    }

    Region::new(
        left - px,
        top - py,
        (right - left) as u32,
        (bottom - top) as u32,
    )
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::strategy::ValueTree;
    use viprs_core::format::U8;
    use viprs_ops_spatial::structural::{Join, JoinDirection};

    fn run_merge_h(
        op: &MergeH<U8>,
        reference: &[u8],
        secondary: &[u8],
        output_region: Region,
    ) -> Vec<u8> {
        run_dyn(op, reference, secondary, output_region)
    }

    fn run_merge_v(
        op: &MergeV<U8>,
        reference: &[u8],
        secondary: &[u8],
        output_region: Region,
    ) -> Vec<u8> {
        run_dyn(op, reference, secondary, output_region)
    }

    fn run_join(op: &Join, reference: &[u8], secondary: &[u8], output_region: Region) -> Vec<u8> {
        let inputs: &[&[u8]] = &[reference, secondary];
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let mut output = vec![0u8; output_region.pixel_count() * op.bands() as usize];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            &mut output,
            &input_regions,
            output_region,
        );
        output
    }

    fn force_non_zero_pixel(pixels: &mut [u8], pixel_index: usize, bands: usize) {
        let base = pixel_index * bands;
        if pixels[base..base + bands].iter().all(|&sample| sample == 0) {
            pixels[base] = 1;
        }
    }

    fn run_dyn(
        op: &dyn DynOperation,
        reference: &[u8],
        secondary: &[u8],
        output_region: Region,
    ) -> Vec<u8> {
        let inputs: &[&[u8]] = &[reference, secondary];
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let mut output = vec![0u8; output_region.pixel_count() * op.bands() as usize];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            &mut output,
            &input_regions,
            output_region,
        );
        output
    }

    #[test]
    fn horizontal_overlap_blends_across_seam() {
        let op = MergeH::<U8>::new(4, 1, 4, 1, -1, 0, 3, 1);
        let output = run_merge_h(
            &op,
            &[10, 20, 30, 40],
            &[100, 110, 120, 130],
            Region::new(0, 0, 5, 1),
        );
        assert_eq!(output[0], 10);
        assert!(output[2] > 30 && output[2] < 110);
        assert_eq!(output[4], 130);
    }

    #[test]
    fn vertical_overlap_blends_across_seam() {
        let op = MergeV::<U8>::new(1, 4, 1, 4, 0, -1, 3, 1);
        let output = run_merge_v(
            &op,
            &[10, 20, 30, 40],
            &[100, 110, 120, 130],
            Region::new(0, 0, 1, 5),
        );
        assert_eq!(output[0], 10);
        assert!(output[2] > 30 && output[2] < 110);
        assert_eq!(output[4], 130);
    }

    #[test]
    fn merge_op_matches_directional_variants() {
        let horizontal = MergeOp::<U8>::new(MergeDirection::Horizontal, 4, 1, 4, 1, -1, 0, 3, 1);
        let vertical = MergeOp::<U8>::new(MergeDirection::Vertical, 1, 4, 1, 4, 0, -1, 3, 1);

        assert_eq!(
            run_dyn(
                &horizontal,
                &[10, 20, 30, 40],
                &[100, 110, 120, 130],
                Region::new(0, 0, 5, 1)
            ),
            run_merge_h(
                &MergeH::<U8>::new(4, 1, 4, 1, -1, 0, 3, 1),
                &[10, 20, 30, 40],
                &[100, 110, 120, 130],
                Region::new(0, 0, 5, 1)
            )
        );
        assert_eq!(
            run_dyn(
                &vertical,
                &[10, 20, 30, 40],
                &[100, 110, 120, 130],
                Region::new(0, 0, 1, 5)
            ),
            run_merge_v(
                &MergeV::<U8>::new(1, 4, 1, 4, 0, -1, 3, 1),
                &[10, 20, 30, 40],
                &[100, 110, 120, 130],
                Region::new(0, 0, 1, 5)
            )
        );
    }

    #[test]
    fn identical_tiles_have_invisible_horizontal_seam() {
        let pixels = [23u8, 23, 23, 23];
        let merge = MergeOp::<U8>::new(MergeDirection::Horizontal, 4, 1, 4, 1, -2, 0, 2, 1);
        let output = run_dyn(&merge, &pixels, &pixels, Region::new(0, 0, 6, 1));

        assert_eq!(output, vec![23; 6]);
        assert_eq!(
            output
                .iter()
                .zip([23u8; 6])
                .map(|(lhs, rhs)| lhs.abs_diff(rhs))
                .max(),
            Some(0)
        );
    }

    #[test]
    fn blend_range_handles_zero_and_degenerate_blends() {
        assert!(clip_seam_range(SeamRange { first: 4, last: 4 }, 0).is_none());

        let degenerate = clip_seam_range(
            SeamRange {
                first: 10,
                last: 14,
            },
            0,
        )
        .unwrap();
        assert_eq!(
            degenerate,
            SeamRange {
                first: 12,
                last: 12
            }
        );
    }

    #[test]
    fn horizontal_blend_respects_transparent_secondary_lead_in() {
        let op = MergeH::<U8>::new(5, 1, 5, 1, -2, 0, 4, 1);
        let output = run_merge_h(
            &op,
            &[10, 20, 30, 40, 50],
            &[0, 0, 120, 130, 140],
            Region::new(0, 0, 7, 1),
        );

        assert_eq!(output[2], 30);
        assert_eq!(output[3], 40);
        assert_eq!(output[4], 120);
    }

    #[test]
    fn vertical_blend_respects_transparent_secondary_lead_in() {
        let op = MergeV::<U8>::new(1, 5, 1, 5, 0, -2, 4, 1);
        let output = run_merge_v(
            &op,
            &[10, 20, 30, 40, 50],
            &[0, 0, 120, 130, 140],
            Region::new(0, 0, 1, 7),
        );

        assert_eq!(output[2], 30);
        assert_eq!(output[3], 40);
        assert_eq!(output[4], 120);
    }

    #[test]
    fn merge_dyn_metadata_matches_layout_and_regions() {
        let horizontal = MergeH::<U8>::new(4, 3, 4, 3, -2, 1, 2, 2);
        let vertical = MergeV::<U8>::new(3, 4, 3, 4, 1, -2, 2, 2);
        let h_dyn: &dyn DynOperation = &horizontal;
        let v_dyn: &dyn DynOperation = &vertical;
        let horizontal_region =
            Region::new(0, 0, horizontal.output_width(), horizontal.output_height());
        let vertical_region = Region::new(0, 0, vertical.output_width(), vertical.output_height());

        assert_eq!(horizontal.inner.placement(99), (0, 0, 0, 0));

        assert_eq!(h_dyn.input_format(), viprs_core::format::BandFormatId::U8);
        assert_eq!(h_dyn.output_format(), viprs_core::format::BandFormatId::U8);
        assert_eq!(h_dyn.bands(), 2);
        assert_eq!(h_dyn.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(h_dyn.input_slot_count(), 2);
        assert_eq!(
            h_dyn.required_input_region(&horizontal_region),
            horizontal_region
        );
        assert_eq!(h_dyn.output_width(1), horizontal.output_width());
        assert_eq!(h_dyn.output_height(1), horizontal.output_height());
        assert_eq!(h_dyn.node_spec(32, 16), NodeSpec::identity(32, 16));
        assert_eq!(
            h_dyn.required_input_region_slot(&Region::new(10, 10, 2, 2), 0),
            Region::new(0, 0, 0, 0)
        );

        assert_eq!(v_dyn.input_format(), viprs_core::format::BandFormatId::U8);
        assert_eq!(v_dyn.output_format(), viprs_core::format::BandFormatId::U8);
        assert_eq!(v_dyn.bands(), 2);
        assert_eq!(v_dyn.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(v_dyn.input_slot_count(), 2);
        assert_eq!(
            v_dyn.required_input_region(&vertical_region),
            vertical_region
        );
        assert_eq!(v_dyn.output_width(1), vertical.output_width());
        assert_eq!(v_dyn.output_height(1), vertical.output_height());
        assert_eq!(v_dyn.node_spec(16, 32), NodeSpec::identity(16, 32));
        assert_eq!(
            v_dyn.required_input_region_slot(&Region::new(10, 10, 2, 2), 1),
            Region::new(0, 0, 0, 0)
        );
    }

    proptest! {
        #[test]
        fn zero_horizontal_overlap_matches_join(
            ref_width in 1u32..6,
            sec_width in 1u32..6,
            height in 1u32..5,
            bands in 1u32..4,
        ) {
            let ref_len = ref_width as usize * height as usize * bands as usize;
            let sec_len = sec_width as usize * height as usize * bands as usize;
            let mut runner = proptest::test_runner::TestRunner::default();
            let reference = prop::collection::vec(any::<u8>(), ref_len)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let secondary = prop::collection::vec(any::<u8>(), sec_len)
                .new_tree(&mut runner)
                .unwrap()
                .current();

            let merge = MergeH::<U8>::new(
                ref_width,
                height,
                sec_width,
                height,
                -(ref_width as i32),
                0,
                0,
                bands,
            );
            let join = Join::new(
                JoinDirection::Horizontal,
                ref_width,
                height,
                sec_width,
                height,
                bands,
                viprs_core::format::BandFormatId::U8,
            );

            let output_region = Region::new(0, 0, ref_width + sec_width, height);
            prop_assert_eq!(
                run_merge_h(&merge, &reference, &secondary, output_region),
                run_join(&join, &reference, &secondary, output_region)
            );
        }

        #[test]
        fn zero_vertical_overlap_matches_join(
            width in 1u32..6,
            ref_height in 1u32..6,
            sec_height in 1u32..6,
            bands in 1u32..4,
        ) {
            let ref_len = width as usize * ref_height as usize * bands as usize;
            let sec_len = width as usize * sec_height as usize * bands as usize;
            let mut runner = proptest::test_runner::TestRunner::default();
            let reference = prop::collection::vec(any::<u8>(), ref_len)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let secondary = prop::collection::vec(any::<u8>(), sec_len)
                .new_tree(&mut runner)
                .unwrap()
                .current();

            let merge = MergeV::<U8>::new(
                width,
                ref_height,
                width,
                sec_height,
                0,
                -(ref_height as i32),
                0,
                bands,
            );
            let join = Join::new(
                JoinDirection::Vertical,
                width,
                ref_height,
                width,
                sec_height,
                bands,
                viprs_core::format::BandFormatId::U8,
            );

            let output_region = Region::new(0, 0, width, ref_height + sec_height);
            prop_assert_eq!(
                run_merge_v(&merge, &reference, &secondary, output_region),
                run_join(&join, &reference, &secondary, output_region)
            );
        }

        #[test]
        fn horizontal_non_overlapping_merge_is_associative(
            a_width in 1u32..5,
            b_width in 1u32..5,
            c_width in 1u32..5,
            height in 1u32..4,
            bands in 1u32..4,
        ) {
            let a_len = a_width as usize * height as usize * bands as usize;
            let b_len = b_width as usize * height as usize * bands as usize;
            let c_len = c_width as usize * height as usize * bands as usize;
            let mut runner = proptest::test_runner::TestRunner::default();
            let a = prop::collection::vec(any::<u8>(), a_len)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let b = prop::collection::vec(any::<u8>(), b_len)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let c = prop::collection::vec(any::<u8>(), c_len)
                .new_tree(&mut runner)
                .unwrap()
                .current();

            let ab = MergeH::<U8>::new(a_width, height, b_width, height, -(a_width as i32), 0, 0, bands);
            let ab_region = Region::new(0, 0, a_width + b_width, height);
            let ab_pixels = run_merge_h(&ab, &a, &b, ab_region);

            let left = MergeH::<U8>::new(
                a_width + b_width,
                height,
                c_width,
                height,
                -((a_width + b_width) as i32),
                0,
                0,
                bands,
            );
            let left_pixels = run_merge_h(&left, &ab_pixels, &c, Region::new(0, 0, a_width + b_width + c_width, height));

            let bc = MergeH::<U8>::new(b_width, height, c_width, height, -(b_width as i32), 0, 0, bands);
            let bc_pixels = run_merge_h(&bc, &b, &c, Region::new(0, 0, b_width + c_width, height));

            let right = MergeH::<U8>::new(a_width, height, b_width + c_width, height, -(a_width as i32), 0, 0, bands);
            let right_pixels = run_merge_h(&right, &a, &bc_pixels, Region::new(0, 0, a_width + b_width + c_width, height));

            prop_assert_eq!(left_pixels, right_pixels);
        }

        #[test]
        fn horizontal_full_overlap_preserves_boundary_pixels_for_multiband_images(
            width in 3u32..7,
            height in 2u32..6,
            bands in 2u32..5,
        ) {
            let len = width as usize * height as usize * bands as usize;
            let mut runner = proptest::test_runner::TestRunner::default();
            let mut reference = prop::collection::vec(any::<u8>(), len)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let mut secondary = prop::collection::vec(any::<u8>(), len)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let band_count = bands as usize;
            force_non_zero_pixel(&mut reference, 0, band_count);
            force_non_zero_pixel(&mut secondary, (width * height - 1) as usize, band_count);

            let merge = MergeH::<U8>::new(width, height, width, height, 0, 0, width - 2, bands);
            let output = run_merge_h(&merge, &reference, &secondary, Region::new(0, 0, width, height));
            let last = output.len() - band_count;

            prop_assert_eq!(&output[..band_count], &reference[..band_count]);
            prop_assert_eq!(&output[last..], &secondary[last..]);
        }

        #[test]
        fn vertical_non_overlapping_merge_is_associative(
            width in 1u32..4,
            a_height in 1u32..5,
            b_height in 1u32..5,
            c_height in 1u32..5,
            bands in 1u32..4,
        ) {
            let a_len = width as usize * a_height as usize * bands as usize;
            let b_len = width as usize * b_height as usize * bands as usize;
            let c_len = width as usize * c_height as usize * bands as usize;
            let mut runner = proptest::test_runner::TestRunner::default();
            let a = prop::collection::vec(any::<u8>(), a_len)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let b = prop::collection::vec(any::<u8>(), b_len)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let c = prop::collection::vec(any::<u8>(), c_len)
                .new_tree(&mut runner)
                .unwrap()
                .current();

            let ab = MergeV::<U8>::new(width, a_height, width, b_height, 0, -(a_height as i32), 0, bands);
            let ab_pixels = run_merge_v(&ab, &a, &b, Region::new(0, 0, width, a_height + b_height));

            let left = MergeV::<U8>::new(
                width,
                a_height + b_height,
                width,
                c_height,
                0,
                -((a_height + b_height) as i32),
                0,
                bands,
            );
            let left_pixels = run_merge_v(&left, &ab_pixels, &c, Region::new(0, 0, width, a_height + b_height + c_height));

            let bc = MergeV::<U8>::new(width, b_height, width, c_height, 0, -(b_height as i32), 0, bands);
            let bc_pixels = run_merge_v(&bc, &b, &c, Region::new(0, 0, width, b_height + c_height));

            let right = MergeV::<U8>::new(width, a_height, width, b_height + c_height, 0, -(a_height as i32), 0, bands);
            let right_pixels = run_merge_v(&right, &a, &bc_pixels, Region::new(0, 0, width, a_height + b_height + c_height));

            prop_assert_eq!(left_pixels, right_pixels);
        }

        #[test]
        fn vertical_full_overlap_preserves_boundary_pixels_for_multiband_images(
            width in 2u32..6,
            height in 3u32..7,
            bands in 2u32..5,
        ) {
            let len = width as usize * height as usize * bands as usize;
            let mut runner = proptest::test_runner::TestRunner::default();
            let mut reference = prop::collection::vec(any::<u8>(), len)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let mut secondary = prop::collection::vec(any::<u8>(), len)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let band_count = bands as usize;
            force_non_zero_pixel(&mut reference, 0, band_count);
            force_non_zero_pixel(&mut reference, (width - 1) as usize, band_count);
            force_non_zero_pixel(&mut secondary, (width * height - 1) as usize, band_count);

            let merge = MergeV::<U8>::new(width, height, width, height, 0, 0, height - 2, bands);
            let output = run_merge_v(&merge, &reference, &secondary, Region::new(0, 0, width, height));
            let row_stride = width as usize * band_count;
            let bottom_right = output.len() - band_count;
            let secondary_bottom_right = secondary.len() - band_count;

            prop_assert_eq!(&output[..band_count], &reference[..band_count]);
            prop_assert_eq!(
                &output[bottom_right..bottom_right + band_count],
                &secondary[secondary_bottom_right..secondary_bottom_right + band_count]
            );
            prop_assert_eq!(
                &output[row_stride - band_count..row_stride],
                &reference[row_stride - band_count..row_stride]
            );
        }
    }
}
