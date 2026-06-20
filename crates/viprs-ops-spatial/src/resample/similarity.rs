#![allow(missing_docs)]
// REASON: these bridge helpers are public only for cross-crate workspace wiring, not end-user API.

use std::marker::PhantomData;

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    kernel::InterpolationKernel,
    op::{DynOperation, NodeSpec, Op, OperationBridge},
    resample::ResampleOp,
};

use super::{
    affine::Affine,
    sample_conv::{FromF64, ToF64},
};

/// Type alias for interpolation kind.
pub type InterpolationKind = InterpolationKernel;

/// Similarity transform (scale + rotate) implemented via affine backward mapping.
///
/// # Examples
/// ```ignore
/// use viprs_ops_spatial::resample::similarity::SimilarityOp;
///
/// let op = SimilarityOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SimilarityOp<F: BandFormat> {
    /// Stores the `scale` value for this item.
    pub scale: f64,
    /// Stores the `angle` value for this item.
    pub angle: f64,
    /// Interpolation kernel associated with this configuration.
    pub interpolate: InterpolationKind,
    affine: Affine<F>,
    output_w: u32,
    output_h: u32,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> SimilarityOp<F> {
    #[must_use]
    /// Returns or performs new auto canvas.
    pub fn new_auto_canvas(
        scale: f64,
        angle: f64,
        interpolate: InterpolationKind,
        input_w: u32,
        input_h: u32,
    ) -> Self {
        let safe_scale = safe_scale(scale);
        let radians = angle.to_radians();
        let cos = radians.cos();
        let sin = radians.sin();
        let forward = [
            safe_scale * cos,
            safe_scale * -sin,
            safe_scale * sin,
            safe_scale * cos,
        ];
        let canvas = auto_canvas(forward, input_w, input_h);
        let matrix = inverse_similarity_matrix(safe_scale, cos, sin);
        let tx = matrix[1].mul_add(f64::from(canvas.top), matrix[0] * f64::from(canvas.left));
        let ty = matrix[3].mul_add(f64::from(canvas.top), matrix[2] * f64::from(canvas.left));
        let affine = Affine::new(matrix, tx, ty, interpolate, canvas.width, canvas.height)
            .with_source_bounds(Region::new(0, 0, input_w, input_h));

        Self {
            scale: safe_scale,
            angle,
            interpolate,
            affine,
            output_w: canvas.width,
            output_h: canvas.height,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    /// Creates a new `SimilarityOp`.
    pub fn new(
        scale: f64,
        angle: f64,
        interpolate: InterpolationKind,
        output_w: u32,
        output_h: u32,
    ) -> Self {
        debug_assert!(
            scale.is_finite() && scale > 0.0,
            "SimilarityOp: scale must be > 0"
        );
        let safe_scale = safe_scale(scale);
        let radians = angle.to_radians();
        let matrix = inverse_similarity_matrix(safe_scale, radians.cos(), radians.sin());

        let cx = f64::from(output_w.saturating_sub(1)) * 0.5;
        let cy = f64::from(output_h.saturating_sub(1)) * 0.5;
        let tx = matrix[1].mul_add(-cy, matrix[0].mul_add(-cx, cx));
        let ty = matrix[3].mul_add(-cy, matrix[2].mul_add(-cx, cy));
        let affine = Affine::new(matrix, tx, ty, interpolate, output_w, output_h)
            .with_source_bounds(Region::new(0, 0, output_w, output_h));

        Self {
            scale: safe_scale,
            angle,
            interpolate,
            affine,
            output_w,
            output_h,
            _phantom: PhantomData,
        }
    }

    #[inline]
    const fn as_affine(&self) -> &Affine<F> {
        &self.affine
    }
}

impl<F: BandFormat> Op for SimilarityOp<F>
where
    F::Sample: ToF64 + FromF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        self.as_affine().demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.as_affine().required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.as_affine().node_spec(tile_w, tile_h)
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let mut affine_state = ();
        self.affine.process_region(&mut affine_state, input, output);
    }
}

impl<F: BandFormat> ResampleOp for SimilarityOp<F>
where
    F::Sample: ToF64 + FromF64,
{
    fn output_size(&self, _input_w: u32, _input_h: u32) -> (u32, u32) {
        (self.output_w, self.output_h)
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        self.output_w
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        self.output_h
    }
}

#[derive(Clone, Copy)]
struct SimilarityCanvas {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

#[inline]
fn safe_scale(scale: f64) -> f64 {
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}

#[inline]
fn inverse_similarity_matrix(scale: f64, cos: f64, sin: f64) -> [f64; 4] {
    [cos / scale, sin / scale, -sin / scale, cos / scale]
}

fn auto_canvas(matrix: [f64; 4], input_w: u32, input_h: u32) -> SimilarityCanvas {
    if input_w == 0 || input_h == 0 {
        return SimilarityCanvas {
            left: 0,
            top: 0,
            width: 0,
            height: 0,
        };
    }

    let [a, b, c, d] = matrix;
    let width = f64::from(input_w);
    let height = f64::from(input_h);
    let (x1, y1) = (0.0_f64, 0.0_f64);
    let (x2, y2) = (a * width, c * width);
    let (x3, y3) = (b * height, d * height);
    let (x4, y4) = (b.mul_add(height, a * width), d.mul_add(height, c * width));

    let left = x1.min(x2).min(x3).min(x4);
    let right = x1.max(x2).max(x3).max(x4);
    let top = y1.min(y2).min(y3).min(y4);
    let bottom = y1.max(y2).max(y3).max(y4);

    SimilarityCanvas {
        left: round_vips(left),
        top: round_vips(top),
        width: round_vips(right - left).max(0) as u32,
        height: round_vips(bottom - top).max(0) as u32,
    }
}

#[inline]
fn round_vips(value: f64) -> i32 {
    if value > 0.0 {
        (value + 0.5) as i32
    } else {
        (value - 0.5) as i32
    }
}

pub struct SimilarityBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + ToF64 + FromF64,
{
    inner: OperationBridge<SimilarityOp<F>>,
    output_w: u32,
    output_h: u32,
}

impl<F: BandFormat> SimilarityBridge<F>
where
    F::Sample: bytemuck::Pod + ToF64 + FromF64,
{
    #[must_use]
    pub fn new(
        scale: f64,
        angle: f64,
        interpolate: InterpolationKind,
        input_w: u32,
        input_h: u32,
        bands: u32,
    ) -> Self {
        let op = SimilarityOp::new_auto_canvas(scale, angle, interpolate, input_w, input_h);
        let output_w = op.output_w;
        let output_h = op.output_h;
        Self {
            inner: OperationBridge::new(op, bands),
            output_w,
            output_h,
        }
    }
}

impl<F: BandFormat> DynOperation for SimilarityBridge<F>
where
    F::Sample: bytemuck::Pod + ToF64 + FromF64 + Send,
{
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
        self.inner.output_format()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    fn demand_hint(&self) -> DemandHint {
        self.inner.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        self.output_w
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        self.output_h
    }

    fn output_size(&self, _input_w: u32, _input_h: u32) -> (u32, u32) {
        (self.output_w, self.output_h)
    }

    fn dyn_start(&self) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start_with_tile(tile_w, tile_h)
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn std::any::Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        self.inner
            .dyn_process_region(state, input, output, input_region, output_region);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::U8,
        image::{Region, Tile, TileMut},
    };

    fn run_similarity(
        input_data: &[u8],
        region: Region,
        scale: f64,
        angle: f64,
        kernel: InterpolationKind,
    ) -> Vec<u8> {
        let op = SimilarityOp::<U8>::new(scale, angle, kernel, region.width, region.height);
        let mut output_data = vec![0u8; region.pixel_count()];
        let input = Tile::<U8>::new(region, 1, input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn scale1_angle0_is_identity() {
        let region = Region::new(0, 0, 4, 4);
        let input: Vec<u8> = (0u8..16).collect();
        let output = run_similarity(&input, region, 1.0, 0.0, InterpolationKernel::Bilinear);
        assert_eq!(output, input);
    }

    #[test]
    fn output_dimensions_stay_fixed_to_canvas() {
        let op = SimilarityOp::<U8>::new(0.75, 12.0, InterpolationKernel::Bilinear, 32, 24);
        assert_eq!(op.output_width(128), 32);
        assert_eq!(op.output_height(128), 24);
    }

    #[test]
    fn auto_canvas_identity_matches_input_dimensions() {
        let op = SimilarityOp::<U8>::new_auto_canvas(1.0, 0.0, InterpolationKernel::Bilinear, 4, 3);
        assert_eq!(op.output_size(4, 3), (4, 3));
    }

    #[test]
    fn auto_canvas_right_angle_matches_libvips_bbox() {
        let op =
            SimilarityOp::<U8>::new_auto_canvas(1.0, 90.0, InterpolationKernel::Bilinear, 4, 2);
        assert_eq!(op.output_size(4, 2), (2, 4));
    }

    #[test]
    fn auto_canvas_scale_rounds_like_libvips_transform_rect() {
        let op =
            SimilarityOp::<U8>::new_auto_canvas(1.25, 0.0, InterpolationKernel::Bilinear, 5, 3);
        assert_eq!(op.output_size(5, 3), (6, 4));
    }

    #[test]
    fn invalid_scales_fall_back_to_unit_scale() {
        let auto =
            SimilarityOp::<U8>::new_auto_canvas(-4.0, 0.0, InterpolationKernel::Nearest, 0, 3);

        assert_eq!(safe_scale(f64::NAN), 1.0);
        assert_eq!(safe_scale(-4.0), 1.0);
        assert_eq!(auto.output_size(0, 3), (0, 0));
    }

    #[test]
    fn round_vips_and_bridge_cover_negative_rounding_and_dyn_contract() {
        assert_eq!(round_vips(-1.2), -1);
        assert_eq!(round_vips(-1.8), -2);

        let bridge = SimilarityBridge::<U8>::new(1.0, 90.0, InterpolationKernel::Nearest, 4, 2, 1);
        let mut state = bridge.dyn_start_with_tile(2, 2);
        let input_region = Region::new(0, 0, 4, 2);
        let output_region = Region::new(0, 0, bridge.output_width(4), bridge.output_height(2));
        let input = vec![0u8; input_region.pixel_count()];
        let mut output = vec![0u8; output_region.pixel_count()];

        bridge.dyn_process_region(
            &mut *state,
            &input,
            &mut output,
            input_region,
            output_region,
        );

        assert_eq!(bridge.output_size(4, 2), (2, 4));
        assert_eq!(bridge.bands(), 1);
        assert_eq!(bridge.demand_hint(), DemandHint::SmallTile);
    }

    proptest! {
        #[test]
        fn identity_preserves_uniform_input(
            value in any::<u8>(),
            width in 1u32..=8,
            height in 1u32..=8,
        ) {
            let region = Region::new(0, 0, width, height);
            let input = vec![value; region.pixel_count()];
            let output = run_similarity(&input, region, 1.0, 0.0, InterpolationKernel::Bilinear);
            prop_assert_eq!(output, input);
        }
    }
}
