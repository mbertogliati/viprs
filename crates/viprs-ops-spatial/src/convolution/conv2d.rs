#![allow(clippy::unused_self)]
// REASON: helper methods remain instance-bound for API symmetry with other convolution ops.

use std::marker::PhantomData;

use bytemuck::Pod;

use viprs_core::{
    error::ViprsError,
    format::{BandFormat, BandFormatId, F32},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

use super::common::{ConvolutionMask2d, apply_scale_offset, validate_kernel_2d};

/// Trait for converting a `BandFormat::Sample` to `f64` for kernel accumulation.
pub trait ToF64: Copy {
    /// Converts this value to f64.
    fn to_f64(self) -> f64;
}

impl ToF64 for u8 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for u16 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for i16 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for u32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for i32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

/// 2D convolution with an arbitrary kernel.
///
/// The output format is always F32 regardless of the input format. This preserves
/// precision across all input types without requiring the caller to reason about
/// overflow or clamping during accumulation.
///
/// The kernel must have odd dimensions in both axes — the center element is the
/// kernel's anchor point. A `kernel_w × kernel_h` kernel requires the input tile
/// to be `radius_x` pixels wider on each horizontal side and `radius_y` pixels
/// taller on each vertical side, where `radius_x = kernel_w / 2` and
/// `radius_y = kernel_h / 2`.
///
/// The source is responsible for edge-extension when the expanded input region
/// goes outside the image boundary.
pub struct Conv2d<F: BandFormat> {
    /// Flattened kernel coefficients, row-major.
    kernel: Vec<f64>,
    kernel_w: u32,
    kernel_h: u32,
    scale: f64,
    offset: f64,
    /// Number of extra input pixels needed on each horizontal side.
    radius_x: u32,
    /// Number of extra input pixels needed on each vertical side.
    radius_y: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Conv2d<F>
where
    F::Sample: ToF64 + Pod,
{
    /// Construct a `Conv2d` from a row-major 2D kernel.
    ///
    /// Returns an error if:
    /// - The kernel is empty.
    /// - The kernel rows are not all the same width (not rectangular).
    /// - Either dimension is even (kernels must have odd dimensions for a center anchor).
    pub fn new(kernel: Vec<Vec<f64>>) -> Result<Self, ViprsError> {
        Self::with_mask(ConvolutionMask2d::from_coefficients(kernel)?)
    }

    /// Returns this value configured with mask.
    pub fn with_mask(mask: ConvolutionMask2d) -> Result<Self, ViprsError> {
        let (kernel_w, kernel_h) = validate_kernel_2d("Conv2d", mask.coefficients())?;
        let scale = mask.scale();
        let offset = mask.offset();
        let flat: Vec<f64> = mask.into_coefficients().into_iter().flatten().collect();
        let kernel_w = kernel_w as u32;
        let kernel_h = kernel_h as u32;
        let radius_x = kernel_w / 2;
        let radius_y = kernel_h / 2;

        Ok(Self {
            kernel: flat,
            kernel_w,
            kernel_h,
            scale,
            offset,
            radius_x,
            radius_y,
            _format: PhantomData,
        })
    }
}

impl<F> Op for Conv2d<F>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    type Input = F;
    type Output = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        // Convolution has a 2D neighbourhood — SmallTile is mandatory.
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        // Expand the output region by the kernel radius on every side.
        // Negative coordinates are valid — the source clamps to image edges.
        Region::new(
            output.x - self.radius_x as i32,
            output.y - self.radius_y as i32,
            output.width + 2 * self.radius_x,
            output.height + 2 * self.radius_y,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius_x,
            input_tile_h: tile_h + 2 * self.radius_y,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F32>) {
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let bands = self.bands_from_tiles(input, output);

        // The input tile is `(out_w + 2*rx) × (out_h + 2*ry)` pixels.
        let in_w = input.region.width as usize;

        let kw = self.kernel_w as usize;
        let kh = self.kernel_h as usize;

        for oy in 0..out_h {
            for ox in 0..out_w {
                for b in 0..bands {
                    let mut acc = 0.0f64;
                    for ky in 0..kh {
                        let iy = oy + ky; // input row: oy + ky (halo starts at 0 in input tile)
                        for kx in 0..kw {
                            let ix = ox + kx; // input col: ox + kx
                            let in_idx = (iy * in_w + ix) * bands + b;
                            let k = self.kernel[ky * kw + kx];
                            acc = input.data[in_idx].to_f64().mul_add(k, acc);
                        }
                    }
                    let out_idx = (oy * out_w + ox) * bands + b;
                    output.data[out_idx] = apply_scale_offset(acc, self.scale, self.offset) as f32;
                }
            }
        }
    }
}

impl<F: BandFormat> Conv2d<F> {
    /// Extract the band count from the tile pair.
    ///
    /// Both tiles carry the same band count — the pipeline enforces this when
    /// building the node chain. Using `input.bands` (set by `OperationBridge`) is
    /// authoritative; `output.bands` is the same value.
    #[inline]
    const fn bands_from_tiles<'a>(&self, input: &Tile<'a, F>, _output: &TileMut<'a, F32>) -> usize {
        input.bands as usize
    }
}

impl<F: BandFormat> Conv2d<F> {
    /// Return the kernel radius in the X direction.
    #[must_use]
    pub const fn radius_x(&self) -> u32 {
        self.radius_x
    }
    /// Return the kernel radius in the Y direction.
    #[must_use]
    pub const fn radius_y(&self) -> u32 {
        self.radius_y
    }
    /// Return the output format ID (always F32).
    #[must_use]
    pub const fn output_format_id() -> BandFormatId {
        BandFormatId::F32
    }
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U8},
        image::{Region, Tile, TileMut},
    };

    // Allocation tests require the root crate test_support (global allocator).
    // Run via: cargo test -p viprs --lib

    /// `node_spec` must report expanded input tile and unchanged output tile.
    #[test]
    fn node_spec_input_expanded() {
        // 3×3 kernel → radius = 1
        let op = Conv2d::<F32>::new(box_3x3_kernel()).unwrap();
        let spec = op.node_spec(512, 512);
        assert_eq!(spec.input_tile_w, 514);
        assert_eq!(spec.input_tile_h, 514);
        assert_eq!(spec.output_tile_w, 512);
        assert_eq!(spec.output_tile_h, 512);
    }

    /// Constructor must reject empty kernels.
    #[test]
    fn rejects_empty_kernel() {
        assert!(Conv2d::<F32>::new(vec![]).is_err());
    }

    /// Constructor must reject even-dimension kernels.
    #[test]
    fn rejects_even_kernel() {
        assert!(Conv2d::<F32>::new(vec![vec![1.0, 1.0]]).is_err());
        assert!(Conv2d::<F32>::new(vec![vec![1.0], vec![1.0]]).is_err());
    }

    /// Constructor must reject non-rectangular kernels.
    #[test]
    fn rejects_non_rectangular_kernel() {
        let jagged = vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0]];
        assert!(Conv2d::<F32>::new(jagged).is_err());
    }

    #[test]
    fn radius_accessors_and_output_format_match_kernel() {
        let op = Conv2d::<F32>::new(box_3x3_kernel()).unwrap();
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::SmallTile);
        assert_eq!(op.radius_x(), 1);
        assert_eq!(op.radius_y(), 1);
        assert_eq!(Conv2d::<F32>::output_format_id(), BandFormatId::F32);
    }

    /// Identity kernel on U8 input must produce output ≈ input (converted to f32).
    #[test]
    fn identity_kernel_u8_input() {
        let op = Conv2d::<U8>::new(identity_kernel()).unwrap();
        let region = Region::new(0, 0, 4, 1);
        let input_data: Vec<u8> = vec![0, 64, 128, 255];
        let mut output_data = vec![0.0f32; 4];

        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);

        assert!((output_data[0] - 0.0).abs() < 1e-6);
        assert!((output_data[1] - 64.0).abs() < 1e-6);
        assert!((output_data[2] - 128.0).abs() < 1e-6);
        assert!((output_data[3] - 255.0).abs() < 1e-6);
    }

    #[test]
    fn zero_kernel_zeroes_multiband_output() {
        let op = Conv2d::<U8>::new(vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ])
        .unwrap();
        let in_region = Region::new(-1, -1, 3, 3);
        let out_region = Region::new(0, 0, 1, 1);
        let input_data = vec![
            1u8, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60, 7, 70, 8, 80, 9, 90,
        ];
        let input = Tile::<U8>::new(in_region, 2, &input_data);
        let mut output_data = vec![1.0f32; 2];
        let mut output = TileMut::<F32>::new(out_region, 2, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![0.0, 0.0]);
    }

    #[test]
    fn large_identity_mask_preserves_single_pixel_with_edge_extension() {
        let mut kernel = vec![vec![0.0; 5]; 5];
        kernel[2][2] = 1.0;
        let op = Conv2d::<F32>::new(kernel).unwrap();
        let in_region = Region::new(-2, -2, 5, 5);
        let out_region = Region::new(0, 0, 1, 1);
        let input_data = vec![9.0f32; 25];
        let input = Tile::<F32>::new(in_region, 1, &input_data);
        let mut output_data = vec![0.0f32; 1];
        let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 9.0).abs() < 1e-6);
    }

    proptest! {
        /// A 1×1 identity kernel applied to any F32 input must produce output ≈ input.
        #[test]
        fn identity_kernel_f32_proptest(
            pixels in proptest::collection::vec(0.0f32..=255.0f32, 1..=64)
        ) {
            let len = pixels.len();
            let op = Conv2d::<F32>::new(identity_kernel()).unwrap();
            let region = Region::new(0, 0, len as u32, 1);
            let mut output_data = vec![0.0f32; len];
            let input = Tile::<F32>::new(region, 1, &pixels);
            let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            for (got, expected) in output_data.iter().zip(pixels.iter()) {
                prop_assert!(
                    (got - expected).abs() < 1e-4,
                    "identity kernel proptest: expected {expected}, got {got}"
                );
            }
        }

        #[test]
        fn symmetric_kernel_commutes_with_horizontal_reflection(
            pixels in proptest::collection::vec(-10.0f32..10.0f32, 1..16)
        ) {
            let kernel = vec![vec![0.25, 0.5, 0.25]];
            let op = Conv2d::<F32>::new(kernel).unwrap();
            let radius = 1usize;
            let input = edge_extend_scanline(&pixels, radius);
            let mirrored_pixels = mirror_scanline(&pixels);
            let mirrored_input = edge_extend_scanline(&mirrored_pixels, radius);
            let in_region = Region::new(-(radius as i32), 0, input.len() as u32, 1);
            let out_region = Region::new(0, 0, pixels.len() as u32, 1);

            let mut output = vec![0.0f32; pixels.len()];
            let mut mirrored_output = vec![0.0f32; pixels.len()];
            let input_tile = Tile::<F32>::new(in_region, 1, &input);
            let mirrored_input_tile = Tile::<F32>::new(in_region, 1, &mirrored_input);
            let mut output_tile = TileMut::<F32>::new(out_region, 1, &mut output);
            let mut mirrored_output_tile =
                TileMut::<F32>::new(out_region, 1, &mut mirrored_output);

            op.process_region(&mut (), &input_tile, &mut output_tile);
            op.process_region(&mut (), &mirrored_input_tile, &mut mirrored_output_tile);

            for (lhs, rhs) in output.iter().zip(mirrored_output.iter().rev()) {
                prop_assert!((lhs - rhs).abs() < 1e-5);
            }
        }
    }
}
