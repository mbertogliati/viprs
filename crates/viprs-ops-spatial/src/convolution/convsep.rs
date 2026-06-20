use std::marker::PhantomData;

use bytemuck::Pod;

use viprs_core::{
    error::ViprsError,
    format::{BandFormat, F32},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

use super::common::{ConvolutionMask1d, ToF64, apply_scale_offset, validate_kernel_1d};

/// Horizontal pass of a generic separable convolution.
pub struct ConvSepH<F: BandFormat> {
    kernel: Vec<f64>,
    radius: usize,
    scale: f64,
    offset: f64,
    _format: PhantomData<F>,
}

impl<F: BandFormat> ConvSepH<F>
where
    F::Sample: ToF64 + Pod,
{
    /// Creates a new `ConvSepH`.
    pub fn new(kernel: Vec<f64>) -> Result<Self, ViprsError> {
        Self::with_mask(ConvolutionMask1d::from_coefficients(kernel)?)
    }

    /// Returns this value configured with mask.
    pub fn with_mask(mask: ConvolutionMask1d) -> Result<Self, ViprsError> {
        let radius = validate_kernel_1d("ConvSepH", mask.coefficients())?;
        let scale = mask.scale();
        let offset = mask.offset();
        Ok(Self {
            kernel: mask.into_coefficients(),
            radius,
            scale,
            offset,
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs radius.
    pub const fn radius(&self) -> usize {
        self.radius
    }
}

impl<F> Op for ConvSepH<F>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    type Input = F;
    type Output = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.radius as i32,
            output.y,
            output.width + 2 * self.radius as u32,
            output.height,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius as u32,
            input_tile_h: tile_h,
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
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;

        for y in 0..out_h {
            for ox in 0..out_w {
                for band in 0..bands {
                    let mut acc = 0.0f64;
                    for (kx, weight) in self.kernel.iter().enumerate() {
                        let ix = ox + kx;
                        let idx = (y * in_w + ix) * bands + band;
                        acc = input.data[idx].to_f64().mul_add(*weight, acc);
                    }
                    let out_idx = (y * out_w + ox) * bands + band;
                    output.data[out_idx] = apply_scale_offset(acc, self.scale, self.offset) as f32;
                }
            }
        }
    }
}

/// Vertical pass of a generic separable convolution.
pub struct ConvSepV<F: BandFormat> {
    kernel: Vec<f64>,
    radius: usize,
    scale: f64,
    offset: f64,
    _format: PhantomData<F>,
}

impl<F: BandFormat> ConvSepV<F>
where
    F::Sample: ToF64 + Pod,
{
    /// Creates a new `ConvSepV`.
    pub fn new(kernel: Vec<f64>) -> Result<Self, ViprsError> {
        Self::with_mask(ConvolutionMask1d::from_coefficients(kernel)?)
    }

    /// Returns this value configured with mask.
    pub fn with_mask(mask: ConvolutionMask1d) -> Result<Self, ViprsError> {
        let radius = validate_kernel_1d("ConvSepV", mask.coefficients())?;
        let scale = mask.scale();
        let offset = mask.offset();
        Ok(Self {
            kernel: mask.into_coefficients(),
            radius,
            scale,
            offset,
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs radius.
    pub const fn radius(&self) -> usize {
        self.radius
    }
}

impl<F> Op for ConvSepV<F>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    type Input = F;
    type Output = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x,
            output.y - self.radius as i32,
            output.width,
            output.height + 2 * self.radius as u32,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w,
            input_tile_h: tile_h + 2 * self.radius as u32,
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
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;

        for oy in 0..out_h {
            for x in 0..out_w {
                for band in 0..bands {
                    let mut acc = 0.0f64;
                    for (ky, weight) in self.kernel.iter().enumerate() {
                        let iy = oy + ky;
                        let idx = (iy * in_w + x) * bands + band;
                        acc = input.data[idx].to_f64().mul_add(*weight, acc);
                    }
                    let out_idx = (oy * out_w + x) * bands + band;
                    output.data[out_idx] = apply_scale_offset(acc, self.scale, self.offset) as f32;
                }
            }
        }
    }
}

/// Convenience facade for chaining both passes of a separable convolution.
pub struct ConvSep {
    /// Stores the `kernel` value for this item.
    pub kernel: Vec<f64>,
    /// Stores the `h` value for this item.
    pub h: ConvSepH<F32>,
    /// Stores the `v` value for this item.
    pub v: ConvSepV<F32>,
}

impl ConvSep {
    /// Creates a new `ConvSep`.
    pub fn new(kernel: Vec<f64>) -> Result<Self, ViprsError> {
        Self::with_mask(ConvolutionMask1d::from_coefficients(kernel)?)
    }

    /// Returns this value configured with mask.
    pub fn with_mask(mask: ConvolutionMask1d) -> Result<Self, ViprsError> {
        let vertical_mask =
            ConvolutionMask1d::new(mask.coefficients().to_vec(), mask.scale(), 0.0)?;
        Ok(Self {
            h: ConvSepH::with_mask(mask.clone())?,
            v: ConvSepV::with_mask(vertical_mask)?,
            kernel: mask.into_coefficients(),
        })
    }
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U16},
        image::{Region, Tile, TileMut},
    };

    fn run_h(
        input_data: &[f32],
        in_region: Region,
        out_region: Region,
        kernel: Vec<f64>,
    ) -> Vec<f32> {
        let op = ConvSepH::<F32>::new(kernel).unwrap();
        let mut output = vec![0.0f32; out_region.width as usize * out_region.height as usize];
        let input = Tile::<F32>::new(in_region, 1, input_data);
        let mut tile = TileMut::<F32>::new(out_region, 1, &mut output);
        let mut state = ();
        op.process_region(&mut state, &input, &mut tile);
        output
    }

    fn run_v(
        input_data: &[f32],
        in_region: Region,
        out_region: Region,
        kernel: Vec<f64>,
    ) -> Vec<f32> {
        let op = ConvSepV::<F32>::new(kernel).unwrap();
        let mut output = vec![0.0f32; out_region.width as usize * out_region.height as usize];
        let input = Tile::<F32>::new(in_region, 1, input_data);
        let mut tile = TileMut::<F32>::new(out_region, 1, &mut output);
        let mut state = ();
        op.process_region(&mut state, &input, &mut tile);
        output
    }

    fn run_h_u16(
        input_data: &[u16],
        in_region: Region,
        out_region: Region,
        bands: u32,
        kernel: Vec<f64>,
    ) -> Vec<f32> {
        let op = ConvSepH::<U16>::new(kernel).unwrap();
        let mut output =
            vec![0.0f32; out_region.width as usize * out_region.height as usize * bands as usize];
        let input = Tile::<U16>::new(in_region, bands, input_data);
        let mut tile = TileMut::<F32>::new(out_region, bands, &mut output);
        let mut state = ();
        op.process_region(&mut state, &input, &mut tile);
        output
    }

    fn run_h_mask(
        input_data: &[f32],
        in_region: Region,
        out_region: Region,
        mask: ConvolutionMask1d,
    ) -> Vec<f32> {
        let op = ConvSepH::<F32>::with_mask(mask).unwrap();
        let mut output = vec![0.0f32; out_region.width as usize * out_region.height as usize];
        let input = Tile::<F32>::new(in_region, 1, input_data);
        let mut tile = TileMut::<F32>::new(out_region, 1, &mut output);
        let mut state = ();
        op.process_region(&mut state, &input, &mut tile);
        output
    }

    fn edge_extend_scanline(samples: &[f32], radius: usize) -> Vec<f32> {
        let mut extended = Vec::with_capacity(samples.len() + 2 * radius);
        for x in 0..(samples.len() + 2 * radius) {
            let src_x = (x as i32 - radius as i32).clamp(0, samples.len() as i32 - 1) as usize;
            extended.push(samples[src_x]);
        }
        extended
    }

    fn reverse_scanline(samples: &[f32]) -> Vec<f32> {
        samples.iter().copied().rev().collect()
    }

    fn run_v_u16(
        input_data: &[u16],
        in_region: Region,
        out_region: Region,
        bands: u32,
        kernel: Vec<f64>,
    ) -> Vec<f32> {
        let op = ConvSepV::<U16>::new(kernel).unwrap();
        let mut output =
            vec![0.0f32; out_region.width as usize * out_region.height as usize * bands as usize];
        let input = Tile::<U16>::new(in_region, bands, input_data);
        let mut tile = TileMut::<F32>::new(out_region, bands, &mut output);
        let mut state = ();
        op.process_region(&mut state, &input, &mut tile);
        output
    }

    #[test]
    fn metadata_and_facade_match_kernel_radius() {
        let kernel = vec![0.25, 0.5, 0.25];
        let h = ConvSepH::<U16>::new(kernel.clone()).unwrap();
        let v = ConvSepV::<U16>::new(kernel.clone()).unwrap();
        let out_region = Region::new(4, 6, 3, 2);
        assert_eq!(h.radius(), 1);
        assert_eq!(v.radius(), 1);
        assert_eq!(h.demand_hint(), viprs_core::image::DemandHint::ThinStrip);
        assert_eq!(v.demand_hint(), viprs_core::image::DemandHint::ThinStrip);
        assert_eq!(
            h.required_input_region(&out_region),
            Region::new(3, 6, 5, 2)
        );
        assert_eq!(
            v.required_input_region(&out_region),
            Region::new(4, 5, 3, 4)
        );
        let h_spec = h.node_spec(3, 2);
        assert_eq!(h_spec.input_tile_w, 5);
        assert_eq!(h_spec.input_tile_h, 2);
        let v_spec = v.node_spec(3, 2);
        assert_eq!(v_spec.input_tile_w, 3);
        assert_eq!(v_spec.input_tile_h, 4);

        let convsep = ConvSep::new(kernel.clone()).unwrap();
        assert_eq!(convsep.kernel, kernel);
        assert_eq!(convsep.h.radius(), 1);
        assert_eq!(convsep.v.radius(), 1);
    }

    #[test]
    fn identity_kernel_preserves_multiband_u16_samples() {
        let region = Region::new(0, 0, 2, 2);
        let input_data = vec![1u16, 10, 2, 20, 3, 30, 4, 40];
        let expected = vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        assert_eq!(
            run_h_u16(&input_data, region, region, 2, vec![1.0]),
            expected
        );
        assert_eq!(
            run_v_u16(&input_data, region, region, 2, vec![1.0]),
            expected
        );
    }

    #[test]
    fn mask_metadata_applies_offset_only_on_first_pass() {
        let mask = ConvolutionMask1d::new(vec![1.0], 2.0, 10.0).unwrap();
        let facade = ConvSep::with_mask(mask).unwrap();
        let region = Region::new(0, 0, 2, 1);
        let input_data = vec![4.0f32, 8.0];

        let after_h = run_h_mask(
            &input_data,
            region,
            region,
            ConvolutionMask1d::new(vec![1.0], 2.0, 10.0).unwrap(),
        );
        let mut output = vec![0.0f32; 2];
        let input = Tile::<F32>::new(region, 1, &after_h);
        let mut tile = TileMut::<F32>::new(region, 1, &mut output);
        let mut state = ();
        facade.v.process_region(&mut state, &input, &mut tile);

        assert_eq!(after_h, vec![12.0, 14.0]);
        assert_eq!(output, vec![6.0, 7.0]);
    }

    #[test]
    fn large_symmetric_kernel_preserves_single_pixel_under_edge_extension() {
        let kernel = vec![0.125, 0.1875, 0.375, 0.1875, 0.125];
        let h_region = Region::new(-2, 0, 5, 1);
        let out_region = Region::new(0, 0, 1, 1);
        let after_h = run_h(&vec![7.0; 5], h_region, out_region, kernel.clone());
        let v_region = Region::new(0, -2, 1, 5);
        let after_v = run_v(&vec![7.0; 5], v_region, out_region, kernel);

        assert!((after_h[0] - 7.0).abs() < 1e-6);
        assert!((after_v[0] - 7.0).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn identity_kernel_round_trips_scanline(samples in prop::collection::vec(-10.0f32..10.0, 1..32)) {
            let width = samples.len() as u32;
            let region = Region::new(0, 0, width, 1);
            let after_h = run_h(&samples, region, region, vec![1.0]);
            let after_v = run_v(&after_h, region, region, vec![1.0]);

            for (actual, expected) in after_v.iter().zip(samples.iter()) {
                prop_assert!((actual - expected).abs() < 1e-6);
            }
        }

        #[test]
        fn zero_input_stays_zero(width in 1usize..5, height in 1usize..5) {
            let kernel = vec![0.25, 0.5, 0.25];
            let h_in_region = Region::new(0, 0, (width + 2) as u32, height as u32);
            let h_out_region = Region::new(0, 0, width as u32, height as u32);
            let h_input = vec![0.0f32; (width + 2) * height];
            let after_h = run_h(&h_input, h_in_region, h_out_region, kernel.clone());

            let v_in_region = Region::new(0, 0, width as u32, (height + 2) as u32);
            let v_out_region = Region::new(0, 0, width as u32, height as u32);
            let v_input = vec![0.0f32; width * (height + 2)];
            let after_v = run_v(&v_input, v_in_region, v_out_region, kernel);

            prop_assert!(after_h.iter().all(|value| value.abs() < 1e-6));
            prop_assert!(after_v.iter().all(|value| value.abs() < 1e-6));
        }

        #[test]
        fn symmetric_kernel_commutes_with_horizontal_reflection(samples in prop::collection::vec(-10.0f32..10.0, 1..16)) {
            let kernel = vec![0.25, 0.5, 0.25];
            let radius = 1usize;
            let input = edge_extend_scanline(&samples, radius);
            let reversed_samples = reverse_scanline(&samples);
            let reversed_input = edge_extend_scanline(&reversed_samples, radius);
            let region = Region::new(-(radius as i32), 0, input.len() as u32, 1);
            let out_region = Region::new(0, 0, samples.len() as u32, 1);

            let forward = run_h(&input, region, out_region, kernel.clone());
            let reversed = run_h(&reversed_input, region, out_region, kernel);

            prop_assert_eq!(forward.len(), reversed.len());
            for (lhs, rhs) in forward.iter().zip(reversed.iter().rev()) {
                prop_assert!((lhs - rhs).abs() < 1e-5);
            }
        }
    }
}
