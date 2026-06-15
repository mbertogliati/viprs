use std::any::Any;

use crate::{
    domain::op::{DynOperation, NodeSpec},
    domain::{
        format::BandFormatId,
        image::{DemandHint, Region},
    },
};

/// Form an interleaved complex image from real and imaginary inputs.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::complex_form::ComplexFormOp;
///
/// let op = ComplexFormOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ComplexFormOp {
    component_bands: u32,
}

impl ComplexFormOp {
    #[must_use]
    /// Creates a new `ComplexFormOp`.
    pub fn new(component_bands: u32) -> Self {
        debug_assert!(
            component_bands > 0,
            "ComplexFormOp: component_bands must be positive"
        );
        Self { component_bands }
    }
}

impl DynOperation for ComplexFormOp {
    fn input_format(&self) -> BandFormatId {
        BandFormatId::F32
    }

    fn output_format(&self) -> BandFormatId {
        BandFormatId::F32
    }

    fn bands(&self) -> u32 {
        self.component_bands * 2
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
            "ComplexFormOp: single-input fallback is invalid for a 2-input node"
        );
    }

    #[inline]
    fn dyn_process_region_multi(
        &self,
        _state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        _input_regions: &[Region],
        output_region: Region,
    ) {
        debug_assert_eq!(inputs.len(), 2, "ComplexFormOp: expected two input slices");
        let (Some(&real_bytes), Some(&imag_bytes)) = (inputs.first(), inputs.get(1)) else {
            debug_assert!(false, "ComplexFormOp: missing input slices");
            return;
        };

        let real: &[f32] = if let Ok(samples) = bytemuck::try_cast_slice(real_bytes) {
            samples
        } else {
            debug_assert!(false, "ComplexFormOp: real input cast failed");
            return;
        };
        let imag: &[f32] = if let Ok(samples) = bytemuck::try_cast_slice(imag_bytes) {
            samples
        } else {
            debug_assert!(false, "ComplexFormOp: imaginary input cast failed");
            return;
        };
        let out: &mut [f32] = if let Ok(samples) = bytemuck::try_cast_slice_mut(output) {
            samples
        } else {
            debug_assert!(false, "ComplexFormOp: output cast failed");
            return;
        };

        let pixels = output_region.pixel_count();
        let component_bands = self.component_bands as usize;
        debug_assert_eq!(real.len(), pixels * component_bands);
        debug_assert_eq!(imag.len(), pixels * component_bands);
        debug_assert_eq!(out.len(), pixels * component_bands * 2);

        for pixel in 0..pixels {
            let src_base = pixel * component_bands;
            let dst_base = pixel * component_bands * 2;
            for band in 0..component_bands {
                out[dst_base + band * 2] = real[src_base + band];
                out[dst_base + band * 2 + 1] = imag[src_base + band];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn interleaves_real_and_imaginary_inputs() {
        let op = ComplexFormOp::new(1);
        let real = [3.0f32];
        let imag = [4.0f32];
        let mut output = vec![0u8; std::mem::size_of::<f32>() * 2];
        let inputs: &[&[u8]] = &[bytemuck::cast_slice(&real), bytemuck::cast_slice(&imag)];
        let regions = [Region::new(0, 0, 1, 1); 2];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            &mut output,
            &regions,
            Region::new(0, 0, 1, 1),
        );
        let out: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(out, &[3.0, 4.0]);
    }

    #[test]
    fn reports_two_inputs_and_identity_node_spec() {
        let op = ComplexFormOp::new(2);
        let region = Region::new(3, 4, 5, 6);

        assert_eq!(op.input_format(), BandFormatId::F32);
        assert_eq!(op.output_format(), BandFormatId::F32);
        assert_eq!(op.bands(), 4);
        assert_eq!(op.input_slot_count(), 2);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.required_input_region_slot(&region, 0), region);
        assert_eq!(op.required_input_region_slot(&region, 1), region);
        assert_eq!(op.node_spec(64, 32), NodeSpec::identity(64, 32));
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
    }

    #[test]
    fn interleaves_multiple_pixels_and_bands() {
        let op = ComplexFormOp::new(2);
        let real = [1.0f32, 2.0, 3.0, 4.0];
        let imag = [10.0f32, 20.0, 30.0, 40.0];
        let mut output = vec![0u8; std::mem::size_of::<f32>() * 8];
        let inputs: &[&[u8]] = &[bytemuck::cast_slice(&real), bytemuck::cast_slice(&imag)];
        let regions = [Region::new(0, 0, 2, 1); 2];
        let mut state = op.dyn_start();

        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            &mut output,
            &regions,
            Region::new(0, 0, 2, 1),
        );

        let out: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(out, &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]);
    }

    #[test]
    fn single_input_fallback_rejects_invalid_arity() {
        let op = ComplexFormOp::new(1);
        let input = [1.0f32, 2.0];
        let mut output = [9.0f32, 9.0];
        let mut state = op.dyn_start();
        let region = Region::new(0, 0, 1, 1);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            op.dyn_process_region(
                state.as_mut(),
                bytemuck::cast_slice(&input),
                bytemuck::cast_slice_mut(&mut output),
                region,
                region,
            );
        }));

        if cfg!(debug_assertions) {
            assert!(result.is_err());
        } else {
            assert!(result.is_ok());
        }
        assert_eq!(output, [9.0f32, 9.0]);
    }

    proptest! {
        #[test]
        fn forms_complex_pairs(
            re in -100.0f32..100.0,
            im in -100.0f32..100.0,
        ) {
            let op = ComplexFormOp::new(1);
            let real = [re];
            let imag = [im];
            let mut output = vec![0u8; std::mem::size_of::<f32>() * 2];
            let inputs: &[&[u8]] = &[bytemuck::cast_slice(&real), bytemuck::cast_slice(&imag)];
            let regions = [Region::new(0, 0, 1, 1); 2];
            let mut state = op.dyn_start();

            op.dyn_process_region_multi(
                state.as_mut(),
                inputs,
                &mut output,
                &regions,
                Region::new(0, 0, 1, 1),
            );

            let out: &[f32] = bytemuck::cast_slice(&output);
            prop_assert_eq!(out, &[re, im]);
        }
    }
}
