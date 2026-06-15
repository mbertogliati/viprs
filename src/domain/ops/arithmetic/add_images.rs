use crate::{
    domain::op::{DynOperation, NodeSpec},
    domain::{
        format::BandFormatId,
        image::{DemandHint, Region},
    },
};
use std::any::Any;

/// Element-wise addition of two image tiles with libvips-compatible promotion.
///
/// `AddImages` is a DAG merge node: it reads from two upstream input slots
/// and writes `output[i] = a[i] + b[i]`.
///
/// Integer inputs promote to the smallest output format that preserves the
/// possible sum range, matching `vips_add_format_table` in libvips:
/// `U8→U16`, `U16→U32`, `I16→I32`, `U32→U32`, `I32→I32`.
/// Float inputs keep their original format.
///
/// This type implements `DynOperation` directly (not via `OperationBridge`)
/// because it is a multi-input operation: `OperationBridge` bridges a
/// single-input `Op`, and there is no equivalent static trait for merge nodes.
/// The input format is stored as a `BandFormatId` and dispatched at runtime.
pub struct AddImages {
    bands: u32,
    input_format: BandFormatId,
    output_format: BandFormatId,
}

#[must_use]
/// Returns or performs add images output format.
pub const fn add_images_output_format(input_format: BandFormatId) -> BandFormatId {
    match input_format {
        BandFormatId::U8 => BandFormatId::U16,
        BandFormatId::U16 | BandFormatId::U32 => BandFormatId::U32,
        BandFormatId::I16 | BandFormatId::I32 => BandFormatId::I32,
        BandFormatId::F32 => BandFormatId::F32,
        BandFormatId::F64 => BandFormatId::F64,
    }
}

impl AddImages {
    /// Construct an `AddImages` merge node.
    ///
    /// `bands` is the channel count shared by both inputs and the output.
    /// `input_format` is the shared format of both inputs; the output format is
    /// derived from libvips' addition promotion table.
    #[must_use]
    pub const fn new(bands: u32, input_format: BandFormatId) -> Self {
        Self {
            bands,
            input_format,
            output_format: add_images_output_format(input_format),
        }
    }
}

impl DynOperation for AddImages {
    fn input_format(&self) -> BandFormatId {
        self.input_format
    }

    fn output_format(&self) -> BandFormatId {
        self.output_format
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
        // Both input slots need exactly the same region as the output:
        // pixel-local operation, no halo, no coordinate transform.
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
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        // Single-input path: copy input to output unchanged.
        // This path should not be reached for a correctly compiled pipeline —
        // the scheduler calls dyn_process_region_multi for nodes with
        // input_slot_count() == 2. It is provided only to satisfy the trait
        // contract and to avoid silent incorrect output if called by mistake.
        debug_assert!(
            false,
            "AddImages: dyn_process_region called on a 2-input node — \
             pipeline construction bug; use dyn_process_region_multi"
        );
        let len = output.len().min(input.len());
        output[..len].copy_from_slice(&input[..len]);
    }

    /// Add two input tiles element-wise.
    #[inline]
    fn dyn_process_region_multi(
        &self,
        _state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        _input_regions: &[Region],
        _output_region: Region,
    ) {
        debug_assert_eq!(
            inputs.len(),
            2,
            "AddImages: expected exactly 2 input slices"
        );

        let (Some(&a_bytes), Some(&b_bytes)) = (inputs.first(), inputs.get(1)) else {
            debug_assert!(false, "AddImages: missing input slices");
            return;
        };

        match self.input_format {
            BandFormatId::U8 => {
                let out: &mut [u16] = if let Ok(s) = bytemuck::try_cast_slice_mut(output) {
                    s
                } else {
                    debug_assert!(false, "AddImages U8: cast failed on promoted output");
                    return;
                };
                debug_assert_eq!(a_bytes.len(), out.len());
                debug_assert_eq!(b_bytes.len(), out.len());
                for ((a, b), o) in a_bytes.iter().zip(b_bytes.iter()).zip(out.iter_mut()) {
                    *o = u16::from(*a) + u16::from(*b);
                }
            }
            BandFormatId::U16 => {
                // SAFETY: BandFormatId::U16 guarantees the buffer was produced by a
                // pipeline node whose sample type is u16 (bytemuck::Pod, align 2).
                // The scheduler guarantees buffer lengths are multiples of
                // size_of::<u16>() == 2. try_cast_slice fails if alignment or
                // length is wrong, which would indicate a pipeline construction bug.
                let a: &[u16] = if let Ok(s) = bytemuck::try_cast_slice(a_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages U16: cast failed on input[0]");
                    return;
                };
                let b: &[u16] = if let Ok(s) = bytemuck::try_cast_slice(b_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages U16: cast failed on input[1]");
                    return;
                };
                let out: &mut [u32] = if let Ok(s) = bytemuck::try_cast_slice_mut(output) {
                    s
                } else {
                    debug_assert!(false, "AddImages U16: cast failed on promoted output");
                    return;
                };
                debug_assert_eq!(a.len(), out.len());
                debug_assert_eq!(b.len(), out.len());
                for ((av, bv), ov) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
                    *ov = u32::from(*av) + u32::from(*bv);
                }
            }
            BandFormatId::I16 => {
                // SAFETY: same invariant as U16 above but for i16 (Pod, align 2).
                let a: &[i16] = if let Ok(s) = bytemuck::try_cast_slice(a_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages I16: cast failed on input[0]");
                    return;
                };
                let b: &[i16] = if let Ok(s) = bytemuck::try_cast_slice(b_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages I16: cast failed on input[1]");
                    return;
                };
                let out: &mut [i32] = if let Ok(s) = bytemuck::try_cast_slice_mut(output) {
                    s
                } else {
                    debug_assert!(false, "AddImages I16: cast failed on promoted output");
                    return;
                };
                debug_assert_eq!(a.len(), out.len());
                debug_assert_eq!(b.len(), out.len());
                for ((av, bv), ov) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
                    *ov = i32::from(*av) + i32::from(*bv);
                }
            }
            BandFormatId::U32 => {
                // SAFETY: Pod, align 4. Same invariant as above.
                let a: &[u32] = if let Ok(s) = bytemuck::try_cast_slice(a_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages U32: cast failed on input[0]");
                    return;
                };
                let b: &[u32] = if let Ok(s) = bytemuck::try_cast_slice(b_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages U32: cast failed on input[1]");
                    return;
                };
                let out: &mut [u32] = if let Ok(s) = bytemuck::try_cast_slice_mut(output) {
                    s
                } else {
                    debug_assert!(false, "AddImages U32: cast failed on output");
                    return;
                };
                debug_assert_eq!(a.len(), out.len());
                debug_assert_eq!(b.len(), out.len());
                for ((av, bv), ov) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
                    *ov = av.wrapping_add(*bv);
                }
            }
            BandFormatId::I32 => {
                // SAFETY: Pod, align 4. Same invariant as above.
                let a: &[i32] = if let Ok(s) = bytemuck::try_cast_slice(a_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages I32: cast failed on input[0]");
                    return;
                };
                let b: &[i32] = if let Ok(s) = bytemuck::try_cast_slice(b_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages I32: cast failed on input[1]");
                    return;
                };
                let out: &mut [i32] = if let Ok(s) = bytemuck::try_cast_slice_mut(output) {
                    s
                } else {
                    debug_assert!(false, "AddImages I32: cast failed on output");
                    return;
                };
                debug_assert_eq!(a.len(), out.len());
                debug_assert_eq!(b.len(), out.len());
                for ((av, bv), ov) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
                    *ov = av.wrapping_add(*bv);
                }
            }
            BandFormatId::F32 => {
                // SAFETY: Pod, align 4. Same invariant as above.
                let a: &[f32] = if let Ok(s) = bytemuck::try_cast_slice(a_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages F32: cast failed on input[0]");
                    return;
                };
                let b: &[f32] = if let Ok(s) = bytemuck::try_cast_slice(b_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages F32: cast failed on input[1]");
                    return;
                };
                let out: &mut [f32] = if let Ok(s) = bytemuck::try_cast_slice_mut(output) {
                    s
                } else {
                    debug_assert!(false, "AddImages F32: cast failed on output");
                    return;
                };
                debug_assert_eq!(a.len(), out.len());
                debug_assert_eq!(b.len(), out.len());
                for ((av, bv), ov) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
                    *ov = av + bv;
                }
            }
            BandFormatId::F64 => {
                // SAFETY: Pod, align 8. Same invariant as above.
                let a: &[f64] = if let Ok(s) = bytemuck::try_cast_slice(a_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages F64: cast failed on input[0]");
                    return;
                };
                let b: &[f64] = if let Ok(s) = bytemuck::try_cast_slice(b_bytes) {
                    s
                } else {
                    debug_assert!(false, "AddImages F64: cast failed on input[1]");
                    return;
                };
                let out: &mut [f64] = if let Ok(s) = bytemuck::try_cast_slice_mut(output) {
                    s
                } else {
                    debug_assert!(false, "AddImages F64: cast failed on output");
                    return;
                };
                debug_assert_eq!(a.len(), out.len());
                debug_assert_eq!(b.len(), out.len());
                for ((av, bv), ov) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
                    *ov = av + bv;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::Region;
    use proptest::prelude::*;

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    fn sample_size(format: BandFormatId) -> usize {
        match format {
            BandFormatId::U8 => 1,
            BandFormatId::U16 | BandFormatId::I16 => 2,
            BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
            BandFormatId::F64 => 8,
        }
    }

    /// Call dyn_process_region_multi with two pre-built byte slices.
    fn run_multi(op: &AddImages, a: &[u8], b: &[u8], output: &mut [u8]) {
        let input_pixels = a.len() / sample_size(op.input_format());
        let output_pixels = output.len() / sample_size(op.output_format());
        assert_eq!(input_pixels, output_pixels);
        let inputs: &[&[u8]] = &[a, b];
        let regions = [make_region(input_pixels as u32, 1); 2];
        let out_region = make_region(output_pixels as u32, 1);
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(state.as_mut(), inputs, output, &regions, out_region);
    }

    fn run_multi_typed<In, Out>(op: &AddImages, a: &[In], b: &[In]) -> Vec<Out>
    where
        In: bytemuck::Pod + Copy,
        Out: bytemuck::Pod + Copy,
    {
        assert_eq!(a.len(), b.len());
        let mut out_bytes = vec![0u8; a.len() * std::mem::size_of::<Out>()];
        run_multi(
            op,
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            &mut out_bytes,
        );
        bytemuck::cast_slice(&out_bytes).to_vec()
    }

    #[test]
    fn input_slot_count_is_2() {
        let op = AddImages::new(1, BandFormatId::U8);
        assert_eq!(op.input_slot_count(), 2);
    }

    #[test]
    fn add_u8_promotes_to_u16_without_saturation() {
        let op = AddImages::new(1, BandFormatId::U8);
        let out = run_multi_typed::<u8, u16>(&op, &[200u8; 4], &[100u8; 4]);
        assert_eq!(out, vec![300u16; 4]);
    }

    #[test]
    fn add_u8_basic_4x4() {
        // 4×4 image, 1 band, 16 pixels.
        let op = AddImages::new(1, BandFormatId::U8);
        let a: Vec<u8> = (0u8..16).collect();
        let b: Vec<u8> = (0u8..16).map(|i| i.saturating_mul(2)).collect();
        let out = run_multi_typed::<u8, u16>(&op, &a, &b);
        let expected: Vec<u16> = (0u8..16)
            .map(|i| u16::from(i) + u16::from(i.saturating_mul(2)))
            .collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn add_u8_zero_identity() {
        let op = AddImages::new(1, BandFormatId::U8);
        let a = vec![0u8, 50, 100, 200];
        let b = vec![0u8; 4];
        let out = run_multi_typed::<u8, u16>(&op, &a, &b);
        let out: Vec<u8> = out.into_iter().map(|value| value as u8).collect();
        assert_eq!(out, a);
    }

    #[test]
    fn add_u16_promotes_to_u32() {
        let op = AddImages::new(1, BandFormatId::U16);
        let out = run_multi_typed::<u16, u32>(&op, &[60_000u16, 1, 0], &[10_000u16, 1, 0]);
        assert_eq!(out, vec![70_000u32, 2, 0]);
    }

    #[test]
    fn add_i16_promotes_to_i32() {
        let op = AddImages::new(1, BandFormatId::I16);
        let out =
            run_multi_typed::<i16, i32>(&op, &[30_000i16, -30_000, 0], &[5_000i16, -5_000, 1]);
        assert_eq!(out, vec![35_000i32, -35_000, 1]);
    }

    #[test]
    fn add_u32_wraps_like_libvips_uint_output() {
        let op = AddImages::new(1, BandFormatId::U32);
        let out = run_multi_typed::<u32, u32>(&op, &[u32::MAX - 1, 7], &[10, 3]);
        assert_eq!(out, vec![8, 10]);
    }

    #[test]
    fn add_i32_wraps_like_libvips_int_output() {
        let op = AddImages::new(1, BandFormatId::I32);
        let out = run_multi_typed::<i32, i32>(&op, &[i32::MAX - 1, i32::MIN + 1], &[10, -10]);
        assert_eq!(out, vec![i32::MIN + 8, i32::MAX - 8]);
    }

    #[test]
    fn add_f32_no_saturation() {
        let op = AddImages::new(1, BandFormatId::F32);
        let result = run_multi_typed::<f32, f32>(&op, &[0.5f32, 1.0, -1.0], &[0.7f32, 1.0, -2.0]);
        assert!(
            (result[0] - 1.2f32).abs() < 1e-6,
            "expected ≈1.2, got {}",
            result[0]
        );
        assert!((result[1] - 2.0f32).abs() < 1e-6);
        assert!((result[2] - (-3.0f32)).abs() < 1e-6);
    }

    #[test]
    fn add_f32_exceeds_255_no_saturation() {
        // Floats must not clamp — 200.0 + 100.0 = 300.0 exactly.
        let op = AddImages::new(1, BandFormatId::F32);
        let result = run_multi_typed::<f32, f32>(&op, &[200.0f32], &[100.0f32]);
        assert!((result[0] - 300.0f32).abs() < 1e-6);
    }

    #[test]
    fn add_f64_keeps_ieee_sum() {
        let op = AddImages::new(2, BandFormatId::F64);
        let out = run_multi_typed::<f64, f64>(&op, &[0.5f64, -1.5], &[2.0f64, -2.5]);
        assert_eq!(out, vec![2.5, -4.0]);
    }

    #[test]
    fn single_input_fallback_panics_in_debug_builds() {
        let op = AddImages::new(1, BandFormatId::U8);
        let input = vec![1u8, 2, 3];
        let mut output = vec![9u8; 5];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut state = op.dyn_start();
            op.dyn_process_region(
                state.as_mut(),
                &input,
                &mut output,
                make_region(3, 1),
                make_region(5, 1),
            );
        }));
        assert!(result.is_err());
    }

    #[test]
    fn malformed_second_input_panics_in_debug_builds() {
        let op = AddImages::new(1, BandFormatId::U16);
        let lhs = bytemuck::cast_slice(&[10u16, 20u16]).to_vec();
        let rhs = vec![0u8; 3];
        let mut output = vec![0u8; 8];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut state = op.dyn_start();
            op.dyn_process_region_multi(
                state.as_mut(),
                &[lhs.as_slice(), rhs.as_slice()],
                &mut output,
                &[make_region(2, 1), make_region(2, 1)],
                make_region(2, 1),
            );
        }));

        assert!(result.is_err());
    }

    #[test]
    fn malformed_output_buffer_panics_in_debug_builds() {
        let op = AddImages::new(1, BandFormatId::F32);
        let lhs = bytemuck::cast_slice(&[1.0f32, 2.0f32]).to_vec();
        let rhs = bytemuck::cast_slice(&[3.0f32, 4.0f32]).to_vec();
        let mut output = vec![7u8; 7];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut state = op.dyn_start();
            op.dyn_process_region_multi(
                state.as_mut(),
                &[lhs.as_slice(), rhs.as_slice()],
                &mut output,
                &[make_region(2, 1), make_region(2, 1)],
                make_region(2, 1),
            );
        }));

        assert!(result.is_err());
    }

    #[test]
    fn malformed_first_input_panics_in_debug_builds() {
        let op = AddImages::new(1, BandFormatId::U32);
        let lhs = vec![0u8; 3];
        let rhs = bytemuck::cast_slice(&[2u32]).to_vec();
        let mut output = bytemuck::cast_slice(&[0u32]).to_vec();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut state = op.dyn_start();
            op.dyn_process_region_multi(
                state.as_mut(),
                &[lhs.as_slice(), rhs.as_slice()],
                &mut output,
                &[make_region(1, 1), make_region(1, 1)],
                make_region(1, 1),
            );
        }));

        assert!(result.is_err());
    }

    #[test]
    fn malformed_u8_promoted_output_panics_in_debug_builds() {
        let op = AddImages::new(1, BandFormatId::U8);
        let lhs = vec![10u8, 20];
        let rhs = vec![1u8, 2];
        let mut output = vec![0u8; 3];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut state = op.dyn_start();
            op.dyn_process_region_multi(
                state.as_mut(),
                &[lhs.as_slice(), rhs.as_slice()],
                &mut output,
                &[make_region(2, 1), make_region(2, 1)],
                make_region(2, 1),
            );
        }));

        assert!(result.is_err());
    }

    #[test]
    fn add_multi_band_f64_preserves_interleaved_order() {
        let op = AddImages::new(2, BandFormatId::F64);
        let out = run_multi_typed::<f64, f64>(&op, &[1.0, 2.0, 3.0, 4.0], &[0.5, 1.5, -1.0, 2.0]);

        assert_eq!(out, vec![1.5, 3.5, 2.0, 6.0]);
    }

    #[test]
    fn output_format_matches_libvips_add_promotion_table() {
        let cases = [
            (BandFormatId::U8, BandFormatId::U16),
            (BandFormatId::U16, BandFormatId::U32),
            (BandFormatId::I16, BandFormatId::I32),
            (BandFormatId::U32, BandFormatId::U32),
            (BandFormatId::I32, BandFormatId::I32),
            (BandFormatId::F32, BandFormatId::F32),
            (BandFormatId::F64, BandFormatId::F64),
        ];

        for (input, expected_output) in cases {
            let op = AddImages::new(3, input);
            assert_eq!(op.input_format(), input);
            assert_eq!(op.output_format(), expected_output);
            assert_eq!(op.bands(), 3);
        }
    }

    #[test]
    fn demand_hint_is_thin_strip() {
        let op = AddImages::new(1, BandFormatId::U8);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
    }

    #[test]
    fn required_input_region_slot_is_identity() {
        let op = AddImages::new(1, BandFormatId::U8);
        let r = make_region(16, 8);
        assert_eq!(op.required_input_region_slot(&r, 0), r);
        assert_eq!(op.required_input_region_slot(&r, 1), r);
        assert_eq!(op.required_input_region(&r), r);
    }

    #[test]
    fn node_spec_is_identity() {
        let op = AddImages::new(1, BandFormatId::U8);
        assert_eq!(op.node_spec(64, 32), NodeSpec::identity(64, 32));
    }

    /// Ported from libvips test_arithmetic.py::test_avg.
    ///
    /// libvips test:
    ///   `im = pyvips.Image.black(50, 100)`
    ///   `test = im.insert(im + 100, 50, 0, expand=True)`
    ///   For all formats: `test.avg() == 50`
    ///
    /// This tests that adding a constant of 100 to black (zero) produces 100,
    /// and that the black tile contributes 0. The average over the composite
    /// image equals 50 (half zeros, half 100s).
    ///
    /// We test the component contract: AddImages(black, black+100) == 100 per pixel.
    #[test]
    fn black_plus_100_equals_100_u8() {
        let op = AddImages::new(1, BandFormatId::U8);
        // black tile: all zeros; black+100: all 100
        let black = vec![0u8; 8];
        let plus_100 = vec![100u8; 8];
        let out = run_multi_typed::<u8, u16>(&op, &black, &plus_100);
        assert!(
            out.iter().all(|&v| v == 100),
            "black + black+100 must be 100 everywhere, got {:?}",
            &out[..4]
        );
    }

    /// Ported from libvips test_arithmetic.py::test_avg.
    ///
    /// libvips test: same as above but for the F32 format — float addition of
    /// 0.0 + 100.0 gives 100.0 (IEEE 754, no saturation).
    #[test]
    fn black_plus_100_equals_100_f32() {
        let op = AddImages::new(1, BandFormatId::F32);
        let black: Vec<f32> = vec![0.0f32; 4];
        let plus_100: Vec<f32> = vec![100.0f32; 4];
        let result = run_multi_typed::<f32, f32>(&op, &black, &plus_100);
        for (i, v) in result.iter().enumerate() {
            assert!((v - 100.0f32).abs() < f32::EPSILON, "pixel {i}: {v}");
        }
    }

    /// Ported from libvips test_arithmetic.py::test_sub (two-image addition parity).
    #[test]
    fn add_images_u8_is_commutative() {
        let op = AddImages::new(1, BandFormatId::U8);
        let a: Vec<u8> = vec![10, 50, 200, 128];
        let b: Vec<u8> = vec![20, 100, 100, 50];

        let ab = run_multi_typed::<u8, u16>(&op, &a, &b);
        let ba = run_multi_typed::<u8, u16>(&op, &b, &a);

        assert_eq!(ab, ba, "u8 promoted add must be commutative");
    }

    proptest! {
        #[test]
        fn add_u8_preserves_sum_in_u16(a in any::<u8>(), b in any::<u8>()) {
            let op = AddImages::new(1, BandFormatId::U8);
            let out = run_multi_typed::<u8, u16>(&op, &[a], &[b]);
            prop_assert_eq!(out, vec![u16::from(a) + u16::from(b)]);
        }
    }
}
