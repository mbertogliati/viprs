use std::marker::PhantomData;

use crate::relational::CmpSample;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vceqq_u8, vld1q_u8, vst1q_u8};

use viprs_core::{
    format::{NumericBand, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

#[inline]
fn process_equal_region<S: CmpSample + bytemuck::Pod>(input: &[S], rhs: S, output: &mut [u8]) {
    if std::mem::size_of::<S>() == std::mem::size_of::<u8>()
        && let Ok(src) = bytemuck::try_cast_slice::<S, u8>(input)
    {
        let rhs_byte = bytemuck::bytes_of(&rhs)[0];
        eq_mask_bulk_u8(src, rhs_byte, output);
        return;
    }

    for (sample, dst) in input.iter().zip(output.iter_mut()) {
        *dst = if *sample == rhs { u8::MAX } else { 0 };
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn eq_mask_bulk_u8(input: &[u8], rhs: u8, output: &mut [u8]) {
    let len = input.len().min(output.len());
    let chunks = len / 64;
    let remainder = len % 64;

    // SAFETY: AArch64 guarantees NEON support. The loop only loads and stores
    // complete 16-byte chunks inside the slice bounds; the scalar tail covers the rest.
    unsafe {
        let rhs_vec = std::arch::aarch64::vdupq_n_u8(rhs);
        let src = input.as_ptr();
        let dst = output.as_mut_ptr();

        for chunk in 0..chunks {
            let base = chunk * 64;
            let v0 = vld1q_u8(src.add(base));
            let v1 = vld1q_u8(src.add(base + 16));
            let v2 = vld1q_u8(src.add(base + 32));
            let v3 = vld1q_u8(src.add(base + 48));

            vst1q_u8(dst.add(base), vceqq_u8(v0, rhs_vec));
            vst1q_u8(dst.add(base + 16), vceqq_u8(v1, rhs_vec));
            vst1q_u8(dst.add(base + 32), vceqq_u8(v2, rhs_vec));
            vst1q_u8(dst.add(base + 48), vceqq_u8(v3, rhs_vec));
        }

        let tail_start = chunks * 64;
        let tail_chunks = remainder / 16;
        for chunk in 0..tail_chunks {
            let offset = tail_start + chunk * 16;
            let values = vld1q_u8(src.add(offset));
            vst1q_u8(dst.add(offset), vceqq_u8(values, rhs_vec));
        }

        let scalar_start = tail_start + tail_chunks * 16;
        for index in scalar_start..len {
            *output.get_unchecked_mut(index) = if *input.get_unchecked(index) == rhs {
                u8::MAX
            } else {
                0
            };
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn eq_mask_bulk_u8(input: &[u8], rhs: u8, output: &mut [u8]) {
    for (sample, dst) in input.iter().zip(output.iter_mut()) {
        *dst = if *sample == rhs { u8::MAX } else { 0 };
    }
}

/// Element-wise equality comparison against a scalar.
///
/// Each sample is compared to `rhs`. The output is `255` if equal and `0`
/// otherwise, matching libvips relational semantics.
///
/// For float formats, equality is exact bit-level comparison (no epsilon). Use
/// this for integer formats where equality is well-defined. For float approximate
/// equality, apply a threshold before using this op.
pub struct Equal<F: NumericBand>
where
    F::Sample: CmpSample,
{
    rhs: F::Sample,
    _format: PhantomData<F>,
}

impl<F: NumericBand> Equal<F>
where
    F::Sample: CmpSample,
{
    /// Creates a new `Equal`.
    pub const fn new(rhs: F::Sample) -> Self {
        Self {
            rhs,
            _format: PhantomData,
        }
    }
}

impl<F> Op for Equal<F>
where
    F: NumericBand,
    F::Sample: CmpSample,
{
    type Input = F;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<U8>) {
        process_equal_region(input.data, self.rhs, output.data);
    }
}

/// `Equal` is pixel-local: each output sample depends only on the corresponding
/// input sample. See `PixelLocalOp` for invariants.
impl<F> PixelLocalOp for Equal<F>
where
    F: NumericBand,
    F::Sample: CmpSample,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U8},
        image::{DemandHint, Region},
    };

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn equal_f32_known_values() {
        let op = Equal::<F32>::new(5.0);
        let r = make_region(3, 1);
        let input_data = vec![3.0f32, 5.0, 7.0];
        let mut output_data = vec![0u8; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![0u8, 255, 0]);
    }

    #[test]
    fn equal_with_no_matches_is_all_false() {
        let op = Equal::<F32>::new(42.0);
        let r = make_region(3, 1);
        let input_data = vec![1.0f32, 2.0, 3.0];
        let mut output_data = vec![99u8; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![0u8, 0, 0]);
    }

    #[test]
    fn equal_u8_true_is_255() {
        let op = Equal::<U8>::new(42);
        let r = make_region(2, 1);
        let input_data = vec![42u8, 0];
        let mut output_data = vec![0u8; 2];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![255u8, 0]);
    }

    #[test]
    fn equal_reports_thin_strip_and_passthrough_region() {
        let op = Equal::<F32>::new(5.0);
        let region = make_region(4, 2);

        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);

        op.start();
    }

    #[test]
    fn equal_multiband_processes_each_band_independently() {
        let op = Equal::<F32>::new(5.0);
        let region = make_region(2, 1);
        let input_data = vec![5.0f32, 4.0, 5.0, 3.0, 5.0, 6.0];
        let mut output_data = vec![0u8; input_data.len()];
        let input = Tile::<F32>::new(region, 3, &input_data);
        let mut output = TileMut::<U8>::new(region, 3, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![255u8, 0, 255, 0, 255, 0]);
    }

    #[test]
    fn equal_large_u8_rows_cover_bulk_dispatch_and_scalar_tail() {
        let op = Equal::<U8>::new(9);
        let len = 64 + 11;
        let region = make_region(len as u32, 1);
        let input_data = (0..len)
            .map(|index| if index % 3 == 0 { 9 } else { index as u8 })
            .collect::<Vec<_>>();
        let mut output_data = vec![0u8; len];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        let expected = input_data
            .iter()
            .map(|sample| if *sample == 9 { u8::MAX } else { 0 })
            .collect::<Vec<_>>();
        assert_eq!(output_data, expected);
    }

    #[test]
    fn equal_uses_exact_float_equality() {
        let rhs = 1.0f32;
        let almost_equal = f32::from_bits(rhs.to_bits() + 1);
        let op = Equal::<F32>::new(rhs);
        let region = make_region(2, 1);
        let input_data = vec![rhs, almost_equal];
        let mut output_data = vec![0u8; input_data.len()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![255u8, 0]);
    }

    proptest! {
        #[test]
        fn equal_output_is_always_true_or_false(
            pixels in proptest::collection::vec(0.0f32..=100.0f32, 1..=64),
            rhs in 0.0f32..=100.0f32,
        ) {
            let len = pixels.len();
            let op = Equal::<F32>::new(rhs);
            let r = Region::new(0, 0, len as u32, 1);
            let mut output_data = vec![0u8; len];
            let input = Tile::<F32>::new(r, 1, &pixels);
            let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            for v in &output_data {
                prop_assert!(*v == 0 || *v == 255, "output must be 0 or 255, got {}", v);
            }
        }

        #[test]
        fn equal_matches_exact_scalar_comparison(
            pixels in proptest::collection::vec(-1000.0f32..=1000.0f32, 1..=64),
            rhs in -1000.0f32..=1000.0f32,
        ) {
            let len = pixels.len();
            let op = Equal::<F32>::new(rhs);
            let region = Region::new(0, 0, len as u32, 1);
            let mut output_data = vec![0u8; len];
            let input = Tile::<F32>::new(region, 1, &pixels);
            let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
            let mut state = ();

            op.process_region(&mut state, &input, &mut output);

            for (sample, actual) in pixels.iter().zip(&output_data) {
                let expected = if *sample == rhs { u8::MAX } else { 0 };
                prop_assert_eq!(*actual, expected);
            }
        }
    }
}
