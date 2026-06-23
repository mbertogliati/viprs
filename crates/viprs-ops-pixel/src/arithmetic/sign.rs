#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vcgtq_u8, vdupq_n_u8, vld1q_u8, vshrq_n_u8, vst1q_u8};

use viprs_core::{
    format::{AbsSample, BandFormat},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, PixelLocalOp},
};

#[inline]
fn process_sign_region<S: AbsSample + bytemuck::Pod>(input: &[S], output: &mut [S]) {
    if std::mem::size_of::<S>() == std::mem::size_of::<u8>()
        && let (Ok(src), Ok(dst)) = (
            bytemuck::try_cast_slice::<S, u8>(input),
            bytemuck::try_cast_slice_mut::<S, u8>(output),
        )
    {
        sign_bulk_u8(src, dst);
        return;
    }

    for (sample, dst) in input.iter().zip(output.iter_mut()) {
        *dst = sample.s_sign();
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn sign_bulk_u8(input: &[u8], output: &mut [u8]) {
    let len = input.len().min(output.len());
    let chunks = len / 64;
    let remainder = len % 64;

    // SAFETY: AArch64 guarantees NEON support. The loop only touches full 16-byte
    // lanes inside `0..chunks * 64`; the scalar tail handles the remainder.
    unsafe {
        let zero = vdupq_n_u8(0);
        let src = input.as_ptr();
        let dst = output.as_mut_ptr();

        for chunk in 0..chunks {
            let base = chunk * 64;
            let v0 = vld1q_u8(src.add(base));
            let v1 = vld1q_u8(src.add(base + 16));
            let v2 = vld1q_u8(src.add(base + 32));
            let v3 = vld1q_u8(src.add(base + 48));

            vst1q_u8(dst.add(base), vshrq_n_u8::<7>(vcgtq_u8(v0, zero)));
            vst1q_u8(dst.add(base + 16), vshrq_n_u8::<7>(vcgtq_u8(v1, zero)));
            vst1q_u8(dst.add(base + 32), vshrq_n_u8::<7>(vcgtq_u8(v2, zero)));
            vst1q_u8(dst.add(base + 48), vshrq_n_u8::<7>(vcgtq_u8(v3, zero)));
        }

        let tail_start = chunks * 64;
        let tail_chunks = remainder / 16;
        for chunk in 0..tail_chunks {
            let offset = tail_start + chunk * 16;
            let values = vld1q_u8(src.add(offset));
            vst1q_u8(dst.add(offset), vshrq_n_u8::<7>(vcgtq_u8(values, zero)));
        }

        let scalar_start = tail_start + tail_chunks * 16;
        for index in scalar_start..len {
            *output.get_unchecked_mut(index) = u8::from(*input.get_unchecked(index) != 0);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn sign_bulk_u8(input: &[u8], output: &mut [u8]) {
    for (sample, dst) in input.iter().zip(output.iter_mut()) {
        *dst = u8::from(*sample != 0);
    }
}

/// Sign of each pixel sample.
/// Returns -1/0/1 for signed and float types, and 0/1 for unsigned types.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::sign::Sign;
///
/// let op = Sign::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Sign<F: BandFormat>(std::marker::PhantomData<F>)
where
    F::Sample: AbsSample;

#[allow(dead_code)]
impl<F: BandFormat> Sign<F>
where
    F::Sample: AbsSample,
{
    #[must_use]
    /// Creates a new `Sign`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: BandFormat> Default for Sign<F>
where
    F::Sample: AbsSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BandFormat> Op for Sign<F>
where
    F::Sample: AbsSample,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }
    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }
    fn node_spec(&self, w: u32, h: u32) -> NodeSpec {
        NodeSpec::identity(w, h)
    }
    fn start(&self) {}
    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        process_sign_region(input.data, output.data);
    }
}

/// `Sign` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: BandFormat> PixelLocalOp for Sign<F> where F::Sample: viprs_core::format::AbsSample {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U8},
        image::Region,
    };

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn sign_f32_known_values() {
        let op = Sign::<F32>::new();
        let r = make_region(3, 1);
        let input_data = vec![-5.0f32, -0.0f32, 3.0];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - (-1.0)).abs() < f32::EPSILON);
        assert!((output_data[1] - 0.0).abs() < f32::EPSILON);
        assert!((output_data[2] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sign_metadata_matches_identity_geometry() {
        let op = Sign::<F32>::default();
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(4, 2), viprs_core::op::NodeSpec::identity(4, 2));
    }

    #[test]
    fn sign_large_rows_cover_bulk_dispatch_and_scalar_tail() {
        let op = Sign::<U8>::new();
        let len = 64 + 19;
        let region = make_region(len as u32, 1);
        let input_data = (0..len)
            .map(|index| if index % 5 == 0 { 0 } else { 255 })
            .collect::<Vec<_>>();
        let mut output_data = vec![0u8; len];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        let expected = input_data
            .iter()
            .map(|sample| u8::from(*sample != 0))
            .collect::<Vec<_>>();
        assert_eq!(output_data, expected);
    }

    #[test]
    fn sign_u8_is_zero_or_one() {
        let op = Sign::<U8>::new();
        let r = make_region(3, 1);
        let input_data = vec![0u8, 1, 255];
        let mut output_data = vec![0u8; 3];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![0u8, 1, 1]);
    }

    proptest! {
        #[test]
        fn sign_f32_result_is_minus_zero_or_plus_one(
            pixels in proptest::collection::vec(-1000.0f32..=1000.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = Sign::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut result = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            for v in &result {
                prop_assert!(
                    *v == -1.0 || *v == 0.0 || *v == 1.0,
                    "sign output not in {{-1, 0, 1}}: {}",
                    v
                );
            }
        }

        #[test]
        fn sign_idempotent(
            pixels in proptest::collection::vec(-1000.0f32..=1000.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = Sign::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut first = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut first);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            let mut second = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &first);
                let mut output = TileMut::<F32>::new(r, 1, &mut second);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            for (a, b) in first.iter().zip(second.iter()) {
                prop_assert!((a - b).abs() < f32::EPSILON, "sign not idempotent: {} vs {}", a, b);
            }
        }

        #[test]
        fn sign_is_identity_for_sign_outputs(
            pixels in proptest::collection::vec(prop_oneof![Just(-1.0f32), Just(0.0f32), Just(1.0f32)], 1..=32)
        ) {
            let len = pixels.len();
            let op = Sign::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut result = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            for (expected, actual) in pixels.iter().zip(result.iter()) {
                prop_assert!((expected - actual).abs() < f32::EPSILON);
            }
        }
    }
}
