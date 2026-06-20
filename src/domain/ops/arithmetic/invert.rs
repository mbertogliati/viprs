use std::marker::PhantomData;

use crate::{
    domain::op::{Op, PixelLocalOp},
    domain::{
        format::NumericBand,
        image::{DemandHint, Region, Tile, TileMut},
    },
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vdupq_n_f32, vdupq_n_u8, vdupq_n_u16, vld1q_f32, vld1q_u8, vld1q_u16, vst1q_f32, vst1q_u8,
    vst1q_u16, vsubq_f32, vsubq_u8, vsubq_u16,
};

/// Per-type inversion of a sample value.
///
/// This trait is an implementation detail of the `Invert` operation; it is not
/// part of the public port surface (`ports/`) because it is only meaningful in
/// the context of pixel-channel inversion.
pub trait Invertible: Copy {
    #[must_use]
    /// Returns or performs invert.
    fn invert(self) -> Self;

    /// Bulk inversion of a slice. Default delegates element-wise; types with SIMD
    /// acceleration override this for throughput.
    #[inline]
    fn invert_bulk(input: &[Self], output: &mut [Self]) {
        for (s, d) in input.iter().zip(output.iter_mut()) {
            *d = s.invert();
        }
    }
}

impl Invertible for u8 {
    #[inline]
    fn invert(self) -> Self {
        255 - self
    }

    #[inline]
    fn invert_bulk(input: &[Self], output: &mut [Self]) {
        invert_bulk_u8(input, output);
    }
}
impl Invertible for u16 {
    #[inline]
    fn invert(self) -> Self {
        65535 - self
    }

    #[inline]
    fn invert_bulk(input: &[Self], output: &mut [Self]) {
        invert_bulk_u16(input, output);
    }
}
impl Invertible for i16 {
    fn invert(self) -> Self {
        self.saturating_neg()
    }
}
impl Invertible for i32 {
    fn invert(self) -> Self {
        self.saturating_neg()
    }
}
impl Invertible for u32 {
    fn invert(self) -> Self {
        Self::MAX - self
    }
}
impl Invertible for f32 {
    #[inline]
    fn invert(self) -> Self {
        1.0 - self
    }

    #[inline]
    fn invert_bulk(input: &[Self], output: &mut [Self]) {
        invert_bulk_f32(input, output);
    }
}
impl Invertible for f64 {
    fn invert(self) -> Self {
        1.0 - self
    }
}

// ─── SIMD-accelerated bulk invert for u8 (NEON on aarch64) ───

#[cfg(target_arch = "aarch64")]
#[inline]
fn invert_bulk_u8(input: &[u8], output: &mut [u8]) {
    let len = input.len().min(output.len());
    let chunks = len / 64;
    let remainder = len % 64;

    // SAFETY: aarch64 always has NEON. We process 64 bytes per iteration (4×16B).
    // Pointer arithmetic stays within bounds: we only process `chunks * 64` bytes.
    unsafe {
        let max = vdupq_n_u8(255);
        let src = input.as_ptr();
        let dst = output.as_mut_ptr();

        for i in 0..chunks {
            let base = i * 64;
            let v0 = vld1q_u8(src.add(base));
            let v1 = vld1q_u8(src.add(base + 16));
            let v2 = vld1q_u8(src.add(base + 32));
            let v3 = vld1q_u8(src.add(base + 48));
            vst1q_u8(dst.add(base), vsubq_u8(max, v0));
            vst1q_u8(dst.add(base + 16), vsubq_u8(max, v1));
            vst1q_u8(dst.add(base + 32), vsubq_u8(max, v2));
            vst1q_u8(dst.add(base + 48), vsubq_u8(max, v3));
        }

        let tail_start = chunks * 64;
        // Process remaining 16-byte chunks
        let tail_chunks_16 = remainder / 16;
        for i in 0..tail_chunks_16 {
            let off = tail_start + i * 16;
            let v = vld1q_u8(src.add(off));
            vst1q_u8(dst.add(off), vsubq_u8(max, v));
        }

        // Scalar remainder
        let scalar_start = tail_start + tail_chunks_16 * 16;
        for i in scalar_start..len {
            *output.get_unchecked_mut(i) = 255 - *input.get_unchecked(i);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn invert_bulk_u8(input: &[u8], output: &mut [u8]) {
    for (s, d) in input.iter().zip(output.iter_mut()) {
        *d = 255 - *s;
    }
}

// ─── SIMD-accelerated bulk invert for u16 (NEON on aarch64) ───

#[cfg(target_arch = "aarch64")]
#[inline]
fn invert_bulk_u16(input: &[u16], output: &mut [u16]) {
    let len = input.len().min(output.len());
    let chunks = len / 32;
    let remainder = len % 32;

    // SAFETY: aarch64 NEON processes 8 u16 per 128-bit register. We unroll 4×.
    unsafe {
        let max = vdupq_n_u16(65535);
        let src = input.as_ptr();
        let dst = output.as_mut_ptr();

        for i in 0..chunks {
            let base = i * 32;
            let v0 = vld1q_u16(src.add(base));
            let v1 = vld1q_u16(src.add(base + 8));
            let v2 = vld1q_u16(src.add(base + 16));
            let v3 = vld1q_u16(src.add(base + 24));
            vst1q_u16(dst.add(base), vsubq_u16(max, v0));
            vst1q_u16(dst.add(base + 8), vsubq_u16(max, v1));
            vst1q_u16(dst.add(base + 16), vsubq_u16(max, v2));
            vst1q_u16(dst.add(base + 24), vsubq_u16(max, v3));
        }

        let tail_start = chunks * 32;
        let tail_8 = remainder / 8;
        for i in 0..tail_8 {
            let off = tail_start + i * 8;
            let v = vld1q_u16(src.add(off));
            vst1q_u16(dst.add(off), vsubq_u16(max, v));
        }

        let scalar_start = tail_start + tail_8 * 8;
        for i in scalar_start..len {
            *output.get_unchecked_mut(i) = 65535 - *input.get_unchecked(i);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn invert_bulk_u16(input: &[u16], output: &mut [u16]) {
    for (s, d) in input.iter().zip(output.iter_mut()) {
        *d = 65535 - *s;
    }
}

// ─── SIMD-accelerated bulk invert for f32 (NEON on aarch64) ───

#[cfg(target_arch = "aarch64")]
#[inline]
fn invert_bulk_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    let chunks = len / 16;
    let remainder = len % 16;

    // SAFETY: NEON processes 4 f32 per 128-bit register. Unroll 4×.
    unsafe {
        let one = vdupq_n_f32(1.0);
        let src = input.as_ptr();
        let dst = output.as_mut_ptr();

        for i in 0..chunks {
            let base = i * 16;
            let v0 = vld1q_f32(src.add(base));
            let v1 = vld1q_f32(src.add(base + 4));
            let v2 = vld1q_f32(src.add(base + 8));
            let v3 = vld1q_f32(src.add(base + 12));
            vst1q_f32(dst.add(base), vsubq_f32(one, v0));
            vst1q_f32(dst.add(base + 4), vsubq_f32(one, v1));
            vst1q_f32(dst.add(base + 8), vsubq_f32(one, v2));
            vst1q_f32(dst.add(base + 12), vsubq_f32(one, v3));
        }

        let tail_start = chunks * 16;
        let tail_4 = remainder / 4;
        for i in 0..tail_4 {
            let off = tail_start + i * 4;
            let v = vld1q_f32(src.add(off));
            vst1q_f32(dst.add(off), vsubq_f32(one, v));
        }

        let scalar_start = tail_start + tail_4 * 4;
        for i in scalar_start..len {
            *output.get_unchecked_mut(i) = 1.0 - *input.get_unchecked(i);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn invert_bulk_f32(input: &[f32], output: &mut [f32]) {
    for (s, d) in input.iter().zip(output.iter_mut()) {
        *d = 1.0 - *s;
    }
}

/// Element-wise inversion of all samples in a tile.
///
/// The exact inversion semantic is type-dependent (see `Invertible` impls above).
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::invert::Invert;
///
/// let op = Invert::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Invert<F: NumericBand> {
    _format: PhantomData<F>,
}

impl<F: NumericBand> Invert<F>
where
    F::Sample: Invertible,
{
    #[must_use]
    /// Creates a new `Invert`.
    pub const fn new() -> Self {
        Self {
            _format: PhantomData,
        }
    }
}

impl<F: NumericBand> Default for Invert<F>
where
    F::Sample: Invertible,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Op for Invert<F>
where
    F: NumericBand,
    F::Sample: Invertible,
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

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        F::Sample::invert_bulk(input.data, output.data);
    }
}

/// `Invert` is pixel-local: it reads one sample and writes one sample with no
/// neighbourhood access and identity tile geometry. See `PixelLocalOp` for invariants.
impl<F> PixelLocalOp for Invert<F>
where
    F: NumericBand,
    F::Sample: Invertible,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::F32, image::Region};
    use proptest::prelude::*;

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn invert_f32_known_values() {
        let op = Invert::<F32>::new();
        let r = make_region(2, 1);
        let input_data = vec![0.0f32, 1.0];
        let mut output_data = vec![0.0f32; 2];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 1.0).abs() < f32::EPSILON);
        assert!((output_data[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn metadata_and_all_numeric_invertible_types_match_contract() {
        use crate::domain::format::U8;

        assert_eq!(Invertible::invert(0u8), 255);
        assert_eq!(Invertible::invert(0u16), 65_535);
        assert_eq!(Invertible::invert(-7i16), 7);
        assert_eq!(Invertible::invert(-9i32), 9);
        assert_eq!(Invertible::invert(11u32), u32::MAX - 11);
        assert!((Invertible::invert(0.25f32) - 0.75).abs() < f32::EPSILON);
        assert!((Invertible::invert(0.25f64) - 0.75).abs() < f64::EPSILON);

        let op = Invert::<U8>::new();
        let region = Region::new(3, -2, 4, 5);
        assert_eq!(
            op.demand_hint(),
            crate::domain::image::DemandHint::ThinStrip
        );
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    #[test]
    fn default_constructor_matches_new() {
        use crate::domain::format::U16;

        let region = Region::new(0, 0, 3, 1);
        let input_data = [1u16, 1024, 65_535];
        let mut output_data = [0u16; 3];
        let input = Tile::<U16>::new(region, 1, &input_data);
        let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
        let mut state = ();

        Invert::<U16>::default().process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, [65_534, 64_511, 0]);
    }

    #[test]
    fn invert_signed_min_saturates_to_max() {
        use crate::domain::format::{I16, I32};

        let region = Region::new(0, 0, 1, 1);

        let op_i16 = Invert::<I16>::new();
        let input_i16 = [i16::MIN];
        let mut output_i16 = [0i16; 1];
        let input = Tile::<I16>::new(region, 1, &input_i16);
        let mut output = TileMut::<I16>::new(region, 1, &mut output_i16);
        let mut state = ();
        op_i16.process_region(&mut state, &input, &mut output);
        assert_eq!(output_i16, [i16::MAX]);

        let op_i32 = Invert::<I32>::new();
        let input_i32 = [i32::MIN];
        let mut output_i32 = [0i32; 1];
        let input = Tile::<I32>::new(region, 1, &input_i32);
        let mut output = TileMut::<I32>::new(region, 1, &mut output_i32);
        let mut state = ();
        op_i32.process_region(&mut state, &input, &mut output);
        assert_eq!(output_i32, [i32::MAX]);
    }

    #[test]
    fn invert_large_rows_cover_simd_dispatch_and_scalar_remainder() {
        use crate::domain::format::U8;

        let op = Invert::<U8>::new();
        let r = make_region(33, 1);
        let input_data = (0u8..33).collect::<Vec<_>>();
        let mut output_data = vec![0u8; input_data.len()];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        let expected: Vec<u8> = input_data.iter().map(|&sample| 255 - sample).collect();
        assert_eq!(output_data, expected);
    }

    #[test]
    fn invert_u16_bulk_path_covers_vector_tail_and_scalar_remainder() {
        use crate::domain::format::U16;

        let op = Invert::<U16>::new();
        let input_data = (0u16..43)
            .map(|sample| sample.saturating_mul(977))
            .collect::<Vec<_>>();
        let mut output_data = vec![0u16; input_data.len()];
        let region = make_region(input_data.len() as u32, 1);
        let input = Tile::<U16>::new(region, 1, &input_data);
        let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        let expected = input_data
            .iter()
            .map(|&sample| 65_535 - sample)
            .collect::<Vec<_>>();
        assert_eq!(output_data, expected);
    }

    #[test]
    fn invert_f32_bulk_path_covers_vector_tail_and_scalar_remainder() {
        let op = Invert::<F32>::new();
        let input_data = (0..21).map(|idx| idx as f32 / 20.0).collect::<Vec<_>>();
        let mut output_data = vec![0.0f32; input_data.len()];
        let region = make_region(input_data.len() as u32, 1);
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        for (actual, expected) in output_data
            .iter()
            .zip(input_data.iter().map(|&sample| 1.0 - sample))
        {
            assert!((actual - expected).abs() < f32::EPSILON);
        }
    }

    proptest! {
        #[test]
        fn invert_twice_is_identity_f32(
            pixels in proptest::collection::vec(0.0f32..=1.0f32, 1..=64)
        ) {
            let len = pixels.len();
            let op = Invert::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);

            // First inversion
            let mut mid = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut mid);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            // Second inversion
            let mut result = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &mid);
                let mut output = TileMut::<F32>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            for (a, b) in pixels.iter().zip(result.iter()) {
                prop_assert!((a - b).abs() < 1e-6, "double-invert mismatch: {} vs {}", a, b);
            }
        }

        /// For u8: invert(invert(x)) == x for all x in 0..=255.
        #[test]
        fn invert_twice_is_identity_u8(
            pixels in proptest::collection::vec(0u8..=255u8, 1..=64)
        ) {
            use crate::domain::format::U8;

            let len = pixels.len();
            let op = Invert::<U8>::new();
            let r = Region::new(0, 0, len as u32, 1);

            // First inversion.
            let mut mid = vec![0u8; len];
            {
                let input = Tile::<U8>::new(r, 1, &pixels);
                let mut output = TileMut::<U8>::new(r, 1, &mut mid);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            // Second inversion.
            let mut result = vec![0u8; len];
            {
                let input = Tile::<U8>::new(r, 1, &mid);
                let mut output = TileMut::<U8>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            prop_assert_eq!(result, pixels, "double-invert must be identity for u8");
        }
    }

    /// Boundary value: invert(0u8) == 255, invert(255u8) == 0.
    #[test]
    fn invert_u8_boundary_values() {
        use crate::domain::format::U8;

        let op = Invert::<U8>::new();
        let r = Region::new(0, 0, 2, 1);
        let input_data = vec![0u8, 255u8];
        let mut output_data = vec![0u8; 2];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0], 255u8, "invert(0) must be 255");
        assert_eq!(output_data[1], 0u8, "invert(255) must be 0");
    }

    /// Ported from libvips test_arithmetic.py::test_invert.
    ///
    /// libvips test: `~x & 0xff` for uchar format, applied per-pixel.
    /// libvips only tests invert on uchar (`fmt=[pyvips.BandFormat.UCHAR]`)
    /// because the max-value trim makes it hard to compare other formats.
    /// This test verifies the per-pixel contract for a known 3-band sRGB-like tile.
    #[test]
    fn invert_u8_multiband_pixel_contract() {
        use crate::domain::format::U8;

        // 2 pixels × 3 bands (RGB layout)
        // Pixel 0: [10, 20, 30] → invert → [245, 235, 225]
        // Pixel 1: [100, 150, 200] → invert → [155, 105, 55]
        let op = Invert::<U8>::new();
        let r = Region::new(0, 0, 2, 1);
        let input_data = vec![10u8, 20, 30, 100, 150, 200];
        let mut output_data = vec![0u8; 6];
        let input = Tile::<U8>::new(r, 3, &input_data);
        let mut output = TileMut::<U8>::new(r, 3, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0], 245, "band 0 px 0");
        assert_eq!(output_data[1], 235, "band 1 px 0");
        assert_eq!(output_data[2], 225, "band 2 px 0");
        assert_eq!(output_data[3], 155, "band 0 px 1");
        assert_eq!(output_data[4], 105, "band 1 px 1");
        assert_eq!(output_data[5], 55, "band 2 px 1");
    }

    /// Ported from libvips test_arithmetic.py::test_invert.
    ///
    /// libvips test: for u16 format, invert(x) = 65535 - x.
    /// Boundary: invert(0) = 65535, invert(65535) = 0.
    #[test]
    fn invert_u16_boundary_values() {
        use crate::domain::format::U16;

        let op = Invert::<U16>::new();
        let r = Region::new(0, 0, 3, 1);
        let input_data = vec![0u16, 32768, 65535];
        let mut output_data = vec![0u16; 3];
        let input = Tile::<U16>::new(r, 1, &input_data);
        let mut output = TileMut::<U16>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0], 65535u16, "invert(0u16) must be 65535");
        assert_eq!(output_data[1], 32767u16, "invert(32768u16) must be 32767");
        assert_eq!(output_data[2], 0u16, "invert(65535u16) must be 0");
    }
}
