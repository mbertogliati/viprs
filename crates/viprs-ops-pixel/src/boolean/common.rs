#![allow(clippy::type_complexity)]
// REASON: boolean dispatch types model exact per-format kernels and avoid heap indirection.
#![allow(clippy::debug_assert_with_mut_call)]
// REASON: debug-only assertions intentionally inspect mutable tile views without changing release codegen.

use crate::arithmetic::rhs_broadcast::{RhsLayout, detect_rhs_layout};
use viprs_core::{
    error::BooleanError,
    format::{BandFormat, F32, F64, I16, I32, U8, U16, U32},
    image::{Tile, TileMut},
};

pub type BooleanOutput<L, R> = <L as BooleanOperand<R>>::Output;
pub type BooleanOutputSample<L, R> = <<L as BooleanOperand<R>>::Output as BandFormat>::Sample;

pub trait BooleanOperand<Rhs: BandFormat>: BandFormat {
    type Output: BandFormat;

    fn cast_left(sample: Self::Sample) -> <Self::Output as BandFormat>::Sample;
    fn cast_right(sample: Rhs::Sample) -> <Self::Output as BandFormat>::Sample;
}

pub trait BooleanResultSample: Copy + 'static {
    #[must_use]
    fn bool_and(self, rhs: Self) -> Self;
    #[must_use]
    fn bool_or(self, rhs: Self) -> Self;
    #[must_use]
    fn bool_xor(self, rhs: Self) -> Self;
    #[must_use]
    fn bool_lshift(self, rhs: Self) -> Self;
    #[must_use]
    fn bool_rshift(self, rhs: Self) -> Self;
}

macro_rules! impl_boolean_result_unsigned {
    ($sample:ty) => {
        impl BooleanResultSample for $sample {
            #[inline(always)]
            fn bool_and(self, rhs: Self) -> Self {
                self & rhs
            }

            #[inline(always)]
            fn bool_or(self, rhs: Self) -> Self {
                self | rhs
            }

            #[inline(always)]
            fn bool_xor(self, rhs: Self) -> Self {
                self ^ rhs
            }

            #[inline(always)]
            fn bool_lshift(self, rhs: Self) -> Self {
                self.checked_shl(rhs as u32).unwrap_or(0)
            }

            #[inline(always)]
            fn bool_rshift(self, rhs: Self) -> Self {
                self.checked_shr(rhs as u32).unwrap_or(0)
            }
        }
    };
}

macro_rules! impl_boolean_result_signed {
    ($sample:ty, $unsigned:ty) => {
        impl BooleanResultSample for $sample {
            #[inline(always)]
            fn bool_and(self, rhs: Self) -> Self {
                self & rhs
            }

            #[inline(always)]
            fn bool_or(self, rhs: Self) -> Self {
                self | rhs
            }

            #[inline(always)]
            fn bool_xor(self, rhs: Self) -> Self {
                self ^ rhs
            }

            #[inline(always)]
            fn bool_lshift(self, rhs: Self) -> Self {
                let shift = rhs as u32;
                (self as $unsigned).checked_shl(shift).unwrap_or(0) as Self
            }

            #[inline(always)]
            fn bool_rshift(self, rhs: Self) -> Self {
                let shift = rhs as u32;
                if shift >= Self::BITS {
                    if self < 0 { -1 } else { 0 }
                } else {
                    self >> shift
                }
            }
        }
    };
}

impl_boolean_result_unsigned!(u8);
impl_boolean_result_unsigned!(u16);
impl_boolean_result_unsigned!(u32);
impl_boolean_result_signed!(i16, u16);
impl_boolean_result_signed!(i32, u32);

macro_rules! impl_boolean_operand {
    ($left:ty, $right:ty => $out:ty) => {
        impl BooleanOperand<$right> for $left {
            type Output = $out;

            #[inline(always)]
            fn cast_left(sample: <$left as BandFormat>::Sample) -> <$out as BandFormat>::Sample {
                sample as <$out as BandFormat>::Sample
            }

            #[inline(always)]
            fn cast_right(sample: <$right as BandFormat>::Sample) -> <$out as BandFormat>::Sample {
                sample as <$out as BandFormat>::Sample
            }
        }
    };
}

macro_rules! impl_boolean_row {
    ($left:ty => { $($right:ty => $out:ty),+ $(,)? }) => {
        $(impl_boolean_operand!($left, $right => $out);)+
    };
}

impl_boolean_row!(U8 => {
    U8 => U8,
    U16 => U8,
    I16 => U8,
    U32 => U8,
    I32 => U8,
    F32 => U8,
    F64 => U8,
});
impl_boolean_row!(U16 => {
    U8 => U16,
    U16 => U16,
    I16 => U16,
    U32 => U16,
    I32 => U16,
    F32 => U16,
    F64 => U16,
});
impl_boolean_row!(I16 => {
    U8 => I16,
    U16 => I16,
    I16 => I16,
    U32 => I16,
    I32 => I16,
    F32 => I16,
    F64 => I16,
});
impl_boolean_row!(U32 => {
    U8 => U32,
    U16 => U32,
    I16 => U32,
    U32 => U32,
    I32 => U32,
    F32 => U32,
    F64 => U32,
});
impl_boolean_row!(I32 => {
    U8 => I32,
    U16 => I32,
    I16 => I32,
    U32 => I32,
    I32 => I32,
    F32 => I32,
    F64 => I32,
});
impl_boolean_row!(F32 => {
    U8 => I32,
    U16 => I32,
    I16 => I32,
    U32 => I32,
    I32 => I32,
    F32 => I32,
    F64 => I32,
});
impl_boolean_row!(F64 => {
    U8 => I32,
    U16 => I32,
    I16 => I32,
    U32 => I32,
    I32 => I32,
    F32 => I32,
    F64 => I32,
});

#[inline(always)]
pub fn cast_rhs_vec<L, R>(rhs: Vec<R::Sample>) -> Vec<BooleanOutputSample<L, R>>
where
    L: BooleanOperand<R>,
    R: BandFormat,
{
    rhs.into_iter()
        .map(<L as BooleanOperand<R>>::cast_right)
        .collect()
}

#[inline]
pub fn cast_rhs_constants<L, R>(
    rhs: Vec<R::Sample>,
    bands: u32,
) -> Result<Vec<BooleanOutputSample<L, R>>, BooleanError>
where
    L: BooleanOperand<R>,
    R: BandFormat,
{
    let len = rhs.len();
    let bands_usize = bands as usize;

    if len == bands_usize {
        return Ok(cast_rhs_vec::<L, R>(rhs));
    }

    match rhs.as_slice() {
        [sample] => Ok(vec![
            <L as BooleanOperand<R>>::cast_right(*sample);
            bands_usize
        ]),
        _ => Err(BooleanError::ConstLengthMismatch { len, bands }),
    }
}

#[inline(always)]
pub fn process_boolean_region<L, R>(
    input: &Tile<L>,
    rhs: &[BooleanOutputSample<L, R>],
    output: &mut TileMut<BooleanOutput<L, R>>,
    apply: fn(BooleanOutputSample<L, R>, BooleanOutputSample<L, R>) -> BooleanOutputSample<L, R>,
) where
    L: BooleanOperand<R>,
    R: BandFormat,
    BooleanOutputSample<L, R>: BooleanResultSample,
{
    let src = input.data;
    let dst = &mut output.data;
    let bands = input.bands as usize;
    let layout = detect_rhs_layout(rhs.len(), src.len(), bands);

    debug_assert!(
        layout.is_some(),
        "Boolean ops require full-tile, scalar, per-band, or single-band-image rhs layout"
    );
    debug_assert_eq!(
        src.len(),
        dst.len(),
        "Boolean ops require matching input and output sample counts"
    );

    match layout {
        Some(RhsLayout::Direct) => {
            for ((sample, rhs_sample), dst_sample) in src.iter().zip(rhs.iter()).zip(dst.iter_mut())
            {
                *dst_sample = apply(<L as BooleanOperand<R>>::cast_left(*sample), *rhs_sample);
            }
        }
        Some(RhsLayout::Scalar) => {
            let rhs_sample = rhs[0];
            for (sample, dst_sample) in src.iter().zip(dst.iter_mut()) {
                *dst_sample = apply(<L as BooleanOperand<R>>::cast_left(*sample), rhs_sample);
            }
        }
        Some(RhsLayout::PerBand) => {
            for ((index, sample), dst_sample) in src.iter().enumerate().zip(dst.iter_mut()) {
                *dst_sample = apply(
                    <L as BooleanOperand<R>>::cast_left(*sample),
                    rhs[index % bands],
                );
            }
        }
        Some(RhsLayout::SingleBandImage) => {
            for ((index, sample), dst_sample) in src.iter().enumerate().zip(dst.iter_mut()) {
                *dst_sample = apply(
                    <L as BooleanOperand<R>>::cast_left(*sample),
                    rhs[index / bands],
                );
            }
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::format::{F32, U8};

    proptest! {
        #[test]
        fn scalar_boolean_const_expands_to_three_bands(sample in any::<u8>()) {
            let expanded = cast_rhs_constants::<U8, U8>(vec![sample], 3).unwrap();
            prop_assert_eq!(expanded, vec![sample; 3]);
        }

        #[test]
        fn matching_boolean_const_preserves_n_band_values(
            rhs in proptest::array::uniform3(any::<u8>()),
        ) {
            let rhs = rhs.to_vec();
            let expanded = cast_rhs_constants::<F32, U8>(rhs.clone(), 3).unwrap();
            let expected = rhs.into_iter().map(i32::from).collect::<Vec<i32>>();
            prop_assert_eq!(expanded, expected);
        }

        #[test]
        fn mismatched_non_scalar_boolean_const_returns_error(
            rhs in proptest::collection::vec(any::<u8>(), 2..=6),
        ) {
            prop_assume!(rhs.len() != 3);
            let err = cast_rhs_constants::<U8, U8>(rhs.clone(), 3).unwrap_err();
            prop_assert_eq!(
                err,
                BooleanError::ConstLengthMismatch {
                    len: rhs.len(),
                    bands: 3,
                }
            );
        }
    }
}
