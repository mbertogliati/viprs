//! Sample conversion traits shared across resampling operations.
//!
//! `ToF64` and `FromF64` are re-exported from `viprs-core` so the spatial and composite crates can share them without cross-crate dependencies.

pub use viprs_core::shared_ops::sample_conv::{FromF64, ToF64};

/// Sample conversions shared by the fixed-point and floating-point reduce paths.
pub trait ReduceSample: ToF64 + FromF64 + Copy {
    /// Associated constant for use fixed point.
    const USE_FIXED_POINT: bool;

    /// Converts this value to i64.
    fn to_i64(self) -> i64;
    /// Creates this value from fixed i64.
    fn from_fixed_i64(v: i64) -> Self;
}

macro_rules! impl_reduce_sample_unsigned {
    ($ty:ty) => {
        impl ReduceSample for $ty {
            const USE_FIXED_POINT: bool = true;

            #[inline(always)]
            fn to_i64(self) -> i64 {
                i64::from(self)
            }

            #[inline(always)]
            fn from_fixed_i64(v: i64) -> Self {
                const ROUND_BY: i64 = 1_i64 << 11;
                ((v + ROUND_BY) >> 12).clamp(<$ty>::MIN as i64, <$ty>::MAX as i64) as $ty
            }
        }
    };
}

macro_rules! impl_reduce_sample_signed {
    ($ty:ty) => {
        impl ReduceSample for $ty {
            const USE_FIXED_POINT: bool = true;

            #[inline(always)]
            fn to_i64(self) -> i64 {
                i64::from(self)
            }

            #[inline(always)]
            fn from_fixed_i64(v: i64) -> Self {
                const ROUND_BY: i64 = 1_i64 << 11;
                let rounded = if v >= 0 {
                    (v + ROUND_BY) >> 12
                } else {
                    (v - ROUND_BY) >> 12
                };
                rounded.clamp(<$ty>::MIN as i64, <$ty>::MAX as i64) as $ty
            }
        }
    };
}

impl_reduce_sample_unsigned!(u8);
impl_reduce_sample_unsigned!(u16);
impl_reduce_sample_unsigned!(u32);
impl_reduce_sample_signed!(i16);
impl_reduce_sample_signed!(i32);

impl ReduceSample for f32 {
    const USE_FIXED_POINT: bool = false;

    #[inline(always)]
    #[allow(clippy::unreachable)]
    // REASON: float reductions always use the floating-point path and never request fixed-point conversion.
    fn to_i64(self) -> i64 {
        unreachable!("fixed-point reduce is not used for f32")
    }

    #[inline(always)]
    #[allow(clippy::unreachable)]
    // REASON: float reductions always use the floating-point path and never request fixed-point conversion.
    fn from_fixed_i64(_v: i64) -> Self {
        unreachable!("fixed-point reduce is not used for f32")
    }
}

impl ReduceSample for f64 {
    const USE_FIXED_POINT: bool = false;

    #[inline(always)]
    #[allow(clippy::unreachable)]
    // REASON: float reductions always use the floating-point path and never request fixed-point conversion.
    fn to_i64(self) -> i64 {
        unreachable!("fixed-point reduce is not used for f64")
    }

    #[inline(always)]
    #[allow(clippy::unreachable)]
    // REASON: float reductions always use the floating-point path and never request fixed-point conversion.
    fn from_fixed_i64(_v: i64) -> Self {
        unreachable!("fixed-point reduce is not used for f64")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_f64_widens_supported_sample_types() {
        assert_eq!(u8::MAX.to_f64(), 255.0);
        assert_eq!(u16::MAX.to_f64(), 65_535.0);
        assert_eq!(i16::MIN.to_f64(), -32_768.0);
        assert_eq!(u32::MAX.to_f64(), 4_294_967_295.0);
        assert_eq!(i32::MIN.to_f64(), -2_147_483_648.0);
        assert!((1.25f32.to_f64() - 1.25).abs() < f64::EPSILON);
        assert_eq!(2.5f64.to_f64(), 2.5);
    }

    #[test]
    fn from_f64_clamps_and_rounds_integral_types() {
        assert_eq!(<u8 as FromF64>::from_f64(-1.0), 0);
        assert_eq!(<u8 as FromF64>::from_f64(255.6), u8::MAX);
        assert_eq!(<u16 as FromF64>::from_f64(12.6), 13);
        assert_eq!(<i16 as FromF64>::from_f64(-12.6), -13);
        assert_eq!(<u32 as FromF64>::from_f64(123.4), 123);
        assert_eq!(<i32 as FromF64>::from_f64(-123.6), -124);
    }

    #[test]
    fn from_f64_preserves_floating_point_types() {
        assert_eq!(<f32 as FromF64>::from_f64(1.25), 1.25f32);
        assert_eq!(<f64 as FromF64>::from_f64(-3.5), -3.5f64);
    }

    #[test]
    fn reduce_sample_fixed_point_rounds_and_clamps() {
        assert_eq!(<u8 as ReduceSample>::from_fixed_i64(12 << 12), 12);
        assert_eq!(<u16 as ReduceSample>::from_fixed_i64(513 << 12), 513);
        assert_eq!(<u32 as ReduceSample>::from_fixed_i64(42 << 12), 42);
        assert_eq!(<i16 as ReduceSample>::from_fixed_i64(-(3 << 12)), -4);
        assert_eq!(<i32 as ReduceSample>::from_fixed_i64(7 << 12), 7);
        assert_eq!(<u8 as ReduceSample>::to_i64(200), 200);
        assert_eq!(<i16 as ReduceSample>::to_i64(-7), -7);
    }

    #[test]
    fn reduce_sample_reports_fixed_point_support() {
        assert!(<u8 as ReduceSample>::USE_FIXED_POINT);
        assert!(<u16 as ReduceSample>::USE_FIXED_POINT);
        assert!(<i16 as ReduceSample>::USE_FIXED_POINT);
        assert!(<u32 as ReduceSample>::USE_FIXED_POINT);
        assert!(<i32 as ReduceSample>::USE_FIXED_POINT);
        assert!(!<f32 as ReduceSample>::USE_FIXED_POINT);
        assert!(!<f64 as ReduceSample>::USE_FIXED_POINT);
    }

    #[test]
    #[should_panic(expected = "fixed-point reduce is not used for f32")]
    fn f32_to_i64_panics_when_fixed_point_is_unavailable() {
        let _ = <f32 as ReduceSample>::to_i64(1.0);
    }

    #[test]
    #[should_panic(expected = "fixed-point reduce is not used for f32")]
    fn f32_from_fixed_i64_panics_when_fixed_point_is_unavailable() {
        let _ = <f32 as ReduceSample>::from_fixed_i64(1);
    }

    #[test]
    #[should_panic(expected = "fixed-point reduce is not used for f64")]
    fn f64_to_i64_panics_when_fixed_point_is_unavailable() {
        let _ = <f64 as ReduceSample>::to_i64(1.0);
    }

    #[test]
    #[should_panic(expected = "fixed-point reduce is not used for f64")]
    fn f64_from_fixed_i64_panics_when_fixed_point_is_unavailable() {
        let _ = <f64 as ReduceSample>::from_fixed_i64(1);
    }
}
