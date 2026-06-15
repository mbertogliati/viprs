//! Arithmetic coefficient pre-analyzed for optimal SIMD width.
//!
//! Ops store [`OptimizedCoeff`] instead of `f64`. At construction, it
//! pre-computes whether the value is an exact integer and caches f32.
//! [`PointSample`] impls call [`as_int::<T>()`](OptimizedCoeff::as_int)
//! to narrow to the optimal arithmetic width — ops carry zero
//! specialization logic.
//!
//! ```rust,ignore
//! // Op stores OptimizedCoeff, has no narrowing logic:
//! pub struct Linear { scale: OptimizedCoeff, offset: OptimizedCoeff }
//!
//! // PointSample for u8 narrows via as_int::<i16>():
//! fn pt_linear(self, scale: OptimizedCoeff, offset: OptimizedCoeff) -> u8 {
//!     if let (Some(s), Some(o)) = (scale.as_int::<i16>(), offset.as_int::<i16>()) {
//!         ((self as i16) * s + o).clamp(0, 255) as u8   // 8 NEON lanes
//!     } else {
//!         (self as f32).mul_add(scale.as_f32(), ...) as u8 // 4 NEON lanes
//!     }
//! }
//! ```
//!
//! LLVM hoists the narrowing branch out of the loop, emitting separate
//! vectorized paths. The branch cost is zero at runtime.
//!
//! [`PointSample`]: crate::domain::format::PointSample

#![allow(clippy::struct_field_names)]
// REASON: coefficient tables use domain-standard field names that match libvips terminology.

/// Wraps an `f64` coefficient with pre-computed narrower representations.
///
/// Constructed once at op creation time. [`PointSample`] impls call
/// [`as_int::<T>()`](OptimizedCoeff::as_int) to narrow to any integer type
/// (u8, i8, u16, i16, u32, i32) in a single generic method.
///
/// [`PointSample`]: crate::domain::format::PointSample
/// Pre-analyzed arithmetic coefficient with cached narrow representations.
///
/// Point operations use this to pick fast integer or float paths once, outside the hot sample
/// loop.
///
/// # Examples
/// ```rust
/// # use viprs::domain::coeff::OptimizedCoeff;
/// let coeff = OptimizedCoeff::new(2.0);
/// assert_eq!(coeff.as_int::<i16>(), Some(2));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct OptimizedCoeff {
    f64_val: f64,
    f32_val: f32,
    /// Exact integer value (if the f64 is a whole number in i64 range).
    int_val: Option<i64>,
}

impl OptimizedCoeff {
    #[inline(always)]
    #[must_use]
    /// Creates a new `OptimizedCoeff`.
    pub fn new(v: f64) -> Self {
        Self {
            f64_val: v,
            f32_val: v as f32,
            int_val: Self::try_exact_int(v),
        }
    }

    fn try_exact_int(v: f64) -> Option<i64> {
        if !v.is_finite() {
            return None;
        }
        // i64 range is [-2^63, 2^63-1]. f64 can represent integers exactly up to 2^53.
        // For safety, only accept values whose round-trip through i64 is exact.
        let i = v as i64;
        if i as f64 == v { Some(i) } else { None }
    }

    #[inline(always)]
    #[must_use]
    /// Returns this value as f64.
    pub const fn as_f64(self) -> f64 {
        self.f64_val
    }

    #[inline(always)]
    #[must_use]
    /// Returns this value as f32.
    pub const fn as_f32(self) -> f32 {
        self.f32_val
    }

    /// Narrow to any integer type. Returns `Some(T)` if the value is an exact
    /// integer representable in `T`.
    ///
    /// Works for u8, i8, u16, i16, u32, i32, u64, i64.
    #[inline(always)]
    #[must_use]
    pub fn as_int<T: TryFrom<i64>>(self) -> Option<T> {
        T::try_from(self.int_val?).ok()
    }

    /// Whether this coefficient is an exact integer.
    #[inline(always)]
    #[must_use]
    pub const fn is_integer(self) -> bool {
        self.int_val.is_some()
    }

    /// Check if `sample * scale + offset` fits in i16 for unsigned samples
    /// in `[0, max_sample]`.
    ///
    /// Used by `PointSample` impls — not by individual ops.
    #[inline(always)]
    #[must_use]
    pub fn i16_mul_add_unsigned(scale: Self, offset: Self, max_sample: i32) -> Option<(i16, i16)> {
        let si: i16 = scale.as_int()?;
        let oi: i16 = offset.as_int()?;
        let a = i32::from(oi);
        let b = max_sample * i32::from(si) + a;
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if lo >= i32::from(i16::MIN) && hi <= i32::from(i16::MAX) {
            Some((si, oi))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrows_to_all_integer_types() {
        let c = OptimizedCoeff::new(42.0);
        assert_eq!(c.as_int::<u8>(), Some(42u8));
        assert_eq!(c.as_int::<i8>(), Some(42i8));
        assert_eq!(c.as_int::<u16>(), Some(42u16));
        assert_eq!(c.as_int::<i16>(), Some(42i16));
        assert_eq!(c.as_int::<u32>(), Some(42u32));
        assert_eq!(c.as_int::<i32>(), Some(42i32));
        assert_eq!(c.as_f32(), 42.0);
        assert_eq!(c.as_f64(), 42.0);
    }

    #[test]
    fn negative_narrows_to_signed_only() {
        let c = OptimizedCoeff::new(-10.0);
        assert_eq!(c.as_int::<i8>(), Some(-10i8));
        assert_eq!(c.as_int::<i16>(), Some(-10i16));
        assert_eq!(c.as_int::<i32>(), Some(-10i32));
        assert_eq!(c.as_int::<u8>(), None);
        assert_eq!(c.as_int::<u16>(), None);
    }

    #[test]
    fn respects_type_range() {
        assert_eq!(OptimizedCoeff::new(255.0).as_int::<u8>(), Some(255u8));
        assert_eq!(OptimizedCoeff::new(256.0).as_int::<u8>(), None);
        assert_eq!(OptimizedCoeff::new(256.0).as_int::<u16>(), Some(256u16));
        assert_eq!(OptimizedCoeff::new(40000.0).as_int::<i16>(), None);
        assert_eq!(OptimizedCoeff::new(40000.0).as_int::<u16>(), Some(40000u16));
        assert_eq!(OptimizedCoeff::new(40000.0).as_int::<i32>(), Some(40000i32));
    }

    #[test]
    fn fractional_never_narrows() {
        let c = OptimizedCoeff::new(2.5);
        assert_eq!(c.as_int::<i32>(), None);
        assert_eq!(c.as_int::<i64>(), None);
        assert!(!c.is_integer());
    }

    #[test]
    fn non_finite_never_narrows() {
        assert_eq!(OptimizedCoeff::new(f64::INFINITY).as_int::<i64>(), None);
        assert_eq!(OptimizedCoeff::new(f64::NAN).as_int::<i64>(), None);
        assert!(!OptimizedCoeff::new(f64::INFINITY).is_integer());
    }

    #[test]
    fn is_integer_works() {
        assert!(OptimizedCoeff::new(3.0).is_integer());
        assert!(OptimizedCoeff::new(-100.0).is_integer());
        assert!(OptimizedCoeff::new(50000.0).is_integer());
        assert!(!OptimizedCoeff::new(1.5).is_integer());
    }

    #[test]
    fn i16_mul_add_fits() {
        let scale = OptimizedCoeff::new(2.0);
        let offset = OptimizedCoeff::new(5.0);
        assert_eq!(
            OptimizedCoeff::i16_mul_add_unsigned(scale, offset, 255),
            Some((2, 5))
        );
    }

    #[test]
    fn i16_mul_add_overflow() {
        let scale = OptimizedCoeff::new(200.0);
        let offset = OptimizedCoeff::new(0.0);
        assert_eq!(
            OptimizedCoeff::i16_mul_add_unsigned(scale, offset, 255),
            None
        );
    }

    #[test]
    fn i16_mul_add_fractional_scale() {
        let scale = OptimizedCoeff::new(1.5);
        let offset = OptimizedCoeff::new(0.0);
        assert_eq!(
            OptimizedCoeff::i16_mul_add_unsigned(scale, offset, 255),
            None
        );
    }

    #[test]
    fn i16_mul_add_negative_offset_fits() {
        let scale = OptimizedCoeff::new(1.0);
        let offset = OptimizedCoeff::new(-20.0);
        assert_eq!(
            OptimizedCoeff::i16_mul_add_unsigned(scale, offset, 255),
            Some((1, -20))
        );
    }
}
