#![allow(clippy::manual_checked_ops)]
// REASON: the explicit checked arithmetic mirrors libvips overflow semantics branch-for-branch.

// ── Sample-level capability traits ───────────────────────────────────────────
//
// These traits are implemented for the concrete primitive types (f32, u8, …)
// and are used as bounds on generic ops. They intentionally avoid `num-traits`
// to keep the dependency tree minimal.

/// Methods available on float-typed pixel samples.
///
/// Implemented for `f32` and `f64`. Used as the bound for `FloatFormat`.
///
/// All methods carry `#[inline(always)]`. A plain `#[inline]` is a hint
/// that LLVM may ignore at link-time optimisation boundaries; `#[inline(always)]`
/// guarantees the call site sees the body, which lets LLVM auto-vectorise loops
/// that call `s_ln`, `s_exp`, etc. without requiring LTO. Verification:
///   `cargo rustc --release -- --emit=asm 2>/dev/null | grep -c "vmovaps\|vaddps\|vmulps"`
/// should be non-zero for any op that calls these methods over a `&[f32]` slice.
pub trait FloatSample: Copy + 'static {
    #[must_use]
    /// Returns or performs s ln.
    fn s_ln(self) -> Self;
    #[must_use]
    /// Returns or performs s log10.
    fn s_log10(self) -> Self;
    #[must_use]
    /// Returns or performs s exp.
    fn s_exp(self) -> Self;
    #[must_use]
    /// Returns or performs s exp10.
    fn s_exp10(self) -> Self;
    #[must_use]
    /// Returns or performs s pow.
    fn s_pow(self, exp: Self) -> Self;
    #[must_use]
    /// Returns or performs s abs.
    fn s_abs(self) -> Self;
    #[must_use]
    /// Returns or performs s round.
    fn s_round(self) -> Self;
    #[must_use]
    /// Returns or performs s floor.
    fn s_floor(self) -> Self;
    #[must_use]
    /// Returns or performs s ceil.
    fn s_ceil(self) -> Self;
    /// Returns -1.0, 0.0, or 1.0.
    #[must_use]
    fn s_sign(self) -> Self;
    #[must_use]
    /// Returns or performs s sin.
    fn s_sin(self) -> Self;
    #[must_use]
    /// Returns or performs s cos.
    fn s_cos(self) -> Self;
    #[must_use]
    /// Returns or performs s tan.
    fn s_tan(self) -> Self;
    #[must_use]
    /// Returns or performs s asin.
    fn s_asin(self) -> Self;
    #[must_use]
    /// Returns or performs s acos.
    fn s_acos(self) -> Self;
    #[must_use]
    /// Returns or performs s atan.
    fn s_atan(self) -> Self;
    #[must_use]
    /// Returns or performs s sqrt.
    fn s_sqrt(self) -> Self;
}

impl FloatSample for f32 {
    #[inline(always)]
    fn s_ln(self) -> Self {
        if self == 0.0 { 0.0 } else { Self::ln(self) }
    }
    #[inline(always)]
    fn s_log10(self) -> Self {
        if self == 0.0 { 0.0 } else { Self::log10(self) }
    }
    #[inline(always)]
    fn s_exp(self) -> Self {
        Self::exp(self)
    }
    #[inline(always)]
    fn s_exp10(self) -> Self {
        Self::powf(10.0_f32, self)
    }
    #[inline(always)]
    fn s_pow(self, exp: Self) -> Self {
        if self == 0.0 {
            0.0
        } else if exp == -1.0 {
            1.0 / self
        } else if exp == 0.5 {
            Self::sqrt(self)
        } else {
            Self::powf(self, exp)
        }
    }
    #[inline(always)]
    fn s_abs(self) -> Self {
        Self::abs(self)
    }
    #[inline(always)]
    fn s_round(self) -> Self {
        Self::round_ties_even(self)
    }
    #[inline(always)]
    fn s_floor(self) -> Self {
        Self::floor(self)
    }
    #[inline(always)]
    fn s_ceil(self) -> Self {
        Self::ceil(self)
    }
    #[inline(always)]
    fn s_sign(self) -> Self {
        if self > 0.0 {
            1.0
        } else if self == 0.0 {
            0.0
        } else {
            -1.0
        }
    }
    #[inline(always)]
    fn s_sin(self) -> Self {
        Self::sin(self.to_radians())
    }
    #[inline(always)]
    fn s_cos(self) -> Self {
        Self::cos(self.to_radians())
    }
    #[inline(always)]
    fn s_tan(self) -> Self {
        Self::tan(self.to_radians())
    }
    #[inline(always)]
    fn s_asin(self) -> Self {
        Self::asin(self).to_degrees()
    }
    #[inline(always)]
    fn s_acos(self) -> Self {
        Self::acos(self).to_degrees()
    }
    #[inline(always)]
    fn s_atan(self) -> Self {
        Self::atan(self).to_degrees()
    }
    #[inline(always)]
    fn s_sqrt(self) -> Self {
        Self::sqrt(self)
    }
}

impl FloatSample for f64 {
    #[inline(always)]
    fn s_ln(self) -> Self {
        if self == 0.0 { 0.0 } else { Self::ln(self) }
    }
    #[inline(always)]
    fn s_log10(self) -> Self {
        if self == 0.0 { 0.0 } else { Self::log10(self) }
    }
    #[inline(always)]
    fn s_exp(self) -> Self {
        Self::exp(self)
    }
    #[inline(always)]
    fn s_exp10(self) -> Self {
        Self::powf(10.0_f64, self)
    }
    #[inline(always)]
    fn s_pow(self, exp: Self) -> Self {
        if self == 0.0 {
            0.0
        } else if exp == -1.0 {
            1.0 / self
        } else if exp == 0.5 {
            Self::sqrt(self)
        } else {
            Self::powf(self, exp)
        }
    }
    #[inline(always)]
    fn s_abs(self) -> Self {
        Self::abs(self)
    }
    #[inline(always)]
    fn s_round(self) -> Self {
        Self::round_ties_even(self)
    }
    #[inline(always)]
    fn s_floor(self) -> Self {
        Self::floor(self)
    }
    #[inline(always)]
    fn s_ceil(self) -> Self {
        Self::ceil(self)
    }
    #[inline(always)]
    fn s_sign(self) -> Self {
        if self > 0.0 {
            1.0
        } else if self == 0.0 {
            0.0
        } else {
            -1.0
        }
    }
    #[inline(always)]
    fn s_sin(self) -> Self {
        Self::sin(self.to_radians())
    }
    #[inline(always)]
    fn s_cos(self) -> Self {
        Self::cos(self.to_radians())
    }
    #[inline(always)]
    fn s_tan(self) -> Self {
        Self::tan(self.to_radians())
    }
    #[inline(always)]
    fn s_asin(self) -> Self {
        Self::asin(self).to_degrees()
    }
    #[inline(always)]
    fn s_acos(self) -> Self {
        Self::acos(self).to_degrees()
    }
    #[inline(always)]
    fn s_atan(self) -> Self {
        Self::atan(self).to_degrees()
    }
    #[inline(always)]
    fn s_sqrt(self) -> Self {
        Self::sqrt(self)
    }
}

/// Methods available on integer-typed pixel samples.
///
/// Implemented for `u8`, `u16`, `i16`, `u32`, `i32`. Used as the bound for
/// `IntegerFormat`.
///
/// All methods carry `#[inline(always)]` for the same reason as `FloatSample`
/// guarantees LLVM sees the body at the call site and can auto-vectorise
/// loops over integer pixel slices.
pub trait IntSample: Copy + Eq + Ord + 'static {
    /// Associated constant for s max.
    const S_MAX: Self;
    /// Associated constant for s min.
    const S_MIN: Self;
    /// Associated constant for s zero.
    const S_ZERO: Self;
    /// Integer division that saturates to `S_MAX` on divide-by-zero (libvips
    /// behavior). For signed types, also saturates on MIN / -1 overflow.
    #[must_use]
    fn s_saturating_div(self, rhs: Self) -> Self;
    /// Absolute value. For unsigned types this is the identity. For signed types
    /// it is `saturating_abs` (so `i16::MIN.s_abs() == i16::MAX`).
    #[must_use]
    fn s_abs(self) -> Self;
    /// Sign: returns -1 for negative, 0 for zero, 1 for positive.
    /// For unsigned types: always 0 or 1 (no negatives).
    #[must_use]
    fn s_sign(self) -> Self;
}

macro_rules! impl_int_sample_unsigned {
    ($t:ty) => {
        impl IntSample for $t {
            const S_MAX: $t = <$t>::MAX;
            const S_MIN: $t = <$t>::MIN;
            const S_ZERO: $t = 0;
            #[inline(always)]
            fn s_saturating_div(self, rhs: $t) -> $t {
                if rhs == 0 { <$t>::MAX } else { self / rhs }
            }
            #[inline(always)]
            fn s_abs(self) -> $t {
                self
            }
            #[inline(always)]
            fn s_sign(self) -> $t {
                if self == 0 { 0 } else { 1 }
            }
        }
    };
}

macro_rules! impl_int_sample_signed {
    ($t:ty) => {
        impl IntSample for $t {
            const S_MAX: $t = <$t>::MAX;
            const S_MIN: $t = <$t>::MIN;
            const S_ZERO: $t = 0;
            #[inline(always)]
            fn s_saturating_div(self, rhs: $t) -> $t {
                // checked_div handles both div-by-zero and MIN / -1 overflow.
                self.checked_div(rhs).unwrap_or(<$t>::MAX)
            }
            #[inline(always)]
            fn s_abs(self) -> $t {
                self.saturating_abs()
            }
            #[inline(always)]
            fn s_sign(self) -> $t {
                if self < 0 {
                    -1
                } else if self == 0 {
                    0
                } else {
                    1
                }
            }
        }
    };
}

impl_int_sample_unsigned!(u8);
impl_int_sample_unsigned!(u16);
impl_int_sample_unsigned!(u32);
impl_int_sample_signed!(i16);
impl_int_sample_signed!(i32);

/// Division semantics for all sample types.
///
/// Unified `Divide<F>` op uses this bound regardless of float/integer format:
/// - Real types: divide-by-zero yields zero, matching libvips.
/// - Signed integer overflow on `MIN / -1` saturates to the type maximum.
///
pub trait DivSample: Copy + 'static {
    #[must_use]
    /// Returns or performs s div.
    fn s_div(self, rhs: Self) -> Self;
}

impl DivSample for u8 {
    #[inline(always)]
    fn s_div(self, rhs: Self) -> Self {
        if rhs == 0 { 0 } else { self / rhs }
    }
}
impl DivSample for u16 {
    #[inline(always)]
    fn s_div(self, rhs: Self) -> Self {
        if rhs == 0 { 0 } else { self / rhs }
    }
}
impl DivSample for u32 {
    #[inline(always)]
    fn s_div(self, rhs: Self) -> Self {
        if rhs == 0 { 0 } else { self / rhs }
    }
}
impl DivSample for i16 {
    #[inline(always)]
    fn s_div(self, rhs: Self) -> Self {
        if rhs == 0 {
            0
        } else {
            self.checked_div(rhs).unwrap_or(Self::MAX)
        }
    }
}
impl DivSample for i32 {
    #[inline(always)]
    fn s_div(self, rhs: Self) -> Self {
        if rhs == 0 {
            0
        } else {
            self.checked_div(rhs).unwrap_or(Self::MAX)
        }
    }
}
impl DivSample for f32 {
    #[inline(always)]
    fn s_div(self, rhs: Self) -> Self {
        if rhs == 0.0 { 0.0 } else { self / rhs }
    }
}
impl DivSample for f64 {
    #[inline(always)]
    fn s_div(self, rhs: Self) -> Self {
        if rhs == 0.0 { 0.0 } else { self / rhs }
    }
}

/// Addition semantics for all sample types.
///
/// Integer types saturate to the target format range, matching libvips'
/// clip-on-write behavior for the currently implemented same-format ops.
pub trait AddSample: Copy + 'static {
    #[must_use]
    /// Returns or performs s add.
    fn s_add(self, rhs: Self) -> Self;
}

macro_rules! impl_add_sample_int {
    ($($t:ty),+ $(,)?) => {
        $(
            impl AddSample for $t {
                #[inline(always)]
                fn s_add(self, rhs: $t) -> $t {
                    self.saturating_add(rhs)
                }
            }
        )+
    };
}

impl_add_sample_int!(u8, u16, u32, i16, i32);

impl AddSample for f32 {
    #[inline(always)]
    fn s_add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl AddSample for f64 {
    #[inline(always)]
    fn s_add(self, rhs: Self) -> Self {
        self + rhs
    }
}

/// Subtraction semantics for all sample types.
pub trait SubSample: Copy + 'static {
    #[must_use]
    /// Returns or performs s sub.
    fn s_sub(self, rhs: Self) -> Self;
}

macro_rules! impl_sub_sample_int {
    ($($t:ty),+ $(,)?) => {
        $(
            impl SubSample for $t {
                #[inline(always)]
                fn s_sub(self, rhs: $t) -> $t {
                    self.saturating_sub(rhs)
                }
            }
        )+
    };
}

impl_sub_sample_int!(u8, u16, u32, i16, i32);

impl SubSample for f32 {
    #[inline(always)]
    fn s_sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

impl SubSample for f64 {
    #[inline(always)]
    fn s_sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

/// Multiplication semantics for all sample types.
pub trait MulSample: Copy + 'static {
    #[must_use]
    /// Returns or performs s mul.
    fn s_mul(self, rhs: Self) -> Self;
}

macro_rules! impl_mul_sample_int {
    ($($t:ty),+ $(,)?) => {
        $(
            impl MulSample for $t {
                #[inline(always)]
                fn s_mul(self, rhs: $t) -> $t {
                    self.saturating_mul(rhs)
                }
            }
        )+
    };
}

impl_mul_sample_int!(u8, u16, u32, i16, i32);

impl MulSample for f32 {
    #[inline(always)]
    fn s_mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl MulSample for f64 {
    #[inline(always)]
    fn s_mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

/// Absolute value and sign for all sample types.
///
/// Implemented for every concrete band format (`u8`, `u16`, `u32`, `i16`, `i32`,
/// `f32`, `f64`). Used as the bound for `Abs` and `Sign` operations, which apply to
/// all formats (float and integer alike).
pub trait AbsSample: Copy + 'static {
    #[must_use]
    /// Returns or performs s abs.
    fn s_abs(self) -> Self;
    /// Sign of the sample value.
    ///
    /// - Float: `-1.0`, `0.0`, or `1.0`.
    /// - Unsigned integer: `0` if zero, `1` if non-zero.
    /// - Signed integer: `-1`, `0`, or `1` (saturating on `MIN`).
    #[must_use]
    fn s_sign(self) -> Self;
}
impl AbsSample for u8 {
    #[inline(always)]
    fn s_abs(self) -> Self {
        self
    }
    #[inline(always)]
    fn s_sign(self) -> Self {
        Self::from(self != 0)
    }
}
impl AbsSample for u16 {
    #[inline(always)]
    fn s_abs(self) -> Self {
        self
    }
    #[inline(always)]
    fn s_sign(self) -> Self {
        Self::from(self != 0)
    }
}
impl AbsSample for u32 {
    #[inline(always)]
    fn s_abs(self) -> Self {
        self
    }
    #[inline(always)]
    fn s_sign(self) -> Self {
        Self::from(self != 0)
    }
}
impl AbsSample for i16 {
    #[inline(always)]
    fn s_abs(self) -> Self {
        self.saturating_abs()
    }
    #[inline(always)]
    fn s_sign(self) -> Self {
        if self < 0 { -1 } else { Self::from(self != 0) }
    }
}
impl AbsSample for i32 {
    #[inline(always)]
    fn s_abs(self) -> Self {
        self.saturating_abs()
    }
    #[inline(always)]
    fn s_sign(self) -> Self {
        if self < 0 { -1 } else { Self::from(self != 0) }
    }
}
impl AbsSample for f32 {
    #[inline(always)]
    fn s_abs(self) -> Self {
        Self::abs(self)
    }
    #[inline(always)]
    fn s_sign(self) -> Self {
        if self > 0.0 {
            1.0
        } else if self == 0.0 {
            0.0
        } else {
            -1.0
        }
    }
}
impl AbsSample for f64 {
    #[inline(always)]
    fn s_abs(self) -> Self {
        Self::abs(self)
    }
    #[inline(always)]
    fn s_sign(self) -> Self {
        if self > 0.0 {
            1.0
        } else if self == 0.0 {
            0.0
        } else {
            -1.0
        }
    }
}

/// Remainder semantics for all sample types.
///
/// - Integer types: `%`, except divide-by-zero yields `-1` (or `MAX` for unsigned),
///   matching libvips.
/// - Float types: `a - b * floor(a / b)`, with divide-by-zero yielding `-1.0`.
pub trait PairMinMaxSample: Copy + 'static {
    #[must_use]
    /// Returns or performs s maxpair.
    fn s_maxpair(self, rhs: Self) -> Self;
    #[must_use]
    /// Returns or performs s minpair.
    fn s_minpair(self, rhs: Self) -> Self;
}

macro_rules! impl_pair_minmax_sample_int {
    ($($t:ty),+ $(,)?) => {
        $(
            impl PairMinMaxSample for $t {
                #[inline(always)]
                fn s_maxpair(self, rhs: $t) -> $t {
                    self.max(rhs)
                }

                #[inline(always)]
                fn s_minpair(self, rhs: $t) -> $t {
                    self.min(rhs)
                }
            }
        )+
    };
}

impl_pair_minmax_sample_int!(u8, u16, i16, u32, i32);

impl PairMinMaxSample for f32 {
    #[inline(always)]
    fn s_maxpair(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline(always)]
    fn s_minpair(self, rhs: Self) -> Self {
        self.min(rhs)
    }
}

impl PairMinMaxSample for f64 {
    #[inline(always)]
    fn s_maxpair(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline(always)]
    fn s_minpair(self, rhs: Self) -> Self {
        self.min(rhs)
    }
}

/// Pairwise math helpers for sample types that support binary transcendental operations.
///
/// Point operations use this to express `pow`, swapped `pow`, and angular `atan2` logic in a
/// format-generic way.
///
/// # Examples
/// ```rust
/// # use viprs::domain::format::Math2Sample;
/// assert_eq!(2.0_f32.s_pow2(3.0), 8.0);
/// ```
pub trait Math2Sample: Copy + 'static {
    #[must_use]
    /// Returns or performs s pow2.
    fn s_pow2(self, rhs: Self) -> Self;
    #[must_use]
    /// Returns or performs s wop.
    fn s_wop(self, rhs: Self) -> Self;
    #[must_use]
    /// Returns or performs s atan2.
    fn s_atan2(self, rhs: Self) -> Self;
}

impl Math2Sample for f32 {
    #[inline(always)]
    fn s_pow2(self, rhs: Self) -> Self {
        if self == 0.0 {
            0.0
        } else if rhs == -1.0 {
            1.0 / self
        } else if rhs == 0.5 {
            self.sqrt()
        } else {
            self.powf(rhs)
        }
    }

    #[inline(always)]
    fn s_wop(self, rhs: Self) -> Self {
        rhs.s_pow2(self)
    }

    #[inline(always)]
    fn s_atan2(self, rhs: Self) -> Self {
        self.atan2(rhs).to_degrees().rem_euclid(360.0)
    }
}

impl Math2Sample for f64 {
    #[inline(always)]
    fn s_pow2(self, rhs: Self) -> Self {
        if self == 0.0 {
            0.0
        } else if rhs == -1.0 {
            1.0 / self
        } else if rhs == 0.5 {
            self.sqrt()
        } else {
            self.powf(rhs)
        }
    }

    #[inline(always)]
    fn s_wop(self, rhs: Self) -> Self {
        rhs.s_pow2(self)
    }

    #[inline(always)]
    fn s_atan2(self, rhs: Self) -> Self {
        self.atan2(rhs).to_degrees().rem_euclid(360.0)
    }
}

/// Remainder semantics for sample types following libvips-compatible edge cases.
///
/// This centralizes divide-by-zero behaviour so remainder-based point ops stay consistent across
/// integer and float formats.
///
/// # Examples
/// ```rust
/// # use viprs::domain::format::RemSample;
/// assert_eq!(5_u8.s_remainder(2), 1);
/// ```
pub trait RemSample: Copy + 'static {
    #[must_use]
    /// Returns or performs s remainder.
    fn s_remainder(self, rhs: Self) -> Self;
}

impl RemSample for u8 {
    #[inline(always)]
    fn s_remainder(self, rhs: Self) -> Self {
        if rhs == 0 { Self::MAX } else { self % rhs }
    }
}

impl RemSample for u16 {
    #[inline(always)]
    fn s_remainder(self, rhs: Self) -> Self {
        if rhs == 0 { Self::MAX } else { self % rhs }
    }
}

impl RemSample for u32 {
    #[inline(always)]
    fn s_remainder(self, rhs: Self) -> Self {
        if rhs == 0 { Self::MAX } else { self % rhs }
    }
}

impl RemSample for i16 {
    #[inline(always)]
    fn s_remainder(self, rhs: Self) -> Self {
        if rhs == 0 {
            -1
        } else {
            self.checked_rem(rhs).unwrap_or_default()
        }
    }
}

impl RemSample for i32 {
    #[inline(always)]
    fn s_remainder(self, rhs: Self) -> Self {
        if rhs == 0 {
            -1
        } else {
            self.checked_rem(rhs).unwrap_or_default()
        }
    }
}

impl RemSample for f32 {
    #[inline(always)]
    fn s_remainder(self, rhs: Self) -> Self {
        if rhs == 0.0 {
            -1.0
        } else {
            rhs.mul_add(-Self::floor(self / rhs), self)
        }
    }
}

impl RemSample for f64 {
    #[inline(always)]
    fn s_remainder(self, rhs: Self) -> Self {
        if rhs == 0.0 {
            -1.0
        } else {
            rhs.mul_add(-Self::floor(self / rhs), self)
        }
    }
}

/// Bitwise operations on unsigned integer samples.
///
/// Only `u8`, `u16`, `u32` implement this trait. Signed types are excluded
/// because bitwise operations on sign-extended integers have platform-dependent
/// semantics (arithmetic vs logical shift right). Float types are excluded
/// because IEEE 754 bit patterns are not meaningful pixel values.
pub trait BitwiseSample:
    IntSample
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitXor<Output = Self>
    + std::ops::Not<Output = Self>
    + std::ops::Shl<u32, Output = Self>
    + std::ops::Shr<u32, Output = Self>
{
}

impl BitwiseSample for u8 {}
impl BitwiseSample for u16 {}
impl BitwiseSample for u32 {}
