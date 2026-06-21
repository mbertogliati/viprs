//! Sample conversion traits shared across resampling and convolution operations.

/// Lossless or widening conversion of `F::Sample` to `f64` for kernel accumulation.
pub trait ToF64: Copy {
    /// Converts this value to f64.
    fn to_f64(self) -> f64;
}

impl ToF64 for u8 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for u16 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for i16 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for u32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for i32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
impl ToF64 for f64 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
}

/// Clamped narrowing conversion from `f64` accumulator back to `F::Sample`.
pub trait FromF64: Copy {
    /// Converts an accumulated floating-point value back into the destination sample type.
    fn from_f64(v: f64) -> Self;
}

impl FromF64 for u8 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v.clamp(f64::from(Self::MIN), f64::from(Self::MAX)).round() as Self
    }
}
impl FromF64 for u16 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v.clamp(f64::from(Self::MIN), f64::from(Self::MAX)).round() as Self
    }
}
impl FromF64 for i16 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v.clamp(f64::from(Self::MIN), f64::from(Self::MAX)).round() as Self
    }
}
impl FromF64 for u32 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v.clamp(f64::from(Self::MIN), f64::from(Self::MAX)).round() as Self
    }
}
impl FromF64 for i32 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v.clamp(f64::from(Self::MIN), f64::from(Self::MAX)).round() as Self
    }
}
impl FromF64 for f32 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as Self
    }
}
impl FromF64 for f64 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
}
