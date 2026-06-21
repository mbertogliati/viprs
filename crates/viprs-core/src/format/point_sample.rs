use crate::coeff::OptimizedCoeff;

// ── PointSample: primitives for Concretize ops ──────────────────────────────
//
// This trait provides the sample-level operations needed by format-erased
// `Concretize` ops. Unlike `Invertible`/`FloatSample`/etc., this is a unified
// trait implemented for ALL sample types (u8, u16, i16, u32, i32, f32, f64).
//
// Each method has a well-defined semantic per type matching libvips behavior.
// The `#[inline(always)]` is mandatory for LLVM auto-vectorization.

/// Sample-level arithmetic for `Concretize` point-ops.
///
/// Implemented for all 7 pixel sample types. These are the building blocks
/// that format-erased operations call inside `apply_sample::<F>()`.
pub trait PointSample: Copy + Send + Sync + 'static + bytemuck::Pod {
    /// Additive inversion: `max - x` for unsigned, `-x` for signed, `1.0 - x` for float.
    #[must_use]
    fn pt_invert(self) -> Self;
    /// Linear transform: `x * scale + offset`, clamped to valid range.
    ///
    /// Takes [`OptimizedCoeff`] instead of raw `f64` so each type can pick the narrowest
    /// arithmetic that is exact — e.g. u8 uses i16 when coefficients fit,
    /// getting 8 NEON lanes instead of 4.
    #[must_use]
    fn pt_linear(self, scale: OptimizedCoeff, offset: OptimizedCoeff) -> Self;
    /// Absolute value.
    #[must_use]
    fn pt_abs(self) -> Self;
    /// Clamp to [min, max] (expressed as f64 for format-erased API).
    #[must_use]
    fn pt_clamp(self, min: f64, max: f64) -> Self;
    /// Convert to f64 for arithmetic.
    fn pt_to_f64(self) -> f64;
    /// Convert from f64, clamping to valid range.
    fn pt_from_f64(v: f64) -> Self;

    // ── Float math (integer formats go through f64 round-trip) ───────────

    /// Sine (radians).
    #[must_use]
    fn pt_sin(self) -> Self {
        Self::pt_from_f64(self.pt_to_f64().sin())
    }
    /// Cosine (radians).
    #[must_use]
    fn pt_cos(self) -> Self {
        Self::pt_from_f64(self.pt_to_f64().cos())
    }
    /// Tangent (radians).
    #[must_use]
    fn pt_tan(self) -> Self {
        Self::pt_from_f64(self.pt_to_f64().tan())
    }
    /// Arcsine.
    #[must_use]
    fn pt_asin(self) -> Self {
        Self::pt_from_f64(self.pt_to_f64().asin())
    }
    /// Arccosine.
    #[must_use]
    fn pt_acos(self) -> Self {
        Self::pt_from_f64(self.pt_to_f64().acos())
    }
    /// Arctangent.
    #[must_use]
    fn pt_atan(self) -> Self {
        Self::pt_from_f64(self.pt_to_f64().atan())
    }
    /// Natural exponential.
    #[must_use]
    fn pt_exp(self) -> Self {
        Self::pt_from_f64(self.pt_to_f64().exp())
    }
    /// Natural logarithm (0 → 0 to match libvips).
    #[must_use]
    fn pt_log(self) -> Self {
        let v = self.pt_to_f64();
        Self::pt_from_f64(if v == 0.0 { 0.0 } else { v.ln() })
    }
    /// Square root.
    #[must_use]
    fn pt_sqrt(self) -> Self {
        Self::pt_from_f64(self.pt_to_f64().sqrt())
    }
    /// Power.
    #[must_use]
    fn pt_pow(self, exp: f64) -> Self {
        let v = self.pt_to_f64();
        Self::pt_from_f64(if v == 0.0 { 0.0 } else { v.powf(exp) })
    }
    /// Round to nearest integer.
    #[must_use]
    fn pt_round(self) -> Self {
        Self::pt_from_f64(self.pt_to_f64().round())
    }
    /// Floor.
    #[must_use]
    fn pt_floor(self) -> Self {
        Self::pt_from_f64(self.pt_to_f64().floor())
    }
    /// Ceil.
    #[must_use]
    fn pt_ceil(self) -> Self {
        Self::pt_from_f64(self.pt_to_f64().ceil())
    }
    /// Sign: -1, 0, or 1 (as f64, then converted).
    #[must_use]
    fn pt_sign(self) -> Self {
        let v = self.pt_to_f64();
        Self::pt_from_f64(if v < 0.0 {
            -1.0
        } else if v == 0.0 {
            0.0
        } else {
            1.0
        })
    }
}

impl PointSample for u8 {
    #[inline(always)]
    fn pt_invert(self) -> Self {
        255u8.wrapping_sub(self)
    }
    #[inline(always)]
    fn pt_linear(self, scale: OptimizedCoeff, offset: OptimizedCoeff) -> Self {
        // If coefficients are exact integers fitting i16, use i16 math (8 NEON lanes).
        // LLVM hoists this branch out of the loop — generates two vectorized loops.
        if let Some((si, oi)) = OptimizedCoeff::i16_mul_add_unsigned(scale, offset, 255) {
            (i16::from(self) * si + oi).clamp(0, 255) as Self
        } else {
            f32::from(self)
                .mul_add(scale.as_f32(), offset.as_f32())
                .clamp(0.0, 255.0) as Self
        }
    }
    #[inline(always)]
    fn pt_abs(self) -> Self {
        self
    }
    #[inline(always)]
    fn pt_clamp(self, min: f64, max: f64) -> Self {
        // Native u8 clamp — avoids float conversion entirely (16 NEON lanes).
        let lo = min.clamp(0.0, 255.0) as Self;
        let hi = max.clamp(0.0, 255.0) as Self;
        self.clamp(lo, hi)
    }
    #[inline(always)]
    fn pt_to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn pt_from_f64(v: f64) -> Self {
        v.clamp(0.0, 255.0) as Self
    }
}

impl PointSample for u16 {
    #[inline(always)]
    fn pt_invert(self) -> Self {
        65535u16.wrapping_sub(self)
    }
    #[inline(always)]
    fn pt_linear(self, scale: OptimizedCoeff, offset: OptimizedCoeff) -> Self {
        // f32 math: 23-bit mantissa covers u16 range (16 bits) with precision to spare.
        f32::from(self)
            .mul_add(scale.as_f32(), offset.as_f32())
            .clamp(0.0, 65535.0) as Self
    }
    #[inline(always)]
    fn pt_abs(self) -> Self {
        self
    }
    #[inline(always)]
    fn pt_clamp(self, min: f64, max: f64) -> Self {
        f32::from(self).clamp(min as f32, max as f32) as Self
    }
    #[inline(always)]
    fn pt_to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn pt_from_f64(v: f64) -> Self {
        v.clamp(0.0, 65535.0) as Self
    }
}

impl PointSample for i16 {
    #[inline(always)]
    fn pt_invert(self) -> Self {
        self.saturating_neg()
    }
    #[inline(always)]
    fn pt_linear(self, scale: OptimizedCoeff, offset: OptimizedCoeff) -> Self {
        f32::from(self)
            .mul_add(scale.as_f32(), offset.as_f32())
            .clamp(-32768.0, 32767.0) as Self
    }
    #[inline(always)]
    fn pt_abs(self) -> Self {
        self.saturating_abs()
    }
    #[inline(always)]
    fn pt_clamp(self, min: f64, max: f64) -> Self {
        f32::from(self).clamp(min as f32, max as f32) as Self
    }
    #[inline(always)]
    fn pt_to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn pt_from_f64(v: f64) -> Self {
        v.clamp(-32768.0, 32767.0) as Self
    }
}

impl PointSample for u32 {
    #[inline(always)]
    fn pt_invert(self) -> Self {
        Self::MAX.wrapping_sub(self)
    }
    #[inline(always)]
    fn pt_linear(self, scale: OptimizedCoeff, offset: OptimizedCoeff) -> Self {
        f64::from(self)
            .mul_add(scale.as_f64(), offset.as_f64())
            .clamp(0.0, f64::from(Self::MAX)) as Self
    }
    #[inline(always)]
    fn pt_abs(self) -> Self {
        self
    }
    #[inline(always)]
    fn pt_clamp(self, min: f64, max: f64) -> Self {
        f64::from(self).clamp(min, max) as Self
    }
    #[inline(always)]
    fn pt_to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn pt_from_f64(v: f64) -> Self {
        v.clamp(0.0, f64::from(Self::MAX)) as Self
    }
}

impl PointSample for i32 {
    #[inline(always)]
    fn pt_invert(self) -> Self {
        self.saturating_neg()
    }
    #[inline(always)]
    fn pt_linear(self, scale: OptimizedCoeff, offset: OptimizedCoeff) -> Self {
        f64::from(self)
            .mul_add(scale.as_f64(), offset.as_f64())
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
    #[inline(always)]
    fn pt_abs(self) -> Self {
        self.saturating_abs()
    }
    #[inline(always)]
    fn pt_clamp(self, min: f64, max: f64) -> Self {
        f64::from(self).clamp(min, max) as Self
    }
    #[inline(always)]
    fn pt_to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn pt_from_f64(v: f64) -> Self {
        v.clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl PointSample for f32 {
    #[inline(always)]
    fn pt_invert(self) -> Self {
        1.0 - self
    }
    #[inline(always)]
    fn pt_linear(self, scale: OptimizedCoeff, offset: OptimizedCoeff) -> Self {
        self.mul_add(scale.as_f32(), offset.as_f32())
    }
    #[inline(always)]
    fn pt_abs(self) -> Self {
        Self::abs(self)
    }
    #[inline(always)]
    fn pt_clamp(self, min: f64, max: f64) -> Self {
        f64::from(self).clamp(min, max) as Self
    }
    #[inline(always)]
    fn pt_to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn pt_from_f64(v: f64) -> Self {
        v as Self
    }
    #[inline(always)]
    fn pt_sin(self) -> Self {
        Self::sin(self)
    }
    #[inline(always)]
    fn pt_cos(self) -> Self {
        Self::cos(self)
    }
    #[inline(always)]
    fn pt_tan(self) -> Self {
        Self::tan(self)
    }
    #[inline(always)]
    fn pt_asin(self) -> Self {
        Self::asin(self)
    }
    #[inline(always)]
    fn pt_acos(self) -> Self {
        Self::acos(self)
    }
    #[inline(always)]
    fn pt_atan(self) -> Self {
        Self::atan(self)
    }
    #[inline(always)]
    fn pt_exp(self) -> Self {
        Self::exp(self)
    }
    #[inline(always)]
    fn pt_log(self) -> Self {
        if self == 0.0 { 0.0 } else { Self::ln(self) }
    }
    #[inline(always)]
    fn pt_sqrt(self) -> Self {
        Self::sqrt(self)
    }
    #[inline(always)]
    fn pt_pow(self, exp: f64) -> Self {
        if self == 0.0 {
            0.0
        } else {
            Self::powf(self, exp as Self)
        }
    }
    #[inline(always)]
    fn pt_round(self) -> Self {
        Self::round(self)
    }
    #[inline(always)]
    fn pt_floor(self) -> Self {
        Self::floor(self)
    }
    #[inline(always)]
    fn pt_ceil(self) -> Self {
        Self::ceil(self)
    }
    #[inline(always)]
    fn pt_sign(self) -> Self {
        if self < 0.0 {
            -1.0
        } else if self == 0.0 {
            0.0
        } else {
            1.0
        }
    }
}

impl PointSample for f64 {
    #[inline(always)]
    fn pt_invert(self) -> Self {
        1.0 - self
    }
    #[inline(always)]
    fn pt_linear(self, scale: OptimizedCoeff, offset: OptimizedCoeff) -> Self {
        self.mul_add(scale.as_f64(), offset.as_f64())
    }
    #[inline(always)]
    fn pt_abs(self) -> Self {
        Self::abs(self)
    }
    #[inline(always)]
    fn pt_clamp(self, min: f64, max: f64) -> Self {
        self.clamp(min, max)
    }
    #[inline(always)]
    fn pt_to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn pt_from_f64(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn pt_sin(self) -> Self {
        Self::sin(self)
    }
    #[inline(always)]
    fn pt_cos(self) -> Self {
        Self::cos(self)
    }
    #[inline(always)]
    fn pt_tan(self) -> Self {
        Self::tan(self)
    }
    #[inline(always)]
    fn pt_asin(self) -> Self {
        Self::asin(self)
    }
    #[inline(always)]
    fn pt_acos(self) -> Self {
        Self::acos(self)
    }
    #[inline(always)]
    fn pt_atan(self) -> Self {
        Self::atan(self)
    }
    #[inline(always)]
    fn pt_exp(self) -> Self {
        Self::exp(self)
    }
    #[inline(always)]
    fn pt_log(self) -> Self {
        if self == 0.0 { 0.0 } else { Self::ln(self) }
    }
    #[inline(always)]
    fn pt_sqrt(self) -> Self {
        Self::sqrt(self)
    }
    #[inline(always)]
    fn pt_pow(self, exp: f64) -> Self {
        if self == 0.0 {
            0.0
        } else {
            Self::powf(self, exp)
        }
    }
    #[inline(always)]
    fn pt_round(self) -> Self {
        Self::round(self)
    }
    #[inline(always)]
    fn pt_floor(self) -> Self {
        Self::floor(self)
    }
    #[inline(always)]
    fn pt_ceil(self) -> Self {
        Self::ceil(self)
    }
    #[inline(always)]
    fn pt_sign(self) -> Self {
        if self < 0.0 {
            -1.0
        } else if self == 0.0 {
            0.0
        } else {
            1.0
        }
    }
}
