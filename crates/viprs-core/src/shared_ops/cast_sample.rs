//! Shared sample-casting trait used by conversion and colour operations.

/// Type-level conversion between sample types.
///
/// Earlier revisions implemented `Cast` as a special dynamic operation because
/// the operation trait assumed `Input == Output`. With `Op`, `Cast` behaves like
/// any other operation and can use the same bridging infrastructure.
pub trait CastSample<To: Copy>: Copy {
    /// Returns or performs cast to.
    fn cast_to(self) -> To;
}

// u8 → f32: normalize to [0.0, 1.0]
impl CastSample<f32> for u8 {
    fn cast_to(self) -> f32 {
        f32::from(self) / 255.0
    }
}
// f32 → u8: clamp to [0,1] then scale to [0,255]
impl CastSample<u8> for f32 {
    fn cast_to(self) -> u8 {
        (self.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}
// f32 → u16: clamp to [0,1] then scale to [0,65535]
impl CastSample<u16> for f32 {
    fn cast_to(self) -> u16 {
        (self.clamp(0.0, 1.0) * 65535.0).round() as u16
    }
}
// u8 → u16: scale 0-255 to 0-65535
impl CastSample<u16> for u8 {
    fn cast_to(self) -> u16 {
        u16::from(self) * 257
    }
}
// u16 → f32: normalize to [0.0, 1.0]
impl CastSample<f32> for u16 {
    fn cast_to(self) -> f32 {
        f32::from(self) / 65535.0
    }
}
// f32 → f64
impl CastSample<f64> for f32 {
    fn cast_to(self) -> f64 {
        f64::from(self)
    }
}
// f64 → f32
impl CastSample<f32> for f64 {
    fn cast_to(self) -> f32 {
        self as f32
    }
}
// identity casts
impl CastSample<Self> for u8 {
    fn cast_to(self) -> Self {
        self
    }
}
impl CastSample<Self> for f32 {
    fn cast_to(self) -> Self {
        self
    }
}
