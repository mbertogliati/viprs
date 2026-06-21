use super::sample_math::{BitwiseSample, FloatSample, IntSample};

// Sealed-trait pattern: only types in this crate can implement BandFormat.
mod private {
    pub trait Sealed {}
}

/// Identifies a band format at runtime, without generics.
///
/// This is used when decoders, schedulers, or dynamic pipelines need to reason about sample
/// layout without monomorphized type parameters.
///
/// # Examples
/// ```rust
/// # use viprs_core::format::BandFormatId;
/// assert_eq!(BandFormatId::U8, BandFormatId::U8);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BandFormatId {
    /// Uses the `U8` variant of `BandFormatId`.
    U8,
    /// Uses the `U16` variant of `BandFormatId`.
    U16,
    /// Uses the `I16` variant of `BandFormatId`.
    I16,
    /// Uses the `U32` variant of `BandFormatId`.
    U32,
    /// Uses the `I32` variant of `BandFormatId`.
    I32,
    /// Uses the `F32` variant of `BandFormatId`.
    F32,
    /// Uses the `F64` variant of `BandFormatId`.
    F64,
}

/// Marker trait for pixel band formats.
///
/// Sealed so that only crate-defined types can be used as format parameters,
/// enabling exhaustive optimisation without breaking downstream code.
///
/// # Examples
/// ```rust
/// # use viprs_core::format::{BandFormat, U8};
/// assert_eq!(U8::ID, <U8 as BandFormat>::ID);
/// ```
pub trait BandFormat: private::Sealed + Send + Sync + 'static {
    /// The scalar type that represents one channel value in memory.
    type Sample: Copy + Send + Sync + 'static + bytemuck::Pod;
    /// Runtime identifier for this format.
    const ID: BandFormatId;
}

/// Refinement of [`BandFormat`] for formats on which arithmetic is well-defined.
///
/// Arithmetic ops use this to exclude non-numeric formats while still sharing the same
/// runtime format identifiers.
///
/// # Examples
/// ```rust
/// # use viprs_core::format::{NumericBand, U8};
/// fn accepts_numeric<F: NumericBand>() {}
/// accepts_numeric::<U8>();
/// ```
pub trait NumericBand: BandFormat {}

// ── Format sub-traits ────────────────────────────────────────────────────────
//
// Effectively sealed: `FloatFormat: BandFormat` and `BandFormat` is sealed, so
// no external crate can implement `FloatFormat`.

/// Refinement of `BandFormat` for IEEE 754 float formats (`F32`, `F64`).
///
/// Bounds `log`, `exp`, `pow`, `round`, `floor`, `ceil`, `sign` operations.
pub trait FloatFormat: BandFormat
where
    Self::Sample: FloatSample,
{
}

/// Refinement of `BandFormat` for integer formats (`U8`, `U16`, `I16`, `U32`, `I32`).
///
/// Bounds `saturating_div`, integer `abs`, and integer `sign` operations.
pub trait IntegerFormat: BandFormat
where
    Self::Sample: IntSample,
{
}

/// Refinement of `IntegerFormat` for unsigned integer formats (`U8`, `U16`, `U32`).
///
/// Bounds `and`, `or`, `xor`, `lshift`, `rshift` operations.
pub trait BitwiseFormat: IntegerFormat
where
    Self::Sample: BitwiseSample,
{
}

// ── Concrete zero-sized format types ─────────────────────────────────────────

/// 8-bit unsigned integer band.
pub struct U8;
/// 16-bit unsigned integer band.
pub struct U16;
/// 16-bit signed integer band.
pub struct I16;
/// 32-bit unsigned integer band.
pub struct U32;
/// 32-bit signed integer band.
pub struct I32;
/// 32-bit IEEE 754 floating-point band.
pub struct F32;
/// 64-bit IEEE 754 floating-point band.
pub struct F64;

// ── Sealed impls ─────────────────────────────────────────────────────────────

impl private::Sealed for U8 {}
impl private::Sealed for U16 {}
impl private::Sealed for I16 {}
impl private::Sealed for U32 {}
impl private::Sealed for I32 {}
impl private::Sealed for F32 {}
impl private::Sealed for F64 {}

// ── BandFormat impls ─────────────────────────────────────────────────────────

impl BandFormat for U8 {
    type Sample = u8;
    const ID: BandFormatId = BandFormatId::U8;
}
impl BandFormat for U16 {
    type Sample = u16;
    const ID: BandFormatId = BandFormatId::U16;
}
impl BandFormat for I16 {
    type Sample = i16;
    const ID: BandFormatId = BandFormatId::I16;
}
impl BandFormat for U32 {
    type Sample = u32;
    const ID: BandFormatId = BandFormatId::U32;
}
impl BandFormat for I32 {
    type Sample = i32;
    const ID: BandFormatId = BandFormatId::I32;
}
impl BandFormat for F32 {
    type Sample = f32;
    const ID: BandFormatId = BandFormatId::F32;
}
impl BandFormat for F64 {
    type Sample = f64;
    const ID: BandFormatId = BandFormatId::F64;
}

// ── NumericBand impls ─────────────────────────────────────────────────────────

impl NumericBand for U8 {}
impl NumericBand for U16 {}
impl NumericBand for I16 {}
impl NumericBand for U32 {}
impl NumericBand for I32 {}
impl NumericBand for F32 {}
impl NumericBand for F64 {}

// ── FloatFormat impls ─────────────────────────────────────────────────────────

impl FloatFormat for F32 {}
impl FloatFormat for F64 {}

// ── IntegerFormat impls ───────────────────────────────────────────────────────

impl IntegerFormat for U8 {}
impl IntegerFormat for U16 {}
impl IntegerFormat for I16 {}
impl IntegerFormat for U32 {}
impl IntegerFormat for I32 {}

// ── BitwiseFormat impls ───────────────────────────────────────────────────────

impl BitwiseFormat for U8 {}
impl BitwiseFormat for U16 {}
impl BitwiseFormat for U32 {}
