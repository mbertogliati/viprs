//! Format-erased boolean/bitwise operations.

use crate::domain::concretize::{Concretize, WideAccum, Width};
use crate::domain::format::{BandFormat, PointSample};

/// Bitwise AND with a constant value. For floats, operates via f64 round-trip
/// (not meaningful but consistent — libvips casts to int first).
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::point::boolean::BoolAnd;
///
/// let op = BoolAnd::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BoolAnd {
    /// Value associated with this item.
    pub value: u64,
}

impl BoolAnd {
    #[must_use]
    /// Creates a new `BoolAnd`.
    pub const fn new(value: u64) -> Self {
        Self { value }
    }
}

impl Concretize for BoolAnd {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        let v = x.pt_to_f64() as u64 & self.value;
        F::Sample::pt_from_f64(v as f64)
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        W::from_f32((x.to_f32() as u64 & self.value) as f32)
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::Native
    }
}

/// Bitwise OR with a constant value.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::point::boolean::BoolOr;
///
/// let op = BoolOr::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BoolOr {
    /// Value associated with this item.
    pub value: u64,
}

impl BoolOr {
    #[must_use]
    /// Creates a new `BoolOr`.
    pub const fn new(value: u64) -> Self {
        Self { value }
    }
}

impl Concretize for BoolOr {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        let v = x.pt_to_f64() as u64 | self.value;
        F::Sample::pt_from_f64(v as f64)
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        W::from_f32((x.to_f32() as u64 | self.value) as f32)
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::Native
    }
}

/// Bitwise XOR with a constant value.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::point::boolean::BoolXor;
///
/// let op = BoolXor::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BoolXor {
    /// Value associated with this item.
    pub value: u64,
}

impl BoolXor {
    #[must_use]
    /// Creates a new `BoolXor`.
    pub const fn new(value: u64) -> Self {
        Self { value }
    }
}

impl Concretize for BoolXor {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        let v = x.pt_to_f64() as u64 ^ self.value;
        F::Sample::pt_from_f64(v as f64)
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        W::from_f32((x.to_f32() as u64 ^ self.value) as f32)
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::Native
    }
}

/// Left shift by a constant number of bits.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::point::boolean::Lshift;
///
/// let op = Lshift::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Lshift {
    /// Stores the `bits` value for this item.
    pub bits: u32,
}

impl Lshift {
    #[must_use]
    /// Creates a new `Lshift`.
    pub const fn new(bits: u32) -> Self {
        Self { bits }
    }
}

impl Concretize for Lshift {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        let v = (x.pt_to_f64() as u64) << self.bits;
        F::Sample::pt_from_f64(v as f64)
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        W::from_f32(((x.to_f32() as u64) << self.bits) as f32)
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::I16
    }
}

/// Right shift by a constant number of bits.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::point::boolean::Rshift;
///
/// let op = Rshift::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Rshift {
    /// Stores the `bits` value for this item.
    pub bits: u32,
}

impl Rshift {
    #[must_use]
    /// Creates a new `Rshift`.
    pub const fn new(bits: u32) -> Self {
        Self { bits }
    }
}

impl Concretize for Rshift {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        let v = (x.pt_to_f64() as u64) >> self.bits;
        F::Sample::pt_from_f64(v as f64)
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        W::from_f32(((x.to_f32() as u64) >> self.bits) as f32)
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::Native
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::concretize::{Concretize, Width, apply_chain_to_slice};
    use crate::domain::format::U8;

    #[test]
    fn and_u8() {
        let mut pixels: Vec<u8> = vec![0xFF, 0xAB, 0x0F];
        apply_chain_to_slice::<U8, _>(&BoolAnd::new(0x0F), &mut pixels);
        assert_eq!(pixels, vec![0x0F, 0x0B, 0x0F]);
    }

    #[test]
    fn or_u8() {
        let mut pixels: Vec<u8> = vec![0x00, 0xA0, 0xFF];
        apply_chain_to_slice::<U8, _>(&BoolOr::new(0x0F), &mut pixels);
        assert_eq!(pixels, vec![0x0F, 0xAF, 0xFF]);
    }

    #[test]
    fn xor_u8() {
        let mut pixels: Vec<u8> = vec![0xFF, 0x00, 0xAA];
        apply_chain_to_slice::<U8, _>(&BoolXor::new(0xFF), &mut pixels);
        assert_eq!(pixels, vec![0x00, 0xFF, 0x55]);
    }

    #[test]
    fn lshift_u8() {
        let mut pixels: Vec<u8> = vec![1, 2, 4];
        apply_chain_to_slice::<U8, _>(&Lshift::new(2), &mut pixels);
        assert_eq!(pixels, vec![4, 8, 16]);
    }

    #[test]
    fn rshift_u8() {
        let mut pixels: Vec<u8> = vec![16, 8, 255];
        apply_chain_to_slice::<U8, _>(&Rshift::new(2), &mut pixels);
        assert_eq!(pixels, vec![4, 2, 63]);
    }

    #[test]
    fn fused_xor_and_chain() {
        // XOR 0xFF (invert bits) then AND 0x0F (keep low nibble)
        let chain = (BoolXor::new(0xFF), BoolAnd::new(0x0F));
        let mut pixels: Vec<u8> = vec![0xAB]; // ~0xAB = 0x54, & 0x0F = 0x04
        apply_chain_to_slice::<U8, _>(&chain, &mut pixels);
        assert_eq!(pixels, vec![0x04]);
    }

    #[test]
    fn bool_and_wide_and_width() {
        let op = BoolAnd::new(0x0F);
        assert_eq!(
            <BoolAnd as Concretize>::apply_wide::<f32>(&op, 0xAB as f32),
            0x0B as f32
        );
        assert_eq!(<BoolAnd as Concretize>::min_width(&op), Width::Native);
    }

    #[test]
    fn bool_or_wide_and_width() {
        let op = BoolOr::new(0x0F);
        assert_eq!(
            <BoolOr as Concretize>::apply_wide::<f32>(&op, 0xA0 as f32),
            0xAF as f32
        );
        assert_eq!(<BoolOr as Concretize>::min_width(&op), Width::Native);
    }

    #[test]
    fn bool_xor_wide_and_width() {
        let op = BoolXor::new(0xFF);
        assert_eq!(
            <BoolXor as Concretize>::apply_wide::<f32>(&op, 0xAA as f32),
            0x55 as f32
        );
        assert_eq!(<BoolXor as Concretize>::min_width(&op), Width::Native);
    }

    #[test]
    fn lshift_wide_and_width() {
        let op = Lshift::new(2);
        assert_eq!(<Lshift as Concretize>::apply_wide::<i16>(&op, 4), 16);
        assert_eq!(<Lshift as Concretize>::min_width(&op), Width::I16);
    }

    #[test]
    fn rshift_wide_and_width() {
        let op = Rshift::new(2);
        assert_eq!(<Rshift as Concretize>::apply_wide::<f32>(&op, 255.0), 63.0);
        assert_eq!(<Rshift as Concretize>::min_width(&op), Width::Native);
    }
}
