//! Static fusion via the `Concretize` trait.
//!
//! `Concretize` is the foundation of the static fusion system. An op that
//! implements `Concretize` describes WHAT transformation to apply without
//! knowing the pixel format (`F`) at definition time. The format is resolved
//! once at pipeline flush/build time, enabling LLVM to monomorphize and
//! auto-vectorize the entire fused chain in a single pass.
//!
//! # Design
//!
//! - Ops are plain structs without generic parameters: `Invert`, `Linear { scale, offset }`
//! - `Concretize::apply_sample::<F>()` is generic over `F` — monomorphizes at call site
//! - `Concretize::apply_wide::<W>()` operates in a wider numeric type without intermediate
//!   clamping, enabling LLVM to produce tight SIMD loops (4-8× throughput improvement).
//! - Tuple `(A, B)` implements Concretize recursively → unlimited fusion depth
//! - A single loop over `&mut [F::Sample]` calling the fused chain achieves
//!   the same effect as hand-written SIMD: LLVM sees through the entire composition.
//!
//! # Wide accumulator path
//!
//! When processing U8 pixels through a chain of ops, intermediate values are kept in a
//! `WideAccum` type (f32 or i16) without clamping between ops. This eliminates N-1
//! unnecessary int↔float conversions and clamps, allowing LLVM to vectorize at full
//! SIMD width (4 lanes for f32, 8 lanes for i16).
//!
//! # Proven performance (see benches/fusion.rs)
//!
//! - Integer ops: 3.5-8× faster than the old runtime-fusion path
//! - Wide accumulator: 7-25× faster than per-op clamp for 8-op chains
//! - LLVM algebraic optimization: 4×invert → identity (eliminated entirely)
//! - Single memory pass regardless of chain depth

use crate::domain::format::{BandFormat, PointSample};

// ─── Wide accumulator types ─────────────────────────────────────────────────

/// Minimum numeric width required by an operation to avoid overflow/precision loss.
///
/// Ordered from narrowest (fastest) to widest (slowest). The chain's required
/// width is the max of all its ops' requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Width {
    /// Operation works in-type (e.g., invert u8→u8, boolean ops).
    Native,
    /// Requires 16-bit integer accumulator (e.g., linear with integer coefs on u8).
    I16,
    /// Requires 32-bit float accumulator (e.g., linear with fractional coefs).
    F32,
}

/// A numeric type suitable for intermediate accumulation without clamping.
///
/// Provides basic arithmetic primitives that ops use to express their logic
/// generically. The bridge selects the concrete `W` type (f32 or i16) based
/// on the chain's `min_width()`, then LLVM monomorphizes the entire loop.
///
/// # Invariant
///
/// `apply_wide` implementations MUST NOT clamp. Clamping happens exactly once,
/// at the end of the chain, via `to_u8_clamped()`.
pub trait WideAccum: Copy + 'static {
    /// The format's maximum value when operating on U8 pixels (255.0 or 255).
    const FORMAT_MAX_U8: Self;

    /// Creates this value from f64.
    fn from_f64(v: f64) -> Self;
    /// Creates this value from u8.
    fn from_u8(v: u8) -> Self;
    /// Converts this value to u8 clamped.
    fn to_u8_clamped(self) -> u8;

    #[must_use]
    /// Returns or performs add.
    fn add(self, rhs: Self) -> Self;
    #[must_use]
    /// Returns or performs sub.
    fn sub(self, rhs: Self) -> Self;
    #[must_use]
    /// Returns or performs mul.
    fn mul(self, rhs: Self) -> Self;
    #[must_use]
    /// Returns or performs mul add.
    fn mul_add(self, scale: Self, offset: Self) -> Self;
    #[must_use]
    /// Returns or performs neg.
    fn neg(self) -> Self;
    #[must_use]
    /// Returns or performs abs.
    fn abs(self) -> Self;
    #[must_use]
    /// Returns or performs min.
    fn min(self, rhs: Self) -> Self;
    #[must_use]
    /// Returns or performs max.
    fn max(self, rhs: Self) -> Self;

    // Transcendental escape hatch — ops that inherently need floating point
    // convert to f32, compute, convert back. For f32 this is identity.
    /// Converts this value to f32.
    fn to_f32(self) -> f32;
    /// Creates this value from f32.
    fn from_f32(v: f32) -> Self;
}

impl WideAccum for f32 {
    const FORMAT_MAX_U8: Self = 255.0;

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as Self
    }
    #[inline(always)]
    fn from_u8(v: u8) -> Self {
        Self::from(v)
    }
    #[inline(always)]
    fn to_u8_clamped(self) -> u8 {
        self.clamp(0.0, 255.0) as u8
    }
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline(always)]
    fn mul_add(self, scale: Self, offset: Self) -> Self {
        Self::mul_add(self, scale, offset)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Self::abs(self)
    }
    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self::min(self, rhs)
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self::max(self, rhs)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }
}

impl WideAccum for i16 {
    const FORMAT_MAX_U8: Self = 255;

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as Self
    }
    #[inline(always)]
    fn from_u8(v: u8) -> Self {
        Self::from(v)
    }
    #[inline(always)]
    fn to_u8_clamped(self) -> u8 {
        self.clamp(0, 255) as u8
    }
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
    #[inline(always)]
    fn mul_add(self, scale: Self, offset: Self) -> Self {
        self.wrapping_mul(scale).wrapping_add(offset)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Self::saturating_abs(self)
    }
    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Ord::min(self, rhs)
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Ord::max(self, rhs)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v as Self
    }
}

/// A point operation that can be applied to any pixel format.
///
/// The key property: `apply_sample` is generic over `F`, so the compiler
/// monomorphizes it at the call site. When multiple `Concretize` ops are
/// composed via tuples, LLVM sees the entire chain and optimizes globally.
///
/// # Rules
///
/// - Implementations MUST be `#[inline(always)]` on `apply_sample`
/// - Implementations MUST NOT allocate
/// - Implementations MUST NOT panic
/// - The op MUST preserve format (input `F::Sample` → output `F::Sample`)
pub trait Concretize: Send + Sync + 'static {
    /// Transform a single sample value.
    ///
    /// Generic over `F` so that the concrete arithmetic is chosen by the compiler
    /// based on the pixel format at the monomorphization site.
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample;

    /// Transform a value in the wide accumulator domain (no clamping).
    ///
    /// Generic over `W: WideAccum` so ops express their logic once using
    /// arithmetic primitives. The bridge selects the concrete W based on
    /// `min_width()` and LLVM monomorphizes the entire loop.
    ///
    /// # Contract
    ///
    /// - MUST NOT clamp the result
    /// - MUST be `#[inline(always)]`
    /// - Semantics must match `apply_sample` for U8 format (modulo clamping)
    fn apply_wide<W: WideAccum>(&self, x: W) -> W;

    /// The minimum numeric width this op requires for correct U8 accumulation.
    ///
    /// The chain's effective width is `max(op.min_width() for op in chain)`.
    fn min_width(&self) -> Width;

    /// Bulk-process a u8 slice with an optional SIMD-optimized path.
    ///
    /// Returns `true` if the bulk path was used, `false` to fall back to the
    /// standard per-element loop. Override this for ops that have hand-tuned
    /// SIMD kernels (e.g., invert, linear) to bypass the u8→i16→u8 widening.
    #[inline(always)]
    fn try_apply_bulk_u8(&self, _src: &[u8], _dst: &mut [u8]) -> bool {
        false
    }
}

// ─── Identity (base case for tuple chains) ──────────────────────────────────

impl Concretize for () {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        x
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        x
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::Native
    }
}

// ─── Tuple composition (left-to-right chaining) ─────────────────────────────

impl<A: Concretize, B: Concretize> Concretize for (A, B) {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        self.1.apply_sample::<F>(self.0.apply_sample::<F>(x))
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        self.1.apply_wide::<W>(self.0.apply_wide::<W>(x))
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        let a = self.0.min_width();
        let b = self.1.min_width();
        if a >= b { a } else { b }
    }
}

// ─── Helper: apply a Concretize chain to a pixel buffer ─────────────────────

/// Apply a `Concretize` chain to every sample in a mutable slice.
///
/// This is the hot loop that LLVM vectorizes. It must remain simple:
/// no bounds checks, no branches, no allocations.
///
/// # Examples
/// ```rust
/// # use viprs::domain::concretize::apply_chain_to_slice;
/// # use viprs::domain::format::U8;
/// let mut pixels = [1_u8, 2, 3];
/// apply_chain_to_slice::<U8, _>(&(), &mut pixels);
/// assert_eq!(pixels, [1, 2, 3]);
/// ```
#[inline(always)]
pub fn apply_chain_to_slice<F: BandFormat, C: Concretize>(chain: &C, pixels: &mut [F::Sample])
where
    F::Sample: PointSample,
{
    for px in pixels.iter_mut() {
        *px = chain.apply_sample::<F>(*px);
    }
}

/// Apply a `Concretize` chain from a source slice to a destination slice.
///
/// For use in `process_region`-style dispatch where src and dst are separate.
///
/// # Examples
/// ```rust
/// # use viprs::domain::concretize::apply_chain_src_dst;
/// # use viprs::domain::format::U8;
/// let src = [1_u8, 2, 3];
/// let mut dst = [0_u8; 3];
/// apply_chain_src_dst::<U8, _>(&(), &src, &mut dst);
/// assert_eq!(src, dst);
/// ```
#[inline(always)]
pub fn apply_chain_src_dst<F: BandFormat, C: Concretize>(
    chain: &C,
    src: &[F::Sample],
    dst: &mut [F::Sample],
) where
    F::Sample: PointSample,
{
    debug_assert_eq!(src.len(), dst.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = chain.apply_sample::<F>(*s);
    }
}

// ─── Wide accumulator loop (U8 fast path) ───────────────────────────────────

/// Apply a `Concretize` chain to U8 pixels using a wide accumulator.
///
/// Converts u8→W at entry, runs the entire chain in W without clamping,
/// then clamps and converts W→u8 at exit. This eliminates N-1 intermediate
/// conversions and clamps, enabling LLVM to vectorize at full SIMD width.
///
/// # Examples
/// ```rust
/// # use viprs::domain::concretize::apply_chain_wide_u8;
/// let src = [1_u8, 2, 3];
/// let mut dst = [0_u8; 3];
/// apply_chain_wide_u8::<f32, _>(&(), &src, &mut dst);
/// assert_eq!(src, dst);
/// ```
#[inline(always)]
pub fn apply_chain_wide_u8<W: WideAccum, C: Concretize>(chain: &C, src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        let w = W::from_u8(*s);
        *d = chain.apply_wide::<W>(w).to_u8_clamped();
    }
}

/// Apply a `Concretize` chain to U8 pixels in-place (same buffer for input and output).
///
/// # Examples
/// ```rust
/// # use viprs::domain::concretize::apply_chain_wide_u8_inplace;
/// let mut buf = [1_u8, 2, 3];
/// apply_chain_wide_u8_inplace::<f32, _>(&(), &mut buf);
/// assert_eq!(buf, [1, 2, 3]);
/// ```
#[inline(always)]
pub fn apply_chain_wide_u8_inplace<W: WideAccum, C: Concretize>(chain: &C, buf: &mut [u8]) {
    for sample in buf.iter_mut() {
        let w = W::from_u8(*sample);
        *sample = chain.apply_wide::<W>(w).to_u8_clamped();
    }
}

/// Apply a `Concretize` chain to U8 pixels, automatically selecting the
/// optimal wide accumulator type based on the chain's `min_width()`.
///
/// # Examples
/// ```rust
/// # use viprs::domain::concretize::apply_chain_wide_u8_auto;
/// let src = [1_u8, 2, 3];
/// let mut dst = [0_u8; 3];
/// apply_chain_wide_u8_auto(&(), &src, &mut dst);
/// assert_eq!(src, dst);
/// ```
#[inline(always)]
pub fn apply_chain_wide_u8_auto<C: Concretize>(chain: &C, src: &[u8], dst: &mut [u8]) {
    match chain.min_width() {
        Width::Native | Width::I16 => apply_chain_wide_u8::<i16, C>(chain, src, dst),
        Width::F32 => apply_chain_wide_u8::<f32, C>(chain, src, dst),
    }
}

// ─── Chain builder (ergonomic tuple composition) ─────────────────────────────

/// Zero-cost chain builder for composing `Concretize` ops.
///
/// Each `.then(op)` extends the chain at the type level via tuple nesting.
/// The final chain is a nested tuple that LLVM monomorphizes into a single
/// fused loop — no allocations, no vtables, no intermediate buffers.
///
/// # Examples
///
/// ```rust
/// # use viprs::domain::concretize::Chain;
/// let chain = Chain::new().then(());
/// let _ = chain;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Chain<C = ()> {
    pub(crate) inner: C,
}

impl Chain<()> {
    /// Start a new empty chain (identity).
    ///
    /// This gives callers a typed entry point for incremental fusion.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::concretize::Chain;
    /// let _chain = Chain::new();
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self { inner: () }
    }
}

impl Default for Chain<()> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Concretize> Chain<C> {
    /// Append an operation to the chain.
    ///
    /// Returns a new `Chain` with the extended type. Zero runtime cost —
    /// this is pure type-level composition.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::concretize::Chain;
    /// let chain = Chain::new().then(());
    /// let _ = chain;
    /// ```
    #[inline(always)]
    pub fn then<D: Concretize>(self, op: D) -> Chain<(C, D)> {
        Chain {
            inner: (self.inner, op),
        }
    }
}

impl<C: Concretize> Concretize for Chain<C> {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        self.inner.apply_sample::<F>(x)
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        self.inner.apply_wide::<W>(x)
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        self.inner.min_width()
    }
}

// TODO: re-enable tests when ops crate exists.
// #[cfg(test)]
// mod tests {
//     ...
// }
