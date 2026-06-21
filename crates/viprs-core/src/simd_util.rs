//! SIMD utility helpers shared by image processing operations.
#![allow(dead_code)]
// REASON: architecture selectors are intentionally kept for future runtime dispatch expansion.

/// Utilities and macros for writing SIMD-dispatched pixel operations.
///
/// # Dispatch pattern
///
/// vipers stores a typed function pointer in each op struct that is resolved
/// once at construction time by calling `SimdLevel::detect()`. The hot path
/// (`process_region`) calls through the pointer with zero branches.
///
/// The pattern for an op with an AVX2 and a scalar path looks like:
///
/// ```rust,ignore
/// use crate::domain::ops::simd_util::SimdFnPtr;
/// use crate::simd::SimdLevel;
///
/// pub struct MyOp {
///     // … fields …
///     kernel: SimdFnPtr<(&[f32], &mut [f32])>,
/// }
///
/// impl MyOp {
///     pub fn new(/* … */) -> Self {
///         let kernel = simd_select_f32!(avx2 => avx2_kernel, scalar => scalar_kernel);
///         Self { /* … */ kernel }
///     }
/// }
/// ```
///
/// # `simd_select!` macro
///
/// Selects the best function pointer for the current CPU at the call site.
/// Expands to a runtime `SimdLevel::detect()` call. Zero heap allocation; the
/// result is a plain `fn` pointer stored in the op struct.
///
/// Syntax:
/// ```rust,ignore
/// let fp = simd_select! {
///     avx2  => avx2_fn,
///     _     => scalar_fn,
/// };
/// ```
///
/// The macro tries each arm in order of descending capability and falls through
/// to the last (scalar) arm if no SIMD level matches. Arms must be listed from
/// highest to lowest capability.
use crate::simd::SimdLevel;

/// A plain function-pointer type alias for unary slice operations on `f32`.
///
/// The concrete signature `fn(&[f32], &mut [f32])` covers the most common
/// unary ops (`Log`, `Exp`, `Abs`, `Round`, …). Binary ops can define their own
/// alias using the same pattern.
pub type UnaryF32Fn = fn(src: &[f32], dst: &mut [f32]);

/// A plain function-pointer type alias for unary slice operations on `u8`.
pub type UnaryU8Fn = fn(src: &[u8], dst: &mut [u8]);

#[inline]
pub(crate) const fn select_avx512f_or_avx2<T: Copy>(
    level: SimdLevel,
    avx512f: T,
    avx2: T,
    scalar: T,
) -> T {
    match level {
        SimdLevel::Avx512F => avx512f,
        SimdLevel::Avx2 => avx2,
        SimdLevel::Neon | SimdLevel::Scalar => scalar,
    }
}

#[inline]
pub(crate) fn select_avx2<T: Copy>(level: SimdLevel, avx2: T, scalar: T) -> T {
    if level.has_avx2() { avx2 } else { scalar }
}

#[inline]
pub(crate) fn select_neon<T: Copy>(level: SimdLevel, neon: T, scalar: T) -> T {
    if level.has_neon() { neon } else { scalar }
}

/// Select the best available implementation for a pixel-path function.
///
/// # Usage
///
/// ```rust,ignore
/// use crate::domain::ops::simd_util::simd_select;
///
/// let kernel: fn(&[f32], &mut [f32]) = simd_select! {
///     avx2  => my_avx2_kernel_f32,
///     _     => my_scalar_kernel_f32,
/// };
/// ```
///
/// On non-x86 targets, or when the required feature is absent, this expands to
/// the scalar arm with zero runtime overhead beyond the initial `SimdLevel::detect()`
/// call (which reads a cached CPUID result).
///
/// # Arms
///
/// - `avx2  => fn`: selected when `SimdLevel::detect() >= SimdLevel::Avx2`
/// - `avx512f => fn`: selected when `SimdLevel::detect() >= SimdLevel::Avx512F`
/// - `neon  => fn`: selected when `SimdLevel::detect().has_neon()` (aarch64 only)
/// - `_     => fn`: scalar fallback, always reachable
///
/// List arms from highest to lowest capability. The macro evaluates them in order
/// and returns the first matching arm.
#[macro_export]
macro_rules! simd_select {
    // avx512f arm present
    (avx512f => $avx512f_fn:expr, avx2 => $avx2_fn:expr, _ => $scalar_fn:expr $(,)?) => {{
        use $crate::simd::SimdLevel;
        let level = SimdLevel::detect();
        $crate::simd_util::select_avx512f_or_avx2(level, $avx512f_fn, $avx2_fn, $scalar_fn)
    }};
    // avx2 arm only
    (avx2 => $avx2_fn:expr, _ => $scalar_fn:expr $(,)?) => {{
        use $crate::simd::SimdLevel;
        let level = SimdLevel::detect();
        $crate::simd_util::select_avx2(level, $avx2_fn, $scalar_fn)
    }};
    // neon arm only (aarch64)
    (neon => $neon_fn:expr, _ => $scalar_fn:expr $(,)?) => {{
        use $crate::simd::SimdLevel;
        let level = SimdLevel::detect();
        $crate::simd_util::select_neon(level, $neon_fn, $scalar_fn)
    }};
}

// Re-export so callers can write `simd_util::simd_select!(...)` or use the
// crate-root re-export `vipers::simd_select!(...)`.
pub use simd_select;

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_add(src: &[f32], dst: &mut [f32]) {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = *s + 1.0;
        }
    }

    fn mock_avx2_add(src: &[f32], dst: &mut [f32]) {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = *s + 2.0;
        }
    }

    fn mock_avx512f_add(src: &[f32], dst: &mut [f32]) {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = *s + 3.0;
        }
    }

    fn scalar_copy_u8(src: &[u8], dst: &mut [u8]) {
        dst.copy_from_slice(src);
    }

    fn mock_neon_copy_u8(src: &[u8], dst: &mut [u8]) {
        dst.copy_from_slice(src);
    }

    #[test]
    fn detect_returns_a_valid_variant() {
        match SimdLevel::detect() {
            SimdLevel::Scalar | SimdLevel::Neon | SimdLevel::Avx2 | SimdLevel::Avx512F => {}
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn detect_reports_neon_on_aarch64() {
        assert!(SimdLevel::detect().has_neon());
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn detect_reports_no_neon_on_other_targets() {
        assert!(!SimdLevel::detect().has_neon());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_feature_helpers_match_detected_level() {
        let level = SimdLevel::detect();

        assert_eq!(
            level.has_avx2(),
            matches!(level, SimdLevel::Avx2 | SimdLevel::Avx512F)
        );
        assert_eq!(level.has_avx512f(), matches!(level, SimdLevel::Avx512F));
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn non_x86_targets_never_report_avx_levels() {
        let level = SimdLevel::detect();

        assert!(!level.has_avx2());
        assert!(!level.has_avx512f());
    }

    #[test]
    fn helper_select_avx512f_or_avx2_covers_all_levels() {
        let scalar = select_avx512f_or_avx2(SimdLevel::Scalar, 3u8, 2u8, 1u8);
        let neon = select_avx512f_or_avx2(SimdLevel::Neon, 3u8, 2u8, 1u8);
        let avx2 = select_avx512f_or_avx2(SimdLevel::Avx2, 3u8, 2u8, 1u8);
        let avx512f = select_avx512f_or_avx2(SimdLevel::Avx512F, 3u8, 2u8, 1u8);

        assert_eq!(scalar, 1);
        assert_eq!(neon, 1);
        assert_eq!(avx2, 2);
        assert_eq!(avx512f, 3);
    }

    #[test]
    fn helper_select_avx2_respects_threshold() {
        assert_eq!(select_avx2(SimdLevel::Scalar, 2u8, 1u8), 1);
        assert_eq!(select_avx2(SimdLevel::Neon, 2u8, 1u8), 1);
        assert_eq!(select_avx2(SimdLevel::Avx2, 2u8, 1u8), 2);
        assert_eq!(select_avx2(SimdLevel::Avx512F, 2u8, 1u8), 2);
    }

    #[test]
    fn helper_select_neon_only_matches_neon() {
        assert_eq!(select_neon(SimdLevel::Scalar, 2u8, 1u8), 1);
        assert_eq!(select_neon(SimdLevel::Neon, 2u8, 1u8), 2);
        assert_eq!(select_neon(SimdLevel::Avx2, 2u8, 1u8), 1);
        assert_eq!(select_neon(SimdLevel::Avx512F, 2u8, 1u8), 1);
    }

    #[test]
    fn mock_kernels_are_callable_directly() {
        let src = [1.0f32, 2.0, 3.0];
        let mut avx2_dst = [0.0f32; 3];
        let mut avx512_dst = [0.0f32; 3];
        let src_u8 = [1u8, 2, 3, 4];
        let mut scalar_u8_dst = [0u8; 4];

        mock_avx2_add(&src, &mut avx2_dst);
        mock_avx512f_add(&src, &mut avx512_dst);
        scalar_copy_u8(&src_u8, &mut scalar_u8_dst);

        assert_eq!(avx2_dst, [3.0, 4.0, 5.0]);
        assert_eq!(avx512_dst, [4.0, 5.0, 6.0]);
        assert_eq!(scalar_u8_dst, src_u8);
    }

    #[test]
    fn simd_select_returns_callable_fn_pointer() {
        let kernel: UnaryF32Fn = simd_select! {
            avx2 => mock_avx2_add,
            _    => scalar_add,
        };
        let src = [1.0f32, 2.0, 3.0];
        let mut dst = [0.0f32; 3];
        kernel(&src, &mut dst);

        assert!(dst == [2.0, 3.0, 4.0] || dst == [3.0, 4.0, 5.0]);
    }

    #[test]
    fn simd_select_three_way_returns_callable_fn_pointer() {
        let kernel: UnaryF32Fn = simd_select! {
            avx512f => mock_avx512f_add,
            avx2    => mock_avx2_add,
            _       => scalar_add,
        };
        let src = [0.0f32, 1.0];
        let mut dst = [0.0f32; 2];
        kernel(&src, &mut dst);

        assert!(dst == [1.0, 2.0] || dst == [2.0, 3.0] || dst == [3.0, 4.0]);
    }

    #[test]
    fn simd_select_neon_returns_callable_u8_fn_pointer() {
        let kernel: UnaryU8Fn = simd_select! {
            neon => mock_neon_copy_u8,
            _    => scalar_copy_u8,
        };
        let src = [1u8, 2, 3, 4];
        let mut dst = [0u8; 4];
        kernel(&src, &mut dst);
        assert_eq!(dst, src);
    }
}
