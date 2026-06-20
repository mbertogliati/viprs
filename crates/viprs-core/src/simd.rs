/// SIMD capability level detected at runtime.
///
/// Variants are ordered by x86 capability for AVX2/AVX-512 comparisons. NEON is
/// placed between Scalar and Avx2 so that `has_avx2()` (`self >= Avx2`) correctly
/// returns false for NEON — the two ISAs are not comparable. This ordering keeps
/// x86 capability comparisons correct.
///
/// Design note: vipers uses runtime detection so that a single compiled
/// binary works on hardware ranging from no-SIMD to AVX-512. The cost is one
/// branch-prediction miss per pipeline construction (the call to `detect()`);
/// after that the chosen function pointer is stored in the op struct and the hot
/// path is branch-free.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// No SIMD — scalar fallback path.
    Scalar,
    /// ARM NEON (mandatory on all aarch64 processors, including Apple Silicon).
    ///
    /// Positioned before Avx2 in the enum so that `has_avx2()` returns false for
    /// NEON — the two ISAs are not comparable. Use `has_neon()` for ARM dispatch.
    Neon,
    /// 256-bit AVX2 (Intel Haswell 2013+, AMD Zen 2017+).
    Avx2,
    /// 512-bit AVX-512F (Intel Ice Lake 2019+, AMD Zen 4 2022+).
    ///
    /// Infrastructure reserved; no op implementations yet.
    Avx512F,
}

impl SimdLevel {
    /// Detect the highest SIMD level available on the current CPU at runtime.
    ///
    /// On `x86`/`x86_64`: uses `std::arch::is_x86_feature_detected!` which reads `CPUID`
    /// once and caches the result (Rust stdlib implementation detail). Subsequent
    /// calls are effectively free.
    ///
    /// On aarch64: NEON is mandatory for all `AArch64` processors (ARMv8-A baseline),
    /// so this always returns `Neon` without a runtime check. Apple Silicon (`M1`/`M2`/`M3`)
    /// is `aarch64` and therefore always returns `Neon`.
    ///
    /// On other targets this returns `Scalar`.
    // `unreachable_code` is expected on aarch64: the `#[cfg(target_arch = "aarch64")]`
    // block returns unconditionally on that target, making `SimdLevel::Scalar` dead.
    // On all other targets the code is live. The allow is the standard pattern for
    // multi-target fallback code that uses early-return inside cfg blocks.
    #[allow(unreachable_code)]
    #[allow(clippy::missing_const_for_fn)]
    // REASON: On aarch64 this function is trivially const (returns Neon unconditionally),
    // but on x86 it calls `is_x86_feature_detected!()` which is not const.
    #[inline]
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on all aarch64 processors (ARMv8-A baseline).
            // No runtime feature check needed — every aarch64 binary can use NEON.
            return Self::Neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("avx512f") {
            return Self::Avx512F;
        } else if std::arch::is_x86_feature_detected!("avx2") {
            return Self::Avx2;
        }
        Self::Scalar
    }

    /// Returns `true` if this level supports at least AVX2.
    ///
    /// Always returns `false` for `Neon` — ARM NEON and x86 AVX2 are not comparable.
    /// Use `has_neon()` for ARM dispatch.
    #[inline]
    #[must_use]
    pub fn has_avx2(self) -> bool {
        self >= Self::Avx2
    }

    /// Returns `true` if this level supports AVX-512F.
    #[inline]
    #[must_use]
    pub fn has_avx512f(self) -> bool {
        self >= Self::Avx512F
    }

    /// Returns `true` if this level is ARM NEON.
    ///
    /// NEON is mandatory on all aarch64 processors. On non-aarch64 targets
    /// this always returns `false`.
    #[inline]
    #[must_use]
    pub fn has_neon(self) -> bool {
        self == Self::Neon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_a_valid_level() {
        // Just verify it compiles and returns without panic. The actual level
        // depends on the CPU running the test — we cannot assert a specific value.
        let level = SimdLevel::detect();
        // Scalar is always a valid answer; higher levels are also valid.
        assert!(level >= SimdLevel::Scalar);
    }

    #[test]
    fn ordering_is_scalar_lt_neon_lt_avx2_lt_avx512f() {
        assert!(SimdLevel::Scalar < SimdLevel::Neon);
        assert!(SimdLevel::Neon < SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 < SimdLevel::Avx512F);
    }

    #[test]
    fn has_avx2_and_has_avx512f_are_consistent() {
        assert!(!SimdLevel::Scalar.has_avx2());
        assert!(!SimdLevel::Scalar.has_avx512f());
        // NEON is not comparable to AVX2 — has_avx2() must return false for Neon.
        assert!(!SimdLevel::Neon.has_avx2());
        assert!(!SimdLevel::Neon.has_avx512f());
        assert!(SimdLevel::Avx2.has_avx2());
        assert!(!SimdLevel::Avx2.has_avx512f());
        assert!(SimdLevel::Avx512F.has_avx2());
        assert!(SimdLevel::Avx512F.has_avx512f());
    }

    #[test]
    fn has_neon_is_consistent() {
        assert!(!SimdLevel::Scalar.has_neon());
        assert!(SimdLevel::Neon.has_neon());
        assert!(!SimdLevel::Avx2.has_neon());
        assert!(!SimdLevel::Avx512F.has_neon());
    }

    /// On aarch64, detect() must return Neon (NEON is mandatory on all aarch64 CPUs).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn detect_returns_neon_on_aarch64() {
        assert_eq!(SimdLevel::detect(), SimdLevel::Neon);
        assert!(SimdLevel::detect().has_neon());
    }
}
