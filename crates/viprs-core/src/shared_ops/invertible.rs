//! Shared inversion trait and sample implementations used by pixel ops.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vdupq_n_f32, vdupq_n_u8, vdupq_n_u16, vld1q_f32, vld1q_u8, vld1q_u16, vst1q_f32, vst1q_u8,
    vst1q_u16, vsubq_f32, vsubq_u8, vsubq_u16,
};

/// Per-type inversion of a sample value.
///
/// This trait is an implementation detail of the `Invert` operation; it is not
/// part of the public port surface (`ports/`) because it is only meaningful in
/// the context of pixel-channel inversion.
pub trait Invertible: Copy {
    #[must_use]
    /// Returns or performs invert.
    fn invert(self) -> Self;

    /// Bulk inversion of a slice. Default delegates element-wise; types with SIMD
    /// acceleration override this for throughput.
    #[inline]
    fn invert_bulk(input: &[Self], output: &mut [Self]) {
        for (s, d) in input.iter().zip(output.iter_mut()) {
            *d = s.invert();
        }
    }
}

impl Invertible for u8 {
    #[inline]
    fn invert(self) -> Self {
        255 - self
    }

    #[inline]
    fn invert_bulk(input: &[Self], output: &mut [Self]) {
        invert_bulk_u8(input, output);
    }
}
impl Invertible for u16 {
    #[inline]
    fn invert(self) -> Self {
        65535 - self
    }

    #[inline]
    fn invert_bulk(input: &[Self], output: &mut [Self]) {
        invert_bulk_u16(input, output);
    }
}
impl Invertible for i16 {
    fn invert(self) -> Self {
        self.saturating_neg()
    }
}
impl Invertible for i32 {
    fn invert(self) -> Self {
        self.saturating_neg()
    }
}
impl Invertible for u32 {
    fn invert(self) -> Self {
        Self::MAX - self
    }
}
impl Invertible for f32 {
    #[inline]
    fn invert(self) -> Self {
        1.0 - self
    }

    #[inline]
    fn invert_bulk(input: &[Self], output: &mut [Self]) {
        invert_bulk_f32(input, output);
    }
}
impl Invertible for f64 {
    fn invert(self) -> Self {
        1.0 - self
    }
}

#[inline]
fn invert_bulk_u8_scalar(input: &[u8], output: &mut [u8]) {
    for (s, d) in input.iter().zip(output.iter_mut()) {
        *d = 255 - *s;
    }
}

#[inline]
fn invert_bulk_u16_scalar(input: &[u16], output: &mut [u16]) {
    for (s, d) in input.iter().zip(output.iter_mut()) {
        *d = 65_535 - *s;
    }
}

#[inline]
fn invert_bulk_f32_scalar(input: &[f32], output: &mut [f32]) {
    for (s, d) in input.iter().zip(output.iter_mut()) {
        *d = 1.0 - *s;
    }
}

// ─── SIMD-accelerated bulk invert for u8 ───

#[cfg(target_arch = "aarch64")]
#[inline]
fn invert_bulk_u8(input: &[u8], output: &mut [u8]) {
    let len = input.len().min(output.len());
    let chunks = len / 64;
    let remainder = len % 64;
    let tail_start = chunks * 64;
    let tail_chunks_16 = remainder / 16;

    // SAFETY: aarch64 always has NEON. We process 64 bytes per iteration (4×16B).
    // Pointer arithmetic stays within bounds: we only process `chunks * 64` bytes.
    unsafe {
        let max = vdupq_n_u8(255);
        let src = input.as_ptr();
        let dst = output.as_mut_ptr();

        for i in 0..chunks {
            let base = i * 64;
            let v0 = vld1q_u8(src.add(base));
            let v1 = vld1q_u8(src.add(base + 16));
            let v2 = vld1q_u8(src.add(base + 32));
            let v3 = vld1q_u8(src.add(base + 48));
            vst1q_u8(dst.add(base), vsubq_u8(max, v0));
            vst1q_u8(dst.add(base + 16), vsubq_u8(max, v1));
            vst1q_u8(dst.add(base + 32), vsubq_u8(max, v2));
            vst1q_u8(dst.add(base + 48), vsubq_u8(max, v3));
        }

        // Process remaining 16-byte chunks
        for i in 0..tail_chunks_16 {
            let off = tail_start + i * 16;
            let v = vld1q_u8(src.add(off));
            vst1q_u8(dst.add(off), vsubq_u8(max, v));
        }
    }

    let scalar_start = tail_start + tail_chunks_16 * 16;
    invert_bulk_u8_scalar(&input[scalar_start..len], &mut output[scalar_start..len]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn invert_bulk_u8(input: &[u8], output: &mut [u8]) {
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime dispatch guarantees AVX2 support and the helper only touches
        // the common in-bounds prefix of `input` and `output`.
        unsafe {
            invert_bulk_u8_avx2(input, output);
        }
    } else {
        invert_bulk_u8_scalar(input, output);
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn invert_bulk_u8(input: &[u8], output: &mut [u8]) {
    invert_bulk_u8_scalar(input, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: caller must dispatch only when AVX2 is available and pass slices whose common
// prefix is valid for unaligned 32-byte loads/stores plus the scalar tail.
// REASON: SIMD intrinsics operate on unaligned memory via loadu/storeu; the pointer casts
// are intentional and remain in-bounds for every processed chunk.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn invert_bulk_u8_avx2(input: &[u8], output: &mut [u8]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256i, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_storeu_si256, _mm256_xor_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_storeu_si256, _mm256_xor_si256,
    };

    let len = input.len().min(output.len());
    let chunks = len / 64;
    let remainder_start = chunks * 64;
    let mask = _mm256_set1_epi8(-1);

    for chunk in 0..chunks {
        let base = chunk * 64;
        // SAFETY: `base + 64 <= len`, both pointers are valid for two unaligned 32-byte
        // vector loads/stores, and AVX2 availability is guaranteed by the caller.
        unsafe {
            let v0 = _mm256_loadu_si256(input.as_ptr().add(base).cast::<__m256i>());
            let v1 = _mm256_loadu_si256(input.as_ptr().add(base + 32).cast::<__m256i>());
            _mm256_storeu_si256(
                output.as_mut_ptr().add(base).cast::<__m256i>(),
                _mm256_xor_si256(v0, mask),
            );
            _mm256_storeu_si256(
                output.as_mut_ptr().add(base + 32).cast::<__m256i>(),
                _mm256_xor_si256(v1, mask),
            );
        }
    }

    invert_bulk_u8_scalar(
        &input[remainder_start..len],
        &mut output[remainder_start..len],
    );
}

// ─── SIMD-accelerated bulk invert for u16 ───

#[cfg(target_arch = "aarch64")]
#[inline]
fn invert_bulk_u16(input: &[u16], output: &mut [u16]) {
    let len = input.len().min(output.len());
    let chunks = len / 32;
    let remainder = len % 32;
    let tail_start = chunks * 32;
    let tail_8 = remainder / 8;

    // SAFETY: aarch64 NEON processes 8 u16 per 128-bit register. We unroll 4×.
    unsafe {
        let max = vdupq_n_u16(65535);
        let src = input.as_ptr();
        let dst = output.as_mut_ptr();

        for i in 0..chunks {
            let base = i * 32;
            let v0 = vld1q_u16(src.add(base));
            let v1 = vld1q_u16(src.add(base + 8));
            let v2 = vld1q_u16(src.add(base + 16));
            let v3 = vld1q_u16(src.add(base + 24));
            vst1q_u16(dst.add(base), vsubq_u16(max, v0));
            vst1q_u16(dst.add(base + 8), vsubq_u16(max, v1));
            vst1q_u16(dst.add(base + 16), vsubq_u16(max, v2));
            vst1q_u16(dst.add(base + 24), vsubq_u16(max, v3));
        }

        for i in 0..tail_8 {
            let off = tail_start + i * 8;
            let v = vld1q_u16(src.add(off));
            vst1q_u16(dst.add(off), vsubq_u16(max, v));
        }
    }

    let scalar_start = tail_start + tail_8 * 8;
    invert_bulk_u16_scalar(&input[scalar_start..len], &mut output[scalar_start..len]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn invert_bulk_u16(input: &[u16], output: &mut [u16]) {
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime dispatch guarantees AVX2 support and the helper only touches
        // the common in-bounds prefix of `input` and `output`.
        unsafe {
            invert_bulk_u16_avx2(input, output);
        }
    } else {
        invert_bulk_u16_scalar(input, output);
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn invert_bulk_u16(input: &[u16], output: &mut [u16]) {
    invert_bulk_u16_scalar(input, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: caller must dispatch only when AVX2 is available and pass slices whose common
// prefix is valid for unaligned 32-byte loads/stores plus the scalar tail.
// REASON: SIMD intrinsics operate on unaligned memory via loadu/storeu; the pointer casts
// are intentional and remain in-bounds for every processed chunk.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn invert_bulk_u16_avx2(input: &[u16], output: &mut [u16]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256i, _mm256_loadu_si256, _mm256_set1_epi16, _mm256_storeu_si256, _mm256_xor_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_set1_epi16, _mm256_storeu_si256, _mm256_xor_si256,
    };

    let len = input.len().min(output.len());
    let chunks = len / 32;
    let remainder_start = chunks * 32;
    let mask = _mm256_set1_epi16(-1);

    for chunk in 0..chunks {
        let base = chunk * 32;
        // SAFETY: `base + 32 <= len`, both pointers are valid for two unaligned 32-byte
        // vector loads/stores of u16 lanes, and AVX2 availability is guaranteed by the caller.
        unsafe {
            let v0 = _mm256_loadu_si256(input.as_ptr().add(base).cast::<__m256i>());
            let v1 = _mm256_loadu_si256(input.as_ptr().add(base + 16).cast::<__m256i>());
            _mm256_storeu_si256(
                output.as_mut_ptr().add(base).cast::<__m256i>(),
                _mm256_xor_si256(v0, mask),
            );
            _mm256_storeu_si256(
                output.as_mut_ptr().add(base + 16).cast::<__m256i>(),
                _mm256_xor_si256(v1, mask),
            );
        }
    }

    invert_bulk_u16_scalar(
        &input[remainder_start..len],
        &mut output[remainder_start..len],
    );
}

// ─── SIMD-accelerated bulk invert for f32 ───

#[cfg(target_arch = "aarch64")]
#[inline]
fn invert_bulk_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    let chunks = len / 16;
    let remainder = len % 16;
    let tail_start = chunks * 16;
    let tail_4 = remainder / 4;

    // SAFETY: NEON processes 4 f32 per 128-bit register. Unroll 4×.
    unsafe {
        let one = vdupq_n_f32(1.0);
        let src = input.as_ptr();
        let dst = output.as_mut_ptr();

        for i in 0..chunks {
            let base = i * 16;
            let v0 = vld1q_f32(src.add(base));
            let v1 = vld1q_f32(src.add(base + 4));
            let v2 = vld1q_f32(src.add(base + 8));
            let v3 = vld1q_f32(src.add(base + 12));
            vst1q_f32(dst.add(base), vsubq_f32(one, v0));
            vst1q_f32(dst.add(base + 4), vsubq_f32(one, v1));
            vst1q_f32(dst.add(base + 8), vsubq_f32(one, v2));
            vst1q_f32(dst.add(base + 12), vsubq_f32(one, v3));
        }

        for i in 0..tail_4 {
            let off = tail_start + i * 4;
            let v = vld1q_f32(src.add(off));
            vst1q_f32(dst.add(off), vsubq_f32(one, v));
        }
    }

    let scalar_start = tail_start + tail_4 * 4;
    invert_bulk_f32_scalar(&input[scalar_start..len], &mut output[scalar_start..len]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn invert_bulk_f32(input: &[f32], output: &mut [f32]) {
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime dispatch guarantees AVX2 support and the helper only touches
        // the common in-bounds prefix of `input` and `output`.
        unsafe {
            invert_bulk_f32_avx2(input, output);
        }
    } else {
        invert_bulk_f32_scalar(input, output);
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn invert_bulk_f32(input: &[f32], output: &mut [f32]) {
    invert_bulk_f32_scalar(input, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: caller must dispatch only when AVX2 is available and pass slices whose common
// prefix is valid for unaligned 32-byte loads/stores plus the scalar tail.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn invert_bulk_f32_avx2(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps};

    let len = input.len().min(output.len());
    let chunks = len / 16;
    let remainder_start = chunks * 16;
    let one = _mm256_set1_ps(1.0);

    for chunk in 0..chunks {
        let base = chunk * 16;
        // SAFETY: `base + 16 <= len`, both pointers are valid for two unaligned 8-lane
        // float loads/stores, and AVX2 availability is guaranteed by the caller.
        unsafe {
            let v0 = _mm256_loadu_ps(input.as_ptr().add(base));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(base + 8));
            _mm256_storeu_ps(output.as_mut_ptr().add(base), _mm256_sub_ps(one, v0));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 8), _mm256_sub_ps(one, v1));
        }
    }

    invert_bulk_f32_scalar(
        &input[remainder_start..len],
        &mut output[remainder_start..len],
    );
}
