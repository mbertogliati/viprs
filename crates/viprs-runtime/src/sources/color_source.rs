//! Color Source image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use std::marker::PhantomData;

use bytemuck::Pod;

use crate::{
    domain::{
        error::ViprsError,
        format::BandFormat,
        image::{DemandHint, Region},
    },
    ports::source::{ImageSource, RandomAccessSource},
};

/// Converts an `f64` color value to the target sample type.
///
/// This trait is local to `color_source` — it exists only to convert the
/// constructor's `Vec<f64>` color spec into the typed sample buffer that
/// `ColorSource` pre-allocates at construction time. It is NOT a port or
/// domain interface.
trait ColorSample: Copy + Pod + 'static {
    fn from_f64_color(v: f64) -> Self;
}

impl ColorSample for u8 {
    #[inline(always)]
    fn from_f64_color(v: f64) -> Self {
        v.round().clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl ColorSample for u16 {
    #[inline(always)]
    fn from_f64_color(v: f64) -> Self {
        v.round().clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl ColorSample for i16 {
    #[inline(always)]
    fn from_f64_color(v: f64) -> Self {
        v.round().clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl ColorSample for u32 {
    #[inline(always)]
    fn from_f64_color(v: f64) -> Self {
        v.round().clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl ColorSample for i32 {
    #[inline(always)]
    fn from_f64_color(v: f64) -> Self {
        v.round().clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl ColorSample for f32 {
    #[inline(always)]
    fn from_f64_color(v: f64) -> Self {
        v as Self
    }
}

impl ColorSample for f64 {
    #[inline(always)]
    fn from_f64_color(v: f64) -> Self {
        v
    }
}

/// Synthetic source that fills every pixel with the same solid color.
///
/// All color values are converted from `f64` to `F::Sample` **at construction time**
/// and stored as a pre-built per-pixel byte pattern (`pixel_bytes`). `read_region`
/// performs a series of byte copies from that pattern — equivalent to a typed `memset`
/// with no per-tile heap allocations. `F` remains a compile-time
/// type parameter.
///
/// Implements [`RandomAccessSource`]: every region request is independent and
/// returns the same constant data.
pub struct ColorSource<F: BandFormat> {
    width: u32,
    height: u32,
    bands: u32,
    /// Raw bytes of one pixel (`bands × size_of::<F::Sample>()` bytes).
    /// Pre-allocated at construction; copied verbatim into every pixel slot in
    /// `read_region` without any per-tile allocation.
    pixel_bytes: Box<[u8]>,
    _format: PhantomData<F>,
}

#[allow(private_bounds)] // ColorSample is intentionally private: it is an implementation detail that restricts F::Sample to supported primitive types without leaking the trait into the public API.
impl<F: BandFormat> ColorSource<F>
where
    F::Sample: ColorSample,
{
    /// Creates a new solid-color source.
    ///
    /// `color` must have exactly `bands` elements — each element is the value
    /// for the corresponding band, expressed as `f64` and cast to `F::Sample`
    /// at construction time.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] if `color.len() != bands`.
    #[allow(clippy::needless_pass_by_value)]
    // REASON: public API stability for callers that already own the color vector.
    pub fn new(width: u32, height: u32, bands: usize, color: Vec<f64>) -> Result<Self, ViprsError> {
        if color.len() != bands {
            return Err(ViprsError::Codec(format!(
                "ColorSource: color.len() ({}) != bands ({})",
                color.len(),
                bands
            )));
        }

        // Convert f64 color values to typed samples and then to raw bytes.
        // This is the only allocation — it happens once at construction, not
        // per tile.
        let samples: Vec<F::Sample> = color
            .iter()
            .map(|&v| F::Sample::from_f64_color(v))
            .collect();
        let pixel_bytes: Box<[u8]> = bytemuck::cast_slice(&samples).to_vec().into_boxed_slice();

        Ok(Self {
            width,
            height,
            bands: bands as u32,
            pixel_bytes,
            _format: PhantomData,
        })
    }
}

impl<F: BandFormat> ImageSource for ColorSource<F> {
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        // Color fill is order-independent; any tile size is equally efficient.
        DemandHint::Any
    }

    /// Fills `output` with the solid color.
    ///
    /// Each pixel is filled by copying `pixel_bytes` (pre-built at construction).
    /// There are no heap allocations here — `pixel_bytes` is a `Box<[u8]>` that
    /// lives for the lifetime of the source. The copy pattern is equivalent to a
    /// typed memset.
    #[inline]
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let pixel_len = self.pixel_bytes.len();
        let n_pixels = region.width as usize * region.height as usize;

        // Iterate over pixel slots and copy the pre-built pixel bytes into each.
        // No branching per sample — only a slice copy per pixel.
        for i in 0..n_pixels {
            let start = i * pixel_len;
            output[start..start + pixel_len].copy_from_slice(&self.pixel_bytes);
        }
        Ok(())
    }
}

/// `ColorSource` generates pixel data on the fly and can respond to any region.
impl<F: BandFormat> RandomAccessSource for ColorSource<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::format::{F32, F64, I16, U8, U16};
    use crate::ports::source::ImageSource;
    use proptest::prelude::*;

    fn make_rgb_source() -> ColorSource<U8> {
        ColorSource::new(8, 8, 3, vec![100.0, 150.0, 200.0]).unwrap()
    }

    #[test]
    fn color_source_dimensions_match_constructor() {
        let src = ColorSource::<U8>::new(10, 20, 3, vec![0.0, 128.0, 255.0]).unwrap();
        assert_eq!(ImageSource::width(&src), 10);
        assert_eq!(ImageSource::height(&src), 20);
        assert_eq!(ImageSource::bands(&src), 3);
    }

    #[test]
    fn color_source_fills_region_with_correct_color() {
        let src = make_rgb_source();
        let region = Region::new(0, 0, 4, 4);
        let mut output = vec![0u8; 4 * 4 * 3];
        ImageSource::read_region(&src, region, &mut output).unwrap();

        // Every pixel must be [100, 150, 200]
        for chunk in output.chunks_exact(3) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 150);
            assert_eq!(chunk[2], 200);
        }
    }

    #[test]
    fn color_source_single_band() {
        let src = ColorSource::<U8>::new(4, 4, 1, vec![42.0]).unwrap();
        let region = Region::new(0, 0, 4, 4);
        let mut output = vec![0u8; 16];
        ImageSource::read_region(&src, region, &mut output).unwrap();
        assert!(output.iter().all(|&b| b == 42));
    }

    #[test]
    fn color_source_f32_bands() {
        let src = ColorSource::<F32>::new(2, 2, 2, vec![0.5_f64, 1.0_f64]).unwrap();
        let region = Region::new(0, 0, 2, 2);
        let mut output = vec![0u8; 2 * 2 * 2 * 4]; // 2x2 pixels × 2 bands × 4 bytes/f32
        ImageSource::read_region(&src, region, &mut output).unwrap();

        // Cast output bytes back to f32 slices and check each pixel
        let samples: &[f32] = bytemuck::cast_slice(&output);
        for chunk in samples.chunks_exact(2) {
            assert!((chunk[0] - 0.5_f32).abs() < 1e-6);
            assert!((chunk[1] - 1.0_f32).abs() < 1e-6);
        }
    }

    #[test]
    fn color_source_mismatched_bands_returns_error() {
        let result = ColorSource::<U8>::new(4, 4, 3, vec![1.0, 2.0]); // 2 values, 3 bands
        assert!(result.is_err());
    }

    #[test]
    fn color_source_demand_hint_is_any() {
        let src = ColorSource::<U8>::new(4, 4, 1, vec![0.0]).unwrap();
        assert_eq!(ImageSource::demand_hint(&src), DemandHint::Any);
    }

    #[test]
    fn color_source_sub_region_still_correct() {
        let src = ColorSource::<U16>::new(100, 100, 1, vec![1024.0]).unwrap();
        let region = Region::new(50, 50, 10, 10);
        let mut output = vec![0u8; 10 * 10 * 2]; // 1 band × 2 bytes/u16
        ImageSource::read_region(&src, region, &mut output).unwrap();
        let samples: &[u16] = bytemuck::cast_slice(&output);
        assert!(samples.iter().all(|&s| s == 1024));
    }

    #[test]
    fn color_source_i16_negative_value() {
        let src = ColorSource::<I16>::new(2, 2, 1, vec![-32768.0]).unwrap();
        let region = Region::new(0, 0, 2, 2);
        let mut output = vec![0u8; 2 * 2 * 2];
        ImageSource::read_region(&src, region, &mut output).unwrap();
        let samples: &[i16] = bytemuck::cast_slice(&output);
        assert!(samples.iter().all(|&s| s == i16::MIN));
    }

    #[test]
    fn color_source_f64_bands() {
        let src = ColorSource::<F64>::new(2, 1, 1, vec![std::f64::consts::PI]).unwrap();
        let region = Region::new(0, 0, 2, 1);
        let mut output = vec![0u8; 2 * 8]; // 2 pixels × 1 band × 8 bytes/f64
        ImageSource::read_region(&src, region, &mut output).unwrap();
        let samples: &[f64] = bytemuck::cast_slice(&output);
        for &s in samples {
            assert!((s - std::f64::consts::PI).abs() < 1e-15);
        }
    }

    proptest! {
        #[test]
        fn prop_same_color_every_tile(
            width in 1u32..=64u32,
            height in 1u32..=64u32,
            r in 0u8..=255u8,
            g in 0u8..=255u8,
            b in 0u8..=255u8,
        ) {
            let src = ColorSource::<U8>::new(
                width,
                height,
                3,
                vec![r as f64, g as f64, b as f64],
            ).unwrap();

            let region = Region::new(0, 0, width, height);
            let mut output = vec![0u8; width as usize * height as usize * 3];
            ImageSource::read_region(&src, region, &mut output).unwrap();

            for chunk in output.chunks_exact(3) {
                prop_assert_eq!(chunk[0], r);
                prop_assert_eq!(chunk[1], g);
                prop_assert_eq!(chunk[2], b);
            }
        }

        #[test]
        fn prop_color_source_f32_identity(
            width in 1u32..=32u32,
            height in 1u32..=32u32,
            v in -1.0f64..1.0f64,
        ) {
            let src = ColorSource::<F32>::new(width, height, 1, vec![v]).unwrap();
            let region = Region::new(0, 0, width, height);
            let mut output = vec![0u8; width as usize * height as usize * 4];
            ImageSource::read_region(&src, region, &mut output).unwrap();
            let samples: &[f32] = bytemuck::cast_slice(&output);
            let expected = v as f32;
            for &s in samples {
                prop_assert!((s - expected).abs() < 1e-6_f32);
            }
        }
    }
}
