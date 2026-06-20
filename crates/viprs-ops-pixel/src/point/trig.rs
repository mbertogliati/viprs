//! Format-erased trigonometric and transcendental operations.

use viprs_core::concretize::{Concretize, WideAccum, Width};
use viprs_core::format::{BandFormat, PointSample};

/// Helper macro for transcendental ops that map through f32.
macro_rules! impl_transcendental {
    ($name:ident, $pt_method:ident, $f32_fn:expr) => {
        impl Concretize for $name {
            #[inline(always)]
            fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
            where
                F::Sample: PointSample,
            {
                x.$pt_method()
            }

            #[inline(always)]
            fn apply_wide<W: WideAccum>(&self, x: W) -> W {
                W::from_f32($f32_fn(x.to_f32()))
            }

            #[inline(always)]
            fn min_width(&self) -> Width {
                Width::F32
            }
        }
    };
}

/// Sine (input in radians).
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::Sin;
///
/// let op = Sin;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Sin;
impl_transcendental!(Sin, pt_sin, f32::sin);

/// Cosine (input in radians).
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::Cos;
///
/// let op = Cos;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Cos;
impl_transcendental!(Cos, pt_cos, f32::cos);

/// Tangent (input in radians).
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::Tan;
///
/// let op = Tan;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Tan;
impl_transcendental!(Tan, pt_tan, f32::tan);

/// Arcsine.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::ASin;
///
/// let op = ASin;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ASin;
impl_transcendental!(ASin, pt_asin, f32::asin);

/// Arccosine.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::ACos;
///
/// let op = ACos;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ACos;
impl_transcendental!(ACos, pt_acos, f32::acos);

/// Arctangent.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::ATan;
///
/// let op = ATan;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ATan;
impl_transcendental!(ATan, pt_atan, f32::atan);

/// Natural exponential (e^x).
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::Exp;
///
/// let op = Exp;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Exp;
impl_transcendental!(Exp, pt_exp, f32::exp);

/// Natural logarithm (ln). Zero maps to zero (libvips convention).
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::Log;
///
/// let op = Log;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Log;

impl Concretize for Log {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        x.pt_log()
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        let v = x.to_f32();
        W::from_f32(if v <= 0.0 { 0.0 } else { v.ln() })
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::F32
    }
}

/// Square root.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::Sqrt;
///
/// let op = Sqrt;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Sqrt;
impl_transcendental!(Sqrt, pt_sqrt, f32::sqrt);

/// Power: x^exponent. Zero maps to zero (libvips convention).
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::Power;
///
/// let op = Power::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Power {
    /// Stores the `exponent` value for this item.
    pub exponent: f64,
}

impl Power {
    #[must_use]
    /// Creates a new `Power`.
    pub const fn new(exponent: f64) -> Self {
        Self { exponent }
    }
}

impl Concretize for Power {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        x.pt_pow(self.exponent)
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        let v = x.to_f32();
        W::from_f32(if v == 0.0 {
            0.0
        } else {
            v.powf(self.exponent as f32)
        })
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::F32
    }
}

/// Round to nearest integer.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::Round;
///
/// let op = Round;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Round;
impl_transcendental!(Round, pt_round, f32::round);

/// Floor (round toward negative infinity).
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::Floor;
///
/// let op = Floor;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Floor;
impl_transcendental!(Floor, pt_floor, f32::floor);

/// Ceil (round toward positive infinity).
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::Ceil;
///
/// let op = Ceil;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Ceil;
impl_transcendental!(Ceil, pt_ceil, f32::ceil);

/// Sign: -1 for negative, 0 for zero, 1 for positive.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::trig::Sign;
///
/// let op = Sign;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Sign;

impl Concretize for Sign {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        x.pt_sign()
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        let v = x.to_f32();
        W::from_f32(if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        })
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::concretize::{Concretize, Width, apply_chain_to_slice};
    use viprs_core::format::{F32, F64, U8};

    #[test]
    fn sin_f32() {
        let mut pixels: Vec<f32> = vec![0.0, std::f32::consts::FRAC_PI_2];
        apply_chain_to_slice::<F32, _>(&Sin, &mut pixels);
        assert!((pixels[0] - 0.0).abs() < 1e-6);
        assert!((pixels[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cos_f64() {
        let mut pixels: Vec<f64> = vec![0.0, std::f64::consts::PI];
        apply_chain_to_slice::<F64, _>(&Cos, &mut pixels);
        assert!((pixels[0] - 1.0).abs() < 1e-10);
        assert!((pixels[1] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn exp_log_roundtrip_f32() {
        let chain = (Exp, Log);
        let mut pixels: Vec<f32> = vec![1.0, 2.0, 3.0];
        let original = pixels.clone();
        apply_chain_to_slice::<F32, _>(&chain, &mut pixels);
        for (a, b) in pixels.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "exp(log({})) != {}, got {}", b, b, a);
        }
    }

    #[test]
    fn sqrt_f32() {
        let mut pixels: Vec<f32> = vec![0.0, 1.0, 4.0, 9.0];
        apply_chain_to_slice::<F32, _>(&Sqrt, &mut pixels);
        assert!((pixels[2] - 2.0).abs() < 1e-6);
        assert!((pixels[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn power_f32() {
        let mut pixels: Vec<f32> = vec![2.0, 3.0];
        apply_chain_to_slice::<F32, _>(&Power::new(2.0), &mut pixels);
        assert!((pixels[0] - 4.0).abs() < 1e-5);
        assert!((pixels[1] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn round_floor_ceil_f32() {
        let mut r = vec![1.4f32, 1.5, 1.6];
        apply_chain_to_slice::<F32, _>(&Round, &mut r);
        assert_eq!(r, vec![1.0, 2.0, 2.0]);

        let mut f = vec![1.4f32, 1.9, -0.1];
        apply_chain_to_slice::<F32, _>(&Floor, &mut f);
        assert_eq!(f, vec![1.0, 1.0, -1.0]);

        let mut c = vec![1.1f32, 1.0, -0.1];
        apply_chain_to_slice::<F32, _>(&Ceil, &mut c);
        assert_eq!(c, vec![2.0, 1.0, 0.0]);
    }

    #[test]
    fn sign_f32() {
        let mut pixels: Vec<f32> = vec![-5.0, 0.0, 3.0];
        apply_chain_to_slice::<F32, _>(&Sign, &mut pixels);
        assert_eq!(pixels, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn sign_u8() {
        let mut pixels: Vec<u8> = vec![0, 1, 255];
        apply_chain_to_slice::<U8, _>(&Sign, &mut pixels);
        assert_eq!(pixels, vec![0, 1, 1]);
    }

    #[test]
    fn round_wide_and_min_width() {
        assert_eq!(<Round as Concretize>::apply_wide::<f32>(&Round, 1.6), 2.0);
        assert_eq!(<Round as Concretize>::min_width(&Round), Width::F32);
    }

    #[test]
    fn log_wide_handles_non_positive_values() {
        assert_eq!(<Log as Concretize>::apply_wide::<f32>(&Log, -1.0), 0.0);
        assert_eq!(<Log as Concretize>::apply_wide::<f32>(&Log, 0.0), 0.0);
        assert!(
            (<Log as Concretize>::apply_wide::<f32>(&Log, std::f32::consts::E) - 1.0).abs() < 1e-6
        );
        assert_eq!(<Log as Concretize>::min_width(&Log), Width::F32);
    }

    #[test]
    fn power_wide_zero_non_zero_and_width() {
        let op = Power::new(3.0);
        assert_eq!(<Power as Concretize>::apply_wide::<f32>(&op, 0.0), 0.0);
        assert_eq!(<Power as Concretize>::apply_wide::<f32>(&op, 2.0), 8.0);
        assert_eq!(<Power as Concretize>::min_width(&op), Width::F32);
    }

    #[test]
    fn sign_wide_covers_all_branches_and_width() {
        assert_eq!(<Sign as Concretize>::apply_wide::<f32>(&Sign, -3.0), -1.0);
        assert_eq!(<Sign as Concretize>::apply_wide::<f32>(&Sign, 0.0), 0.0);
        assert_eq!(<Sign as Concretize>::apply_wide::<f32>(&Sign, 3.0), 1.0);
        assert_eq!(<Sign as Concretize>::min_width(&Sign), Width::F32);
    }
}
