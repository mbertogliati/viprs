//! Compile-time and runtime colorspace identifiers for viprs.
//!
//! This module keeps colorspace reasoning explicit: runtime metadata uses [`ColorspaceId`], while
//! generic colour operations use the sealed [`Colorspace`] marker trait.
//!
//! Design summary:
//! - `ColorspaceId` — runtime enum, stored in pipeline metadata.
//! - `Colorspace` — sealed marker trait with `const ID`, used as compile-time
//!   bounds on colorspace-aware ops. NOT a type parameter on `Tile` or `Op`.
//! - Concrete zero-sized types: `SRgb`, `Lab`, `Xyz`, `Yxy`, `Hsv`, `Lch`, `Ucs`
//!   (libvips CMC space), `Oklab`, `Oklch`, `Cmyk`, `Greyscale`, `ScRgb`, `Rgb16`,
//!   `Cicp`.

// Sealed to prevent external colorspace types. The set of colorspaces is
// finite — exhaustive match in conversion dispatch is a correctness guarantee.
mod private {
    pub trait Sealed {}
}

/// Runtime identifier for a colorspace interpretation.
///
/// Use this when colorspace is only known from metadata or decoded headers and must be matched at
/// runtime.
///
/// # Examples
/// ```rust
/// # use viprs::domain::colorspace::ColorspaceId;
/// assert_eq!(ColorspaceId::SRgb.band_count(), Some(3));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ColorspaceId {
    /// Standard sRGB (gamma-corrected, 3 bands).
    SRgb,
    /// CIE L*a*b* perceptual uniform (linear light, 3 bands).
    Lab,
    /// CIE XYZ tristimulus (linear light, 3 bands).
    Xyz,
    /// CIE Yxy tristimulus chromaticity (3 bands).
    Yxy,
    /// Hue-Saturation-Value (3 bands, cylindrical).
    Hsv,
    /// CIE L*C*h° (cylindrical Lab, 3 bands).
    Lch,
    /// libvips CMC perceptual space derived from `LCh` (historically named UCS).
    Ucs,
    /// Oklab perceptual space (3 bands).
    Oklab,
    /// Oklch cylindrical Oklab (3 bands).
    Oklch,
    /// CMYK subtractive (4 bands).
    Cmyk,
    /// Linear greyscale (1 band, no gamma).
    Greyscale,
    /// scRGB — linear sRGB (linear light, 3 bands, higher precision than sRGB).
    ScRgb,
    /// 16-bit linear RGB (3 bands). Used for HDR workflows.
    Rgb16,
    /// ITU-T H.273 coded RGB plus transfer metadata (3 bands).
    Cicp,
    /// Colorspace is not known (e.g., raw decoder output before probing).
    Unknown,
}

impl ColorspaceId {
    /// Expected number of bands for this colorspace, if fixed.
    /// Returns `None` for colorspaces where band count is not fixed by the spec.
    #[must_use]
    pub const fn band_count(self) -> Option<u32> {
        match self {
            Self::SRgb
            | Self::Lab
            | Self::Xyz
            | Self::Yxy
            | Self::Hsv
            | Self::Lch
            | Self::Ucs
            | Self::Oklab
            | Self::Oklch
            | Self::ScRgb
            | Self::Rgb16
            | Self::Cicp => Some(3),
            Self::Cmyk => Some(4),
            Self::Greyscale => Some(1),
            Self::Unknown => None,
        }
    }

    /// Returns true if this colorspace uses linear light (no gamma encoding).
    #[must_use]
    pub const fn is_linear(self) -> bool {
        matches!(
            self,
            Self::Lab | Self::Xyz | Self::ScRgb | Self::Rgb16 | Self::Greyscale
        )
    }

    /// Returns the libvips interpretation-native maximum alpha value.
    #[must_use]
    pub const fn max_alpha(self) -> f64 {
        match self {
            Self::Rgb16 => 65535.0,
            Self::ScRgb => 1.0,
            _ => 255.0,
        }
    }
}

/// Marker trait for colorspace types.
///
/// Sealed — only concrete types defined in this module can implement it.
/// Used as a compile-time bound on colorspace-aware operations, without
/// adding a type parameter to `Tile` or `Op`.
///
/// # Examples
/// ```rust
/// # use viprs::domain::colorspace::{Colorspace, SRgb};
/// assert_eq!(SRgb::ID, <SRgb as Colorspace>::ID);
/// ```
pub trait Colorspace: private::Sealed + Send + Sync + 'static {
    /// Runtime identifier for this colorspace.
    const ID: ColorspaceId;
}

// ── Concrete zero-sized colorspace types ─────────────────────────────────────

/// Standard sRGB (gamma-corrected, 3 bands).
pub struct SRgb;
/// CIE L*a*b* perceptual (3 bands). Typically uses F32 samples.
pub struct Lab;
/// CIE XYZ tristimulus (3 bands). Typically uses F32 samples.
pub struct Xyz;
/// CIE Yxy chromaticity (3 bands).
pub struct Yxy;
/// Hue-Saturation-Value (3 bands, cylindrical).
pub struct Hsv;
/// CIE L*C*h° — cylindrical Lab (3 bands).
pub struct Lch;
/// libvips CMC perceptual space derived from `LCh` (historically named UCS).
pub struct Ucs;
/// Oklab perceptual space (3 bands).
pub struct Oklab;
/// Oklch — cylindrical Oklab (3 bands).
pub struct Oklch;
/// CMYK subtractive (4 bands).
pub struct Cmyk;
/// Linear greyscale (1 band).
pub struct Greyscale;
/// scRGB — linear sRGB, higher precision than sRGB (3 bands).
pub struct ScRgb;
/// 16-bit linear RGB (3 bands).
pub struct Rgb16;
/// ITU-T H.273 coded RGB plus transfer metadata (3 bands).
pub struct Cicp;
/// Placeholder when colorspace is not yet known (e.g., pre-probe decoder output).
///
/// Operations that require a specific colorspace do NOT accept `Unknown`.
/// The user must call `identify_colorspace` or use `with_colorspace` to
/// resolve it before applying colourspace-aware ops.
pub struct Unknown;

// ── Sealed impls ─────────────────────────────────────────────────────────────

impl private::Sealed for SRgb {}
impl private::Sealed for Lab {}
impl private::Sealed for Xyz {}
impl private::Sealed for Yxy {}
impl private::Sealed for Hsv {}
impl private::Sealed for Lch {}
impl private::Sealed for Ucs {}
impl private::Sealed for Oklab {}
impl private::Sealed for Oklch {}
impl private::Sealed for Cmyk {}
impl private::Sealed for Greyscale {}
impl private::Sealed for ScRgb {}
impl private::Sealed for Rgb16 {}
impl private::Sealed for Cicp {}
impl private::Sealed for Unknown {}

// ── Colorspace impls ─────────────────────────────────────────────────────────

impl Colorspace for SRgb {
    const ID: ColorspaceId = ColorspaceId::SRgb;
}
impl Colorspace for Lab {
    const ID: ColorspaceId = ColorspaceId::Lab;
}
impl Colorspace for Xyz {
    const ID: ColorspaceId = ColorspaceId::Xyz;
}
impl Colorspace for Yxy {
    const ID: ColorspaceId = ColorspaceId::Yxy;
}
impl Colorspace for Hsv {
    const ID: ColorspaceId = ColorspaceId::Hsv;
}
impl Colorspace for Lch {
    const ID: ColorspaceId = ColorspaceId::Lch;
}
impl Colorspace for Ucs {
    const ID: ColorspaceId = ColorspaceId::Ucs;
}
impl Colorspace for Oklab {
    const ID: ColorspaceId = ColorspaceId::Oklab;
}
impl Colorspace for Oklch {
    const ID: ColorspaceId = ColorspaceId::Oklch;
}
impl Colorspace for Cmyk {
    const ID: ColorspaceId = ColorspaceId::Cmyk;
}
impl Colorspace for Greyscale {
    const ID: ColorspaceId = ColorspaceId::Greyscale;
}
impl Colorspace for ScRgb {
    const ID: ColorspaceId = ColorspaceId::ScRgb;
}
impl Colorspace for Rgb16 {
    const ID: ColorspaceId = ColorspaceId::Rgb16;
}
impl Colorspace for Cicp {
    const ID: ColorspaceId = ColorspaceId::Cicp;
}
impl Colorspace for Unknown {
    const ID: ColorspaceId = ColorspaceId::Unknown;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_has_three_bands() {
        assert_eq!(ColorspaceId::SRgb.band_count(), Some(3));
    }

    #[test]
    fn cmyk_has_four_bands() {
        assert_eq!(ColorspaceId::Cmyk.band_count(), Some(4));
    }

    #[test]
    fn oklab_and_yxy_have_three_bands() {
        assert_eq!(ColorspaceId::Oklab.band_count(), Some(3));
        assert_eq!(ColorspaceId::Oklch.band_count(), Some(3));
        assert_eq!(ColorspaceId::Yxy.band_count(), Some(3));
        assert_eq!(ColorspaceId::Ucs.band_count(), Some(3));
    }

    #[test]
    fn lab_is_linear() {
        assert!(ColorspaceId::Lab.is_linear());
        assert!(!ColorspaceId::SRgb.is_linear());
    }

    #[test]
    fn max_alpha_matches_libvips_interpretations() {
        assert_eq!(ColorspaceId::SRgb.max_alpha(), 255.0);
        assert_eq!(ColorspaceId::Lab.max_alpha(), 255.0);
        assert_eq!(ColorspaceId::ScRgb.max_alpha(), 1.0);
        assert_eq!(ColorspaceId::Rgb16.max_alpha(), 65535.0);
    }

    #[test]
    fn colorspace_ids_accessible_at_compile_time() {
        const _: ColorspaceId = SRgb::ID;
        const _: ColorspaceId = Lab::ID;
        const _: ColorspaceId = Oklab::ID;
        const _: ColorspaceId = Yxy::ID;
        const _: ColorspaceId = Unknown::ID;
    }
}
