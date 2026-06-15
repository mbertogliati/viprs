//! Colour conversion and colour-difference operations for image pipelines.
pub mod bridge;
/// Provides conversion support from `bw` to `srgb`.
pub mod bw_to_srgb;
/// Provides the `cicp2scrgb` module for this domain area.
pub mod cicp2scrgb;
/// Provides the `cmyk` module for this domain area.
pub mod cmyk;
/// Provides conversion support from `cmyk` to `xyz`.
pub mod cmyk_to_xyz;
/// Provides the `de00` module for this domain area.
pub mod de00;
/// Provides the `de76` module for this domain area.
pub mod de76;
/// Provides the `decmc` module for this domain area.
pub mod decmc;
/// Provides conversion support from `float` to `radiance`.
pub mod float_to_radiance;
/// Provides conversion support from `hsv` to `srgb`.
pub mod hsv_to_srgb;
/// Provides the `icc` module for this domain area.
pub mod icc;
/// Provides conversion support from `lab` to `labq`.
pub mod lab_to_labq;
/// Provides conversion support from `lab` to `labs`.
pub mod lab_to_labs;
/// Provides conversion support from `lab` to `lch`.
pub mod lab_to_lch;
/// Provides conversion support from `lab` to `srgb`.
pub mod lab_to_srgb;
/// Provides conversion support from `lab` to `xyz`.
pub mod lab_to_xyz;
/// Provides conversion support from `labq` to `lab`.
pub mod labq_to_lab;
/// Provides conversion support from `labq` to `labs`.
pub mod labq_to_labs;
/// Provides conversion support from `labq` to `srgb`.
pub mod labq_to_srgb;
/// Provides conversion support from `labs` to `lab`.
pub mod labs_to_lab;
/// Provides conversion support from `labs` to `labq`.
pub mod labs_to_labq;
/// Provides conversion support from `lch` to `lab`.
pub mod lch_to_lab;
/// Provides conversion support from `lch` to `ucs`.
pub mod lch_to_ucs;
mod math;
/// Provides the `oklab` module for this domain area.
pub mod oklab;
/// Provides conversion support from `radiance` to `float`.
pub mod radiance_to_float;
/// Provides conversion support from `rgb16` to `srgb`.
pub mod rgb16_to_srgb;
/// Provides conversion support from `scrgb` to `bw`.
pub mod scrgb_to_bw;
/// Provides conversion support from `scrgb` to `srgb`.
pub mod scrgb_to_srgb;
/// Provides conversion support from `scrgb` to `xyz`.
pub mod scrgb_to_xyz;
/// Provides the `srgb_lab_adjust` module for this domain area.
pub mod srgb_lab_adjust;
/// Provides the `srgb_lab_roundtrip` module for this domain area.
pub mod srgb_lab_roundtrip;
/// Provides conversion support from `srgb` to `hsv`.
pub mod srgb_to_hsv;
/// Provides conversion support from `srgb` to `lab`.
pub mod srgb_to_lab;
/// Provides conversion support from `srgb` to `rgb16`.
pub mod srgb_to_rgb16;
/// Provides conversion support from `srgb` to `scrgb`.
pub mod srgb_to_scrgb;
/// Provides conversion support from `srgb` to `xyz`.
pub mod srgb_to_xyz;
/// Provides conversion support from `ucs` to `lch`.
pub mod ucs_to_lch;
/// Provides the `uhdr2scrgb` module for this domain area.
pub mod uhdr2scrgb;
/// Provides conversion support from `xyz` to `cmyk`.
pub mod xyz_to_cmyk;
/// Provides conversion support from `xyz` to `lab`.
pub mod xyz_to_lab;
/// Provides conversion support from `xyz` to `scrgb`.
pub mod xyz_to_scrgb;
/// Provides conversion support from `xyz` to `srgb`.
pub mod xyz_to_srgb;
/// Provides conversion support from `xyz` to `yxy`.
pub mod xyz_to_yxy;
/// Provides conversion support from `yxy` to `xyz`.
pub mod yxy_to_xyz;

pub use bridge::ColourConvertBridge;
pub use bw_to_srgb::BwToSRgb;
pub use cicp2scrgb::{
    CicpColourPrimaries, CicpMatrixCoefficients, CicpProfile, CicpToScRgb,
    CicpTransferCharacteristics,
};
pub use cmyk::{CmykToRgbOp, RgbToCmykOp};
pub use cmyk_to_xyz::CmykToXyz;
pub use de00::DE00;
pub use de76::DE76;
pub use decmc::DECMC;
pub use float_to_radiance::FloatToRadiance;
pub use hsv_to_srgb::HsvToSRgb;
pub use icc::{
    IccImage, IccIntent, IccTransformOptions, icc_export, icc_import, icc_transform, profile_load,
};
pub use lab_to_labq::LabToLabQ;
pub use lab_to_labs::LabToLabS;
pub use lab_to_lch::LabToLch;
pub use lab_to_srgb::LabToSRgb;
pub use lab_to_xyz::LabToXyz;
pub use labq_to_lab::LabQToLab;
pub use labq_to_labs::LabQToLabS;
pub use labq_to_srgb::LabQToSRgb;
pub use labs_to_lab::LabSToLab;
pub use labs_to_labq::LabSToLabQ;
pub use lch_to_lab::LchToLab;
pub use lch_to_ucs::LchToUcs;
pub use oklab::{OklabToOklch, OklabToXyz, OklchToOklab, XyzToOklab};
pub use radiance_to_float::RadianceToFloat;
pub use rgb16_to_srgb::Rgb16ToSRgb;
pub use scrgb_to_bw::ScRgbToBw;
pub use scrgb_to_srgb::ScRgbToSRgb;
pub use scrgb_to_xyz::ScRgbToXyz;
pub use srgb_lab_adjust::SRgbLabAdjust;
pub use srgb_lab_roundtrip::SRgbLabRoundtrip;
pub use srgb_to_hsv::SRgbToHsv;
pub use srgb_to_lab::SRgbToLab;
pub use srgb_to_rgb16::SRgbToRgb16;
pub use srgb_to_scrgb::SRgbToScRgb;
pub use srgb_to_xyz::SRgbToXyz;
pub use ucs_to_lch::UcsToLch;
pub use uhdr2scrgb::{UhdrGainMapMetadata, UhdrToScRgb};
pub use xyz_to_cmyk::XyzToCmyk;
pub use xyz_to_lab::XyzToLab;
pub use xyz_to_scrgb::XyzToScRgb;
pub use xyz_to_srgb::XyzToSRgb;
pub use xyz_to_yxy::XyzToYxy;
pub use yxy_to_xyz::YxyToXyz;
