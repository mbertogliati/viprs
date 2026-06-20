use std::marker::PhantomData;

use bytemuck::Pod;

use viprs_core::{
    colorspace::{Cicp, ScRgb},
    colour::ColourConvert,
    format::{BandFormat, U8, U16},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

const SDR_WHITE_NITS: f32 = 80.0;
const HLG_OOTF_LUT_SIZE: usize = 4096;

const BT709_TO_BT709: [f32; 9] = [
    1.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, //
    0.0, 0.0, 1.0,
];

const BT2020_TO_BT709: [f32; 9] = [
    1.660_491,
    -0.587_641_1,
    -0.072_849_86,
    -0.124_550_47,
    1.132_899_9,
    -0.008_349_42,
    -0.018_150_76,
    -0.100_578_9,
    1.118_729_7,
];

const DCI_P3_TO_BT709: [f32; 9] = [
    1.157_516_4,
    -0.154_962_38,
    -0.002_554_03,
    -0.041_500_07,
    1.045_567_9,
    -0.004_067_85,
    -0.018_050_04,
    -0.078_578_27,
    1.096_628_3,
];

const DISPLAY_P3_TO_BT709: [f32; 9] = [
    1.224_940_2,
    -0.224_940_18,
    0.0,
    -0.042_056_955,
    1.042_056_9,
    0.0,
    -0.019_637_555,
    -0.078_636_04,
    1.098_273_6,
];

const BT470M_TO_BT709: [f32; 9] = [
    1.486_156_8,
    -0.403_554_92,
    -0.082_601_94,
    -0.025_101_11,
    0.954_024_7,
    0.071_076_42,
    -0.027_224,
    -0.044_095_23,
    1.071_319_2,
];

const BT470BG_TO_BT709: [f32; 9] = [
    1.044_043_2,
    -0.044_043_21,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.011_793_38,
    0.988_206_6,
];

const BT601_TO_BT709: [f32; 9] = [
    0.939_542_06,
    0.050_181_36,
    0.010_276_58,
    0.017_772_22,
    0.965_792_83,
    0.016_434_91,
    -0.001_621_6,
    -0.004_369_75,
    1.005_991_3,
];

const GENERIC_FILM_TO_BT709: [f32; 9] = [
    1.346_175_9,
    -0.339_195_07,
    -0.006_980_84,
    -0.047_351_02,
    1.066_051_5,
    -0.018_700_51,
    -0.021_664_98,
    -0.061_313_1,
    1.082_978_1,
];

const EBU3213_TO_BT709: [f32; 9] = [
    1.025_252_5,
    -0.026_547_53,
    0.001_295_08,
    0.019_393_51,
    0.948_028,
    0.032_578_48,
    -0.001_769_53,
    -0.001_442_32,
    1.003_211_9,
];

const BT709_LUMINANCE: [f32; 3] = [0.2126, 0.7152, 0.0722];
const BT2020_LUMINANCE: [f32; 3] = [0.2627, 0.6780, 0.0593];
const DCI_P3_LUMINANCE: [f32; 3] = [0.2095, 0.7216, 0.0689];
const DISPLAY_P3_LUMINANCE: [f32; 3] = [0.2290, 0.6917, 0.0793];
const BT470M_LUMINANCE: [f32; 3] = [0.2990, 0.5864, 0.1146];
const BT470BG_LUMINANCE: [f32; 3] = [0.2220, 0.7067, 0.0713];
const BT601_LUMINANCE: [f32; 3] = [0.2124, 0.7011, 0.0866];
const GENERIC_FILM_LUMINANCE: [f32; 3] = [0.2536, 0.6783, 0.0681];
const EBU3213_LUMINANCE: [f32; 3] = [0.2318, 0.6723, 0.0960];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
/// Enumerates the available cicp colour primaries values.
pub enum CicpColourPrimaries {
    /// Uses the `Bt709` variant of `CicpColourPrimaries`.
    Bt709 = 1,
    /// Uses the `Unspecified` variant of `CicpColourPrimaries`.
    Unspecified = 2,
    /// Uses the `Bt470M` variant of `CicpColourPrimaries`.
    Bt470M = 4,
    /// Uses the `Bt470Bg` variant of `CicpColourPrimaries`.
    Bt470Bg = 5,
    /// Uses the `Bt601` variant of `CicpColourPrimaries`.
    Bt601 = 6,
    /// Uses the `Smpte240` variant of `CicpColourPrimaries`.
    Smpte240 = 7,
    /// Uses the `GenericFilm` variant of `CicpColourPrimaries`.
    GenericFilm = 8,
    /// Uses the `Bt2020` variant of `CicpColourPrimaries`.
    Bt2020 = 9,
    /// Uses the `Smpte431` variant of `CicpColourPrimaries`.
    Smpte431 = 11,
    /// Uses the `Smpte432` variant of `CicpColourPrimaries`.
    Smpte432 = 12,
    /// Uses the `Ebu3213` variant of `CicpColourPrimaries`.
    Ebu3213 = 22,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
/// Enumerates the available cicp transfer characteristics values.
pub enum CicpTransferCharacteristics {
    /// Uses the `Bt709` variant of `CicpTransferCharacteristics`.
    Bt709 = 1,
    /// Uses the `Unspecified` variant of `CicpTransferCharacteristics`.
    Unspecified = 2,
    /// Uses the `Bt470M` variant of `CicpTransferCharacteristics`.
    Bt470M = 4,
    /// Uses the `Bt470Bg` variant of `CicpTransferCharacteristics`.
    Bt470Bg = 5,
    /// Uses the `Bt601` variant of `CicpTransferCharacteristics`.
    Bt601 = 6,
    /// Uses the `Smpte240` variant of `CicpTransferCharacteristics`.
    Smpte240 = 7,
    /// Uses the `Linear` variant of `CicpTransferCharacteristics`.
    Linear = 8,
    /// Uses the `Log100` variant of `CicpTransferCharacteristics`.
    Log100 = 9,
    /// Uses the `Log100Sqrt10` variant of `CicpTransferCharacteristics`.
    Log100Sqrt10 = 10,
    /// Uses the `Iec61966` variant of `CicpTransferCharacteristics`.
    Iec61966 = 11,
    /// Uses the `Bt1361` variant of `CicpTransferCharacteristics`.
    Bt1361 = 12,
    /// Uses the `SRgb` variant of `CicpTransferCharacteristics`.
    SRgb = 13,
    /// Uses the `Bt2020_10Bit` variant of `CicpTransferCharacteristics`.
    Bt2020_10Bit = 14,
    /// Uses the `Bt2020_12Bit` variant of `CicpTransferCharacteristics`.
    Bt2020_12Bit = 15,
    /// Uses the `Pq` variant of `CicpTransferCharacteristics`.
    Pq = 16,
    /// Uses the `Smpte428` variant of `CicpTransferCharacteristics`.
    Smpte428 = 17,
    /// Uses the `Hlg` variant of `CicpTransferCharacteristics`.
    Hlg = 18,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
/// Enumerates the available cicp matrix coefficients values.
pub enum CicpMatrixCoefficients {
    /// Uses the `RgbIdentity` variant of `CicpMatrixCoefficients`.
    RgbIdentity = 0,
    /// Uses the `Bt709` variant of `CicpMatrixCoefficients`.
    Bt709 = 1,
    /// Uses the `Unspecified` variant of `CicpMatrixCoefficients`.
    Unspecified = 2,
    /// Uses the `Bt470Bg` variant of `CicpMatrixCoefficients`.
    Bt470Bg = 5,
    /// Uses the `Bt601` variant of `CicpMatrixCoefficients`.
    Bt601 = 6,
    /// Uses the `Smpte240` variant of `CicpMatrixCoefficients`.
    Smpte240 = 7,
    /// Uses the `Bt2020Ncl` variant of `CicpMatrixCoefficients`.
    Bt2020Ncl = 9,
    /// Uses the `Bt2020Cl` variant of `CicpMatrixCoefficients`.
    Bt2020Cl = 10,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Represents a cicp profile.
pub struct CicpProfile {
    /// Stores the `colour_primaries` value for this item.
    pub colour_primaries: CicpColourPrimaries,
    /// Stores the `transfer_characteristics` value for this item.
    pub transfer_characteristics: CicpTransferCharacteristics,
    /// Stores the `matrix_coefficients` value for this item.
    pub matrix_coefficients: CicpMatrixCoefficients,
    /// Stores the `full_range_flag` value for this item.
    pub full_range_flag: bool,
}

impl CicpProfile {
    #[must_use]
    /// Creates a new `CicpProfile`.
    pub const fn new(
        colour_primaries: CicpColourPrimaries,
        transfer_characteristics: CicpTransferCharacteristics,
        matrix_coefficients: CicpMatrixCoefficients,
        full_range_flag: bool,
    ) -> Self {
        Self {
            colour_primaries,
            transfer_characteristics,
            matrix_coefficients,
            full_range_flag,
        }
    }
}

/// Defines the contract for cicp lut format.
pub trait CicpLutFormat: BandFormat {
    /// Associated constant for lut size.
    const LUT_SIZE: usize;
    /// Converts this value to lut index.
    fn to_lut_index(sample: Self::Sample) -> usize;
}

impl CicpLutFormat for U8 {
    const LUT_SIZE: usize = 256;

    #[inline(always)]
    fn to_lut_index(sample: Self::Sample) -> usize {
        usize::from(sample)
    }
}

impl CicpLutFormat for U16 {
    const LUT_SIZE: usize = 65_536;

    #[inline(always)]
    fn to_lut_index(sample: Self::Sample) -> usize {
        usize::from(sample)
    }
}

/// Applies the `cicp2scrgb` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::cicp2scrgb::CicpToScRgb;
///
/// let op = CicpToScRgb::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct CicpToScRgb<F: BandFormat> {
    profile: CicpProfile,
    conversion_matrix: [f32; 9],
    luminance_coeffs: [f32; 3],
    transfer_lut: Box<[f32]>,
    hlg_ootf_lut: Option<Box<[f32]>>,
    _format: PhantomData<F>,
}

impl<F> CicpToScRgb<F>
where
    F: CicpLutFormat,
    F::Sample: Pod,
{
    /// Creates a new `CicpToScRgb`.
    pub fn new(profile: CicpProfile) -> Self {
        let (conversion_matrix, luminance_coeffs) = primaries_tables(profile.colour_primaries);
        let transfer_lut = build_transfer_lut::<F>(profile.transfer_characteristics);
        let hlg_ootf_lut = matches!(
            profile.transfer_characteristics,
            CicpTransferCharacteristics::Hlg
        )
        .then(build_hlg_ootf_lut);

        Self {
            profile,
            conversion_matrix,
            luminance_coeffs,
            transfer_lut,
            hlg_ootf_lut,
            _format: PhantomData,
        }
    }

    #[inline(always)]
    #[must_use]
    /// Returns or performs profile.
    pub const fn profile(&self) -> CicpProfile {
        self.profile
    }

    #[inline(always)]
    fn transfer_sample(&self, sample: F::Sample) -> f32 {
        self.transfer_lut[F::to_lut_index(sample)]
    }

    #[inline]
    fn process_tile(&self, input: &Tile<F>, output: &mut TileMut<viprs_core::format::F32>) {
        let is_hlg = matches!(
            self.profile.transfer_characteristics,
            CicpTransferCharacteristics::Hlg
        );

        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(3))
        {
            let mut red = self.transfer_sample(pixel_in[0]);
            let mut green = self.transfer_sample(pixel_in[1]);
            let mut blue = self.transfer_sample(pixel_in[2]);

            if is_hlg {
                apply_hlg_ootf(
                    &mut red,
                    &mut green,
                    &mut blue,
                    self.luminance_coeffs,
                    self.hlg_ootf_lut.as_deref().unwrap_or(&[]),
                );
            }

            let (out_red, out_green, out_blue) =
                apply_matrix(self.conversion_matrix, red, green, blue);
            pixel_out[0] = out_red;
            pixel_out[1] = out_green;
            pixel_out[2] = out_blue;
        }
    }
}

impl<F> ColourConvert<Cicp, ScRgb> for CicpToScRgb<F>
where
    F: CicpLutFormat,
    F::Sample: Pod,
{
    type InputFormat = F;
    type OutputFormat = viprs_core::format::F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn convert_region(
        &self,
        (): &mut Self::State,
        input: &Tile<Self::InputFormat>,
        output: &mut TileMut<Self::OutputFormat>,
    ) {
        self.process_tile(input, output);
    }
}

impl<F> Op for CicpToScRgb<F>
where
    F: CicpLutFormat,
    F::Sample: Pod,
{
    type Input = F;
    type Output = viprs_core::format::F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(
        &self,
        (): &mut Self::State,
        input: &Tile<Self::Input>,
        output: &mut TileMut<Self::Output>,
    ) {
        self.process_tile(input, output);
    }
}

impl<F> PixelLocalOp for CicpToScRgb<F>
where
    F: CicpLutFormat,
    F::Sample: Pod,
{
}

const fn primaries_tables(primaries: CicpColourPrimaries) -> ([f32; 9], [f32; 3]) {
    match primaries {
        CicpColourPrimaries::Bt709 | CicpColourPrimaries::Unspecified => {
            (BT709_TO_BT709, BT709_LUMINANCE)
        }
        CicpColourPrimaries::Bt2020 => (BT2020_TO_BT709, BT2020_LUMINANCE),
        CicpColourPrimaries::Smpte431 => (DCI_P3_TO_BT709, DCI_P3_LUMINANCE),
        CicpColourPrimaries::Smpte432 => (DISPLAY_P3_TO_BT709, DISPLAY_P3_LUMINANCE),
        CicpColourPrimaries::Bt470M => (BT470M_TO_BT709, BT470M_LUMINANCE),
        CicpColourPrimaries::Bt470Bg => (BT470BG_TO_BT709, BT470BG_LUMINANCE),
        CicpColourPrimaries::Bt601 | CicpColourPrimaries::Smpte240 => {
            (BT601_TO_BT709, BT601_LUMINANCE)
        }
        CicpColourPrimaries::GenericFilm => (GENERIC_FILM_TO_BT709, GENERIC_FILM_LUMINANCE),
        CicpColourPrimaries::Ebu3213 => (EBU3213_TO_BT709, EBU3213_LUMINANCE),
    }
}

#[inline(always)]
fn apply_matrix(matrix: [f32; 9], red: f32, green: f32, blue: f32) -> (f32, f32, f32) {
    (
        matrix[2].mul_add(blue, matrix[1].mul_add(green, matrix[0] * red)),
        matrix[5].mul_add(blue, matrix[4].mul_add(green, matrix[3] * red)),
        matrix[8].mul_add(blue, matrix[7].mul_add(green, matrix[6] * red)),
    )
}

#[inline(always)]
fn pq_eotf(signal: f32) -> f32 {
    const M1: f32 = 2610.0 / 16_384.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;

    if signal <= 0.0 {
        return 0.0;
    }

    let signal_m2 = signal.powf(1.0 / M2);
    let numerator = (signal_m2 - C1).max(0.0);
    let denominator = C3.mul_add(-signal_m2, C2);
    if denominator <= 0.0 {
        return 0.0;
    }

    let linear = (numerator / denominator).powf(1.0 / M1);
    linear * (10_000.0 / SDR_WHITE_NITS)
}

#[inline(always)]
fn hlg_inverse_oetf(signal: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 0.284_668_92;
    const C: f32 = 0.559_910_7;

    if signal <= 0.0 {
        0.0
    } else if signal <= 0.5 {
        signal * signal / 3.0
    } else {
        (((signal - C) / A).exp() + B) / 12.0
    }
}

#[inline(always)]
fn bt709_inverse_oetf(signal: f32) -> f32 {
    const ALPHA: f32 = 1.099_296_8;
    const LINEAR_BETA: f32 = 0.018_053_968;
    const SIGNAL_BETA: f32 = 4.5 * LINEAR_BETA;

    if signal < 0.0 {
        0.0
    } else if signal < SIGNAL_BETA {
        signal / 4.5
    } else {
        ((signal + (ALPHA - 1.0)) / ALPHA).powf(1.0 / 0.45)
    }
}

#[inline(always)]
fn srgb_inverse_oetf(signal: f32) -> f32 {
    if signal < 0.0 {
        0.0
    } else if signal <= 0.04045 {
        signal / 12.92
    } else {
        ((signal + 0.055) / 1.055).powf(2.4)
    }
}

#[inline(always)]
fn cicp_transfer(transfer: CicpTransferCharacteristics, input: f32) -> f32 {
    match transfer {
        CicpTransferCharacteristics::Pq => pq_eotf(input),
        CicpTransferCharacteristics::Hlg => hlg_inverse_oetf(input),
        CicpTransferCharacteristics::Bt709
        | CicpTransferCharacteristics::Bt601
        | CicpTransferCharacteristics::Bt2020_10Bit
        | CicpTransferCharacteristics::Bt2020_12Bit => bt709_inverse_oetf(input),
        CicpTransferCharacteristics::Smpte240 => {
            const ALPHA: f32 = 1.1115;
            const LINEAR_BETA: f32 = 0.0228;
            const SLOPE: f32 = 4.0;
            if input < SLOPE * LINEAR_BETA {
                input / SLOPE
            } else {
                ((input + (ALPHA - 1.0)) / ALPHA).powf(1.0 / 0.45)
            }
        }
        CicpTransferCharacteristics::SRgb => srgb_inverse_oetf(input),
        CicpTransferCharacteristics::Bt470M => input.max(0.0).powf(2.2),
        CicpTransferCharacteristics::Bt470Bg => input.max(0.0).powf(2.8),
        CicpTransferCharacteristics::Linear | CicpTransferCharacteristics::Unspecified => input,
        CicpTransferCharacteristics::Log100 => {
            if input > 0.0 {
                10.0_f32.powf(2.0 * (input - 1.0))
            } else {
                0.0
            }
        }
        CicpTransferCharacteristics::Log100Sqrt10 => {
            if input > 0.0 {
                10.0_f32.powf(2.5 * (input - 1.0))
            } else {
                0.0
            }
        }
        CicpTransferCharacteristics::Iec61966 | CicpTransferCharacteristics::Bt1361 => {
            if input >= 0.0 {
                bt709_inverse_oetf(input)
            } else {
                -bt709_inverse_oetf(-input)
            }
        }
        CicpTransferCharacteristics::Smpte428 => {
            let display_luminance = (52.37 / 48.0) * input.max(0.0).powf(2.6);
            display_luminance * (48.0 / SDR_WHITE_NITS)
        }
    }
}

fn build_transfer_lut<F: CicpLutFormat>(transfer: CicpTransferCharacteristics) -> Box<[f32]> {
    let scale = 1.0 / (F::LUT_SIZE - 1) as f32;
    (0..F::LUT_SIZE)
        .map(|index| cicp_transfer(transfer, index as f32 * scale))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn build_hlg_ootf_lut() -> Box<[f32]> {
    const GAMMA_MINUS_ONE: f32 = 0.2;
    const SCALE: f32 = 1000.0 / SDR_WHITE_NITS;

    let mut lut = vec![0.0; HLG_OOTF_LUT_SIZE];
    for (index, sample) in lut.iter_mut().enumerate().skip(1) {
        *sample = SCALE * (index as f32 / (HLG_OOTF_LUT_SIZE - 1) as f32).powf(GAMMA_MINUS_ONE);
    }
    lut.into_boxed_slice()
}

#[inline(always)]
fn apply_hlg_ootf(
    red: &mut f32,
    green: &mut f32,
    blue: &mut f32,
    luminance: [f32; 3],
    ootf_lut: &[f32],
) {
    let scene_luminance =
        luminance[2].mul_add(*blue, luminance[1].mul_add(*green, luminance[0] * *red));
    if scene_luminance <= 0.0 {
        *red = 0.0;
        *green = 0.0;
        *blue = 0.0;
        return;
    }

    let index = scene_luminance * (HLG_OOTF_LUT_SIZE - 1) as f32;
    let mut lo = index as usize;
    if lo >= HLG_OOTF_LUT_SIZE - 1 {
        lo = HLG_OOTF_LUT_SIZE - 2;
    }
    let fraction = index - lo as f32;
    let factor = ootf_lut[lo] + fraction * (ootf_lut[lo + 1] - ootf_lut[lo]);
    *red *= factor;
    *green *= factor;
    *blue *= factor;
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U8, U16},
        image::{Tile, TileMut},
    };

    const EPSILON: f32 = 1e-6;

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    fn linear_bt709_profile() -> CicpProfile {
        CicpProfile::new(
            CicpColourPrimaries::Bt709,
            CicpTransferCharacteristics::Linear,
            CicpMatrixCoefficients::RgbIdentity,
            true,
        )
    }

    fn run_u16(profile: CicpProfile, input_data: [u16; 3]) -> [f32; 3] {
        let op = CicpToScRgb::<U16>::new(profile);
        let region = make_region(1);
        let input = Tile::<U16>::new(region, 3, &input_data);
        let mut output_data = [0.0_f32; 3];
        let mut output = TileMut::<F32>::new(region, 3, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn assert_triplet_close(actual: [f32; 3], expected: [f32; 3], epsilon: f32) {
        for (actual_channel, expected_channel) in actual.into_iter().zip(expected) {
            assert!((actual_channel - expected_channel).abs() <= epsilon);
        }
    }

    proptest! {
        #[test]
        fn linear_bt709_u8_behaves_like_normalized_identity(
            red in any::<u8>(),
            green in any::<u8>(),
            blue in any::<u8>(),
        ) {
            let op = CicpToScRgb::<U8>::new(linear_bt709_profile());
            let region = make_region(1);
            let input_data = [red, green, blue];
            let input = Tile::<U8>::new(region, 3, &input_data);
            let mut output_data = [0.0_f32; 3];
            let mut output = TileMut::<F32>::new(region, 3, &mut output_data);

            op.process_region(&mut (), &input, &mut output);

            prop_assert!((output_data[0] - f32::from(red) / 255.0).abs() <= 1e-6);
            prop_assert!((output_data[1] - f32::from(green) / 255.0).abs() <= 1e-6);
            prop_assert!((output_data[2] - f32::from(blue) / 255.0).abs() <= 1e-6);
        }
    }

    #[test]
    fn linear_bt709_u16_max_maps_to_unit_white() {
        let op = CicpToScRgb::<U16>::new(linear_bt709_profile());
        let region = make_region(1);
        let input_data = [u16::MAX, u16::MAX, u16::MAX];
        let input = Tile::<U16>::new(region, 3, &input_data);
        let mut output_data = [0.0_f32; 3];
        let mut output = TileMut::<F32>::new(region, 3, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn op_reports_profile_and_identity_region_contract() {
        let op = CicpToScRgb::<U16>::new(linear_bt709_profile());
        let region = Region::new(3, 4, 5, 6);

        assert_eq!(op.profile(), linear_bt709_profile());
        assert_eq!(
            <CicpToScRgb<U16> as ColourConvert<Cicp, ScRgb>>::demand_hint(&op),
            DemandHint::Any
        );
        assert_eq!(<CicpToScRgb<U16> as Op>::demand_hint(&op), DemandHint::Any);
        assert_eq!(
            <CicpToScRgb<U16> as ColourConvert<Cicp, ScRgb>>::required_input_region(&op, &region),
            region
        );
        assert_eq!(
            <CicpToScRgb<U16> as Op>::required_input_region(&op, &region),
            region
        );

        <CicpToScRgb<U16> as ColourConvert<Cicp, ScRgb>>::start(&op);
        <CicpToScRgb<U16> as Op>::start(&op);
    }

    #[test]
    fn pq_white_maps_to_10000_nit_sc_rgb_reference() {
        let op = CicpToScRgb::<U16>::new(CicpProfile::new(
            CicpColourPrimaries::Bt709,
            CicpTransferCharacteristics::Pq,
            CicpMatrixCoefficients::RgbIdentity,
            true,
        ));
        let region = make_region(1);
        let input_data = [u16::MAX, u16::MAX, u16::MAX];
        let input = Tile::<U16>::new(region, 3, &input_data);
        let mut output_data = [0.0_f32; 3];
        let mut output = TileMut::<F32>::new(region, 3, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        let expected = 10_000.0 / SDR_WHITE_NITS;
        assert!((output_data[0] - expected).abs() < 1e-3);
        assert!((output_data[1] - expected).abs() < 1e-3);
        assert!((output_data[2] - expected).abs() < 1e-3);
    }

    #[test]
    fn pq_bt2020_near_zero_input_stays_small_and_finite() {
        let actual = run_u16(
            CicpProfile::new(
                CicpColourPrimaries::Bt2020,
                CicpTransferCharacteristics::Pq,
                CicpMatrixCoefficients::RgbIdentity,
                true,
            ),
            [1, 1, 1],
        );

        let signal = 1.0 / 65_535.0;
        let linear = cicp_transfer(CicpTransferCharacteristics::Pq, signal);
        let expected = apply_matrix(BT2020_TO_BT709, linear, linear, linear);

        assert!(actual.into_iter().all(|channel| channel.is_finite()));
        assert_triplet_close(actual, [expected.0, expected.1, expected.2], 5e-5);
    }

    #[test]
    fn pq_display_p3_near_max_saturated_red_matches_conversion_matrix() {
        let actual = run_u16(
            CicpProfile::new(
                CicpColourPrimaries::Smpte432,
                CicpTransferCharacteristics::Pq,
                CicpMatrixCoefficients::RgbIdentity,
                true,
            ),
            [u16::MAX - 1, 0, 0],
        );

        let red = cicp_transfer(
            CicpTransferCharacteristics::Pq,
            (u16::MAX - 1) as f32 / 65_535.0,
        );
        assert_triplet_close(
            actual,
            [
                DISPLAY_P3_TO_BT709[0] * red,
                DISPLAY_P3_TO_BT709[3] * red,
                DISPLAY_P3_TO_BT709[6] * red,
            ],
            1e-3,
        );
    }

    #[test]
    fn bt2020_primaries_are_mapped_to_bt709_sc_rgb() {
        let op = CicpToScRgb::<U16>::new(CicpProfile::new(
            CicpColourPrimaries::Bt2020,
            CicpTransferCharacteristics::Linear,
            CicpMatrixCoefficients::RgbIdentity,
            true,
        ));
        let region = make_region(1);
        let input_data = [u16::MAX, 0, 0];
        let input = Tile::<U16>::new(region, 3, &input_data);
        let mut output_data = [0.0_f32; 3];
        let mut output = TileMut::<F32>::new(region, 3, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert!((output_data[0] - BT2020_TO_BT709[0]).abs() < 1e-6);
        assert!((output_data[1] - BT2020_TO_BT709[3]).abs() < 1e-6);
        assert!((output_data[2] - BT2020_TO_BT709[6]).abs() < 1e-6);
    }

    #[test]
    fn generic_film_primaries_are_mapped_to_bt709_sc_rgb() {
        let actual = run_u16(
            CicpProfile::new(
                CicpColourPrimaries::GenericFilm,
                CicpTransferCharacteristics::Linear,
                CicpMatrixCoefficients::RgbIdentity,
                true,
            ),
            [0, u16::MAX, 0],
        );

        assert_triplet_close(
            actual,
            [
                GENERIC_FILM_TO_BT709[1],
                GENERIC_FILM_TO_BT709[4],
                GENERIC_FILM_TO_BT709[7],
            ],
            EPSILON,
        );
    }

    #[test]
    fn hlg_transfer_applies_inverse_oetf_and_ootf() {
        let op = CicpToScRgb::<U16>::new(CicpProfile::new(
            CicpColourPrimaries::Bt2020,
            CicpTransferCharacteristics::Hlg,
            CicpMatrixCoefficients::RgbIdentity,
            true,
        ));
        let region = make_region(1);
        let half = u16::MAX / 2;
        let input_data = [half, half, half];
        let input = Tile::<U16>::new(region, 3, &input_data);
        let mut output_data = [0.0_f32; 3];
        let mut output = TileMut::<F32>::new(region, 3, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        let signal = half as f32 / 65_535.0;
        let linear = hlg_inverse_oetf(signal);
        let scene_luminance = BT2020_LUMINANCE.iter().sum::<f32>() * linear;
        let expected_scale = (1000.0 / SDR_WHITE_NITS) * scene_luminance.powf(0.2);
        let expected = linear * expected_scale;
        let (expected_red, expected_green, expected_blue) =
            apply_matrix(BT2020_TO_BT709, expected, expected, expected);

        assert!((output_data[0] - expected_red).abs() < 2e-3);
        assert!((output_data[1] - expected_green).abs() < 2e-3);
        assert!((output_data[2] - expected_blue).abs() < 2e-3);
    }

    #[test]
    fn hlg_bt2020_saturated_white_matches_reference() {
        let actual = run_u16(
            CicpProfile::new(
                CicpColourPrimaries::Bt2020,
                CicpTransferCharacteristics::Hlg,
                CicpMatrixCoefficients::RgbIdentity,
                true,
            ),
            [u16::MAX, u16::MAX, u16::MAX],
        );

        let linear = hlg_inverse_oetf(1.0);
        let scene_luminance = BT2020_LUMINANCE.iter().sum::<f32>() * linear;
        let expected_scale = (1000.0 / SDR_WHITE_NITS) * scene_luminance.powf(0.2);
        let expected = linear * expected_scale;
        let expected = apply_matrix(BT2020_TO_BT709, expected, expected, expected);

        assert_triplet_close(actual, [expected.0, expected.1, expected.2], 2e-3);
    }

    #[test]
    fn hlg_ootf_zero_luminance_clamps_output_to_black() {
        let mut red = -0.25;
        let mut green = 0.0;
        let mut blue = 0.0;

        apply_hlg_ootf(
            &mut red,
            &mut green,
            &mut blue,
            BT2020_LUMINANCE,
            &build_hlg_ootf_lut(),
        );

        assert_eq!([red, green, blue], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn primaries_tables_cover_all_supported_profiles() {
        let supported = [
            (CicpColourPrimaries::Bt709, BT709_TO_BT709, BT709_LUMINANCE),
            (
                CicpColourPrimaries::Unspecified,
                BT709_TO_BT709,
                BT709_LUMINANCE,
            ),
            (
                CicpColourPrimaries::Bt470M,
                BT470M_TO_BT709,
                BT470M_LUMINANCE,
            ),
            (
                CicpColourPrimaries::Bt470Bg,
                BT470BG_TO_BT709,
                BT470BG_LUMINANCE,
            ),
            (CicpColourPrimaries::Bt601, BT601_TO_BT709, BT601_LUMINANCE),
            (
                CicpColourPrimaries::Smpte240,
                BT601_TO_BT709,
                BT601_LUMINANCE,
            ),
            (
                CicpColourPrimaries::GenericFilm,
                GENERIC_FILM_TO_BT709,
                GENERIC_FILM_LUMINANCE,
            ),
            (
                CicpColourPrimaries::Bt2020,
                BT2020_TO_BT709,
                BT2020_LUMINANCE,
            ),
            (
                CicpColourPrimaries::Smpte431,
                DCI_P3_TO_BT709,
                DCI_P3_LUMINANCE,
            ),
            (
                CicpColourPrimaries::Smpte432,
                DISPLAY_P3_TO_BT709,
                DISPLAY_P3_LUMINANCE,
            ),
            (
                CicpColourPrimaries::Ebu3213,
                EBU3213_TO_BT709,
                EBU3213_LUMINANCE,
            ),
        ];

        for (primaries, expected_matrix, expected_luminance) in supported {
            let (matrix, luminance) = primaries_tables(primaries);
            assert_eq!(matrix, expected_matrix);
            assert_eq!(luminance, expected_luminance);
        }
    }

    #[test]
    fn transfer_helpers_cover_boundary_branches() {
        assert_eq!(pq_eotf(0.0), 0.0);
        assert_eq!(hlg_inverse_oetf(0.0), 0.0);
        assert!((hlg_inverse_oetf(0.5) - (0.25 / 3.0)).abs() <= EPSILON);
        assert_eq!(bt709_inverse_oetf(-0.1), 0.0);
        assert!((bt709_inverse_oetf(0.01) - (0.01 / 4.5)).abs() <= EPSILON);
        assert_eq!(srgb_inverse_oetf(-0.1), 0.0);
        assert!((srgb_inverse_oetf(0.04045) - (0.04045 / 12.92)).abs() <= EPSILON);
        assert_eq!(cicp_transfer(CicpTransferCharacteristics::Log100, 0.0), 0.0);
        assert_eq!(
            cicp_transfer(CicpTransferCharacteristics::Log100Sqrt10, 0.0),
            0.0
        );
        assert!(cicp_transfer(CicpTransferCharacteristics::Iec61966, -0.25) < 0.0);
        assert!(cicp_transfer(CicpTransferCharacteristics::Bt1361, -0.25) < 0.0);
        assert_eq!(
            cicp_transfer(CicpTransferCharacteristics::Smpte240, 0.01),
            0.01 / 4.0
        );
        assert!(cicp_transfer(CicpTransferCharacteristics::Smpte240, 0.5) > 0.0);
        assert_eq!(
            cicp_transfer(CicpTransferCharacteristics::Smpte428, 0.0),
            0.0
        );
        assert!(cicp_transfer(CicpTransferCharacteristics::Bt470M, 0.5) > 0.0);
        assert!(cicp_transfer(CicpTransferCharacteristics::Bt470Bg, 0.5) > 0.0);
        assert_eq!(
            cicp_transfer(CicpTransferCharacteristics::Linear, 0.25),
            0.25
        );
        assert_eq!(
            cicp_transfer(CicpTransferCharacteristics::Unspecified, 0.25),
            0.25
        );
    }

    #[test]
    fn lut_builders_cover_endpoints() {
        let transfer_lut = build_transfer_lut::<U8>(CicpTransferCharacteristics::Linear);
        assert_eq!(transfer_lut.len(), 256);
        assert_eq!(transfer_lut[0], 0.0);
        assert_eq!(transfer_lut[255], 1.0);

        let ootf_lut = build_hlg_ootf_lut();
        assert_eq!(ootf_lut.len(), HLG_OOTF_LUT_SIZE);
        assert_eq!(ootf_lut[0], 0.0);
        assert!(ootf_lut[HLG_OOTF_LUT_SIZE - 1] > 0.0);
    }
}
