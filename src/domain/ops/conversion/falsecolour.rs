use std::marker::PhantomData;

use crate::domain::{
    format::{BandFormat, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, OperationBridge, PixelLocalOp},
};

/// Constant value for falsecolour output bands.
pub const FALSECOLOUR_OUTPUT_BANDS: usize = 3;

/// PET false-colour scale used by libvips `falsecolour`.
pub const FALSECOLOUR_PET_LUT: [[u8; FALSECOLOUR_OUTPUT_BANDS]; 256] = [
    [12, 0, 25],
    [17, 0, 34],
    [20, 0, 41],
    [22, 0, 45],
    [23, 0, 47],
    [27, 0, 55],
    [12, 0, 25],
    [5, 0, 11],
    [5, 0, 11],
    [5, 0, 11],
    [1, 0, 4],
    [1, 0, 4],
    [6, 0, 13],
    [15, 0, 30],
    [19, 0, 40],
    [23, 0, 48],
    [28, 0, 57],
    [36, 0, 74],
    [42, 0, 84],
    [46, 0, 93],
    [51, 0, 102],
    [59, 0, 118],
    [65, 0, 130],
    [69, 0, 138],
    [72, 0, 146],
    [81, 0, 163],
    [47, 0, 95],
    [12, 0, 28],
    [64, 0, 144],
    [61, 0, 146],
    [55, 0, 140],
    [52, 0, 137],
    [47, 0, 132],
    [43, 0, 128],
    [38, 0, 123],
    [30, 0, 115],
    [26, 0, 111],
    [23, 0, 108],
    [17, 0, 102],
    [9, 0, 94],
    [6, 0, 91],
    [2, 0, 87],
    [0, 0, 88],
    [0, 0, 100],
    [0, 0, 104],
    [0, 0, 108],
    [0, 0, 113],
    [0, 0, 121],
    [0, 0, 125],
    [0, 0, 129],
    [0, 0, 133],
    [0, 0, 141],
    [0, 0, 146],
    [0, 0, 150],
    [0, 0, 155],
    [0, 0, 162],
    [0, 0, 167],
    [0, 0, 173],
    [0, 0, 180],
    [0, 0, 188],
    [0, 0, 193],
    [0, 0, 197],
    [0, 0, 201],
    [0, 0, 209],
    [0, 0, 214],
    [0, 0, 218],
    [0, 0, 222],
    [0, 0, 230],
    [0, 0, 235],
    [0, 0, 239],
    [0, 0, 243],
    [0, 0, 247],
    [0, 4, 251],
    [0, 10, 255],
    [0, 14, 255],
    [0, 18, 255],
    [0, 24, 255],
    [0, 31, 255],
    [0, 36, 255],
    [0, 39, 255],
    [0, 45, 255],
    [0, 53, 255],
    [0, 56, 255],
    [0, 60, 255],
    [0, 66, 255],
    [0, 74, 255],
    [0, 77, 255],
    [0, 81, 255],
    [0, 88, 251],
    [0, 99, 239],
    [0, 104, 234],
    [0, 108, 230],
    [0, 113, 225],
    [0, 120, 218],
    [0, 125, 213],
    [0, 128, 210],
    [0, 133, 205],
    [0, 141, 197],
    [0, 145, 193],
    [0, 150, 188],
    [0, 154, 184],
    [0, 162, 176],
    [0, 167, 172],
    [0, 172, 170],
    [0, 180, 170],
    [0, 188, 170],
    [0, 193, 170],
    [0, 197, 170],
    [0, 201, 170],
    [0, 205, 170],
    [0, 211, 170],
    [0, 218, 170],
    [0, 222, 170],
    [0, 226, 170],
    [0, 232, 170],
    [0, 239, 170],
    [0, 243, 170],
    [0, 247, 170],
    [0, 251, 161],
    [0, 255, 147],
    [0, 255, 139],
    [0, 255, 131],
    [0, 255, 120],
    [0, 255, 105],
    [0, 255, 97],
    [0, 255, 89],
    [0, 255, 78],
    [0, 255, 63],
    [0, 255, 55],
    [0, 255, 47],
    [0, 255, 37],
    [0, 255, 21],
    [0, 255, 13],
    [0, 255, 5],
    [2, 255, 2],
    [13, 255, 13],
    [18, 255, 18],
    [23, 255, 23],
    [27, 255, 27],
    [35, 255, 35],
    [40, 255, 40],
    [43, 255, 43],
    [48, 255, 48],
    [55, 255, 55],
    [60, 255, 60],
    [64, 255, 64],
    [69, 255, 69],
    [72, 255, 72],
    [79, 255, 79],
    [90, 255, 82],
    [106, 255, 74],
    [113, 255, 70],
    [126, 255, 63],
    [140, 255, 56],
    [147, 255, 53],
    [155, 255, 48],
    [168, 255, 42],
    [181, 255, 36],
    [189, 255, 31],
    [197, 255, 27],
    [209, 255, 21],
    [224, 255, 14],
    [231, 255, 10],
    [239, 255, 7],
    [247, 251, 3],
    [255, 243, 0],
    [255, 239, 0],
    [255, 235, 0],
    [255, 230, 0],
    [255, 222, 0],
    [255, 218, 0],
    [255, 214, 0],
    [255, 209, 0],
    [255, 201, 0],
    [255, 197, 0],
    [255, 193, 0],
    [255, 188, 0],
    [255, 180, 0],
    [255, 176, 0],
    [255, 172, 0],
    [255, 167, 0],
    [255, 156, 0],
    [255, 150, 0],
    [255, 146, 0],
    [255, 142, 0],
    [255, 138, 0],
    [255, 131, 0],
    [255, 125, 0],
    [255, 121, 0],
    [255, 117, 0],
    [255, 110, 0],
    [255, 104, 0],
    [255, 100, 0],
    [255, 96, 0],
    [255, 90, 0],
    [255, 83, 0],
    [255, 78, 0],
    [255, 75, 0],
    [255, 71, 0],
    [255, 67, 0],
    [255, 65, 0],
    [255, 63, 0],
    [255, 59, 0],
    [255, 54, 0],
    [255, 52, 0],
    [255, 50, 0],
    [255, 46, 0],
    [255, 41, 0],
    [255, 39, 0],
    [255, 36, 0],
    [255, 32, 0],
    [255, 25, 0],
    [255, 22, 0],
    [255, 20, 0],
    [255, 17, 0],
    [255, 13, 0],
    [255, 10, 0],
    [255, 7, 0],
    [255, 4, 0],
    [255, 0, 0],
    [252, 0, 0],
    [251, 0, 0],
    [249, 0, 0],
    [248, 0, 0],
    [244, 0, 0],
    [242, 0, 0],
    [240, 0, 0],
    [237, 0, 0],
    [234, 0, 0],
    [231, 0, 0],
    [229, 0, 0],
    [228, 0, 0],
    [225, 0, 0],
    [222, 0, 0],
    [221, 0, 0],
    [219, 0, 0],
    [216, 0, 0],
    [213, 0, 0],
    [212, 0, 0],
    [210, 0, 0],
    [207, 0, 0],
    [204, 0, 0],
    [201, 0, 0],
    [199, 0, 0],
    [196, 0, 0],
    [193, 0, 0],
    [192, 0, 0],
    [190, 0, 0],
    [188, 0, 0],
    [184, 0, 0],
    [183, 0, 0],
    [181, 0, 0],
    [179, 0, 0],
    [175, 0, 0],
    [174, 0, 0],
    [174, 0, 0],
];

/// Sample-level cast to the U8 LUT index used by libvips falsecolour.
pub trait FalsecolourSample: Copy {
    /// Converts this value to falsecolour index.
    fn to_falsecolour_index(self) -> u8;
}

impl FalsecolourSample for u8 {
    #[inline(always)]
    fn to_falsecolour_index(self) -> u8 {
        self
    }
}

impl FalsecolourSample for u16 {
    #[inline(always)]
    fn to_falsecolour_index(self) -> u8 {
        self.min(Self::from(u8::MAX)) as u8
    }
}

impl FalsecolourSample for i16 {
    #[inline(always)]
    fn to_falsecolour_index(self) -> u8 {
        self.clamp(0, Self::from(u8::MAX)) as u8
    }
}

impl FalsecolourSample for u32 {
    #[inline(always)]
    fn to_falsecolour_index(self) -> u8 {
        self.min(Self::from(u8::MAX)) as u8
    }
}

impl FalsecolourSample for i32 {
    #[inline(always)]
    fn to_falsecolour_index(self) -> u8 {
        self.clamp(0, Self::from(u8::MAX)) as u8
    }
}

impl FalsecolourSample for f32 {
    #[inline(always)]
    fn to_falsecolour_index(self) -> u8 {
        f64_to_u8_index(f64::from(self))
    }
}

impl FalsecolourSample for f64 {
    #[inline(always)]
    fn to_falsecolour_index(self) -> u8 {
        f64_to_u8_index(self)
    }
}

#[inline(always)]
fn f64_to_u8_index(value: f64) -> u8 {
    if value.is_nan() {
        0
    } else {
        value.round().clamp(0.0, f64::from(u8::MAX)) as u8
    }
}

/// Map the first input band through libvips' 256-entry false-colour LUT.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::conversion::falsecolour::FalsecolourOp;
///
/// let op = FalsecolourOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct FalsecolourOp<F: BandFormat> {
    _format: PhantomData<F>,
}

impl<F: BandFormat> FalsecolourOp<F>
where
    F::Sample: FalsecolourSample,
{
    #[must_use]
    /// Creates a new `FalsecolourOp`.
    pub const fn new() -> Self {
        Self {
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs into bridge.
    pub fn into_bridge(self, input_bands: u32) -> OperationBridge<Self> {
        OperationBridge::new_pixel_local(self, input_bands)
    }
}

impl<F: BandFormat> Default for FalsecolourOp<F>
where
    F::Sample: FalsecolourSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Op for FalsecolourOp<F>
where
    F: BandFormat,
    F::Sample: FalsecolourSample,
{
    type Input = F;
    type Output = U8;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(FALSECOLOUR_OUTPUT_BANDS);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<U8>) {
        debug_assert!(
            input.bands >= 1,
            "FalsecolourOp requires at least one input band"
        );
        debug_assert_eq!(
            output.bands as usize, FALSECOLOUR_OUTPUT_BANDS,
            "FalsecolourOp always writes RGB output"
        );

        let input_bands = input.bands as usize;
        let pixel_count = input.region.pixel_count();

        for pixel in 0..pixel_count {
            let src = pixel * input_bands;
            let dst = pixel * FALSECOLOUR_OUTPUT_BANDS;
            let rgb = FALSECOLOUR_PET_LUT[input.data[src].to_falsecolour_index() as usize];
            output.data[dst..dst + FALSECOLOUR_OUTPUT_BANDS].copy_from_slice(&rgb);
        }
    }
}

impl<F> PixelLocalOp for FalsecolourOp<F>
where
    F: BandFormat,
    F::Sample: FalsecolourSample,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            format::{F32, I16, U8, U16},
            image::{Region, Tile, TileMut},
            op::DynOperation,
        },
        ports::scheduler::TileScheduler,
    };
    use proptest::prelude::*;

    fn run_falsecolour_u8(input_data: &[u8], input_bands: u32) -> Vec<u8> {
        let pixel_count = input_data.len() / input_bands as usize;
        let op = FalsecolourOp::<U8>::new();
        let region = Region::new(0, 0, pixel_count as u32, 1);
        let input = Tile::<U8>::new(region, input_bands, input_data);
        let mut output_data = vec![0u8; pixel_count * FALSECOLOUR_OUTPUT_BANDS];
        let mut output =
            TileMut::<U8>::new(region, FALSECOLOUR_OUTPUT_BANDS as u32, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn maps_first_and_last_lut_entries() {
        let output = run_falsecolour_u8(&[0, 255], 1);
        assert_eq!(&output[0..3], &FALSECOLOUR_PET_LUT[0]);
        assert_eq!(&output[3..6], &FALSECOLOUR_PET_LUT[255]);
    }

    #[test]
    fn uses_first_band_only() {
        let output = run_falsecolour_u8(&[10, 200, 11, 201], 2);
        assert_eq!(&output[0..3], &FALSECOLOUR_PET_LUT[10]);
        assert_eq!(&output[3..6], &FALSECOLOUR_PET_LUT[11]);
    }

    #[test]
    fn non_u8_inputs_cast_to_lut_index() {
        assert_eq!(u16::MAX.to_falsecolour_index(), 255);
        assert_eq!((-1i16).to_falsecolour_index(), 0);
        assert_eq!(128.4f32.to_falsecolour_index(), 128);
        assert_eq!(128.5f64.to_falsecolour_index(), 129);
    }

    #[test]
    fn bridge_reports_three_output_bands() {
        let bridge = FalsecolourOp::<U16>::new().into_bridge(4);
        assert_eq!(
            bridge.input_format(),
            crate::domain::format::BandFormatId::U16
        );
        assert_eq!(
            bridge.output_format(),
            crate::domain::format::BandFormatId::U8
        );
        assert_eq!(bridge.bands(), 3);
    }

    #[test]
    fn pipeline_maps_grayscale_to_rgb() {
        let source = MemorySource::<U8>::new(2, 1, 1, vec![0, 255]).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(FalsecolourOp::<U8>::new().into_bridge(1)))
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(1)
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();

        assert_eq!(pipeline.output_bands, 3);
        assert_eq!(
            sink.into_buffer(),
            [FALSECOLOUR_PET_LUT[0], FALSECOLOUR_PET_LUT[255]].concat()
        );
    }

    #[test]
    fn f32_boundary_values_map_to_expected_lut_entries() {
        let op = FalsecolourOp::<F32>::new();
        let region = Region::new(0, 0, 2, 1);
        let input_data = [-1.0f32, 300.0];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output_data = vec![0u8; 6];
        let mut output = TileMut::<U8>::new(region, 3, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(&output_data[0..3], &FALSECOLOUR_PET_LUT[0]);
        assert_eq!(&output_data[3..6], &FALSECOLOUR_PET_LUT[255]);
    }

    proptest! {
        #[test]
        fn u8_pixels_index_the_lut(samples in proptest::collection::vec(any::<u8>(), 1..=128)) {
            let output = run_falsecolour_u8(&samples, 1);
            for (pixel, sample) in samples.iter().enumerate() {
                let dst = pixel * FALSECOLOUR_OUTPUT_BANDS;
                prop_assert_eq!(&output[dst..dst + FALSECOLOUR_OUTPUT_BANDS], &FALSECOLOUR_PET_LUT[*sample as usize]);
            }
        }

        #[test]
        fn required_region_is_identity(width in 1u32..=16, height in 1u32..=16) {
            let op = FalsecolourOp::<I16>::new();
            let region = Region::new(3, 4, width, height);
            prop_assert_eq!(op.required_input_region(&region), region);
        }
    }
}
