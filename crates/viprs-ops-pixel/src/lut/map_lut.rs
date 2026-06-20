use std::marker::PhantomData;

use viprs_core::{
    error::ViprsError,
    format::{BandFormat, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, OperationBridge, PixelLocalOp},
    shared_ops::sample_conv::FromF64,
};

/// Sample types that can index a `MapLut` using libvips-compatible cast semantics.
pub trait LutIndexSample: Copy {
    /// Returns or performs clipped lut index.
    fn clipped_lut_index(self, last_index: usize) -> usize;
}

/// Sample types that can represent libvips identity-table entries exactly.
pub trait LutIdentitySample: Copy {
    /// Creates this value from identity index.
    fn from_identity_index(index: usize) -> Self;
}

impl LutIdentitySample for u8 {
    #[inline(always)]
    fn from_identity_index(index: usize) -> Self {
        index as Self
    }
}

impl LutIdentitySample for u16 {
    #[inline(always)]
    fn from_identity_index(index: usize) -> Self {
        index as Self
    }
}

impl LutIdentitySample for i16 {
    #[inline(always)]
    fn from_identity_index(index: usize) -> Self {
        index as Self
    }
}

impl LutIdentitySample for u32 {
    #[inline(always)]
    fn from_identity_index(index: usize) -> Self {
        index as Self
    }
}

impl LutIdentitySample for i32 {
    #[inline(always)]
    fn from_identity_index(index: usize) -> Self {
        index as Self
    }
}

impl LutIdentitySample for f32 {
    #[inline(always)]
    fn from_identity_index(index: usize) -> Self {
        index as Self
    }
}

impl LutIdentitySample for f64 {
    #[inline(always)]
    fn from_identity_index(index: usize) -> Self {
        index as Self
    }
}

impl LutIndexSample for u8 {
    #[inline(always)]
    fn clipped_lut_index(self, last_index: usize) -> usize {
        usize::from(self).min(last_index)
    }
}

impl LutIndexSample for u16 {
    #[inline(always)]
    fn clipped_lut_index(self, last_index: usize) -> usize {
        usize::from(self).min(last_index)
    }
}

impl LutIndexSample for u32 {
    #[inline(always)]
    fn clipped_lut_index(self, last_index: usize) -> usize {
        (self as usize).min(last_index)
    }
}

impl LutIndexSample for i8 {
    #[inline(always)]
    fn clipped_lut_index(self, last_index: usize) -> usize {
        if self <= 0 {
            0
        } else {
            (self as usize).min(last_index)
        }
    }
}

impl LutIndexSample for i16 {
    #[inline(always)]
    fn clipped_lut_index(self, last_index: usize) -> usize {
        if self <= 0 {
            0
        } else {
            (self as usize).min(last_index)
        }
    }
}

impl LutIndexSample for i32 {
    #[inline(always)]
    fn clipped_lut_index(self, last_index: usize) -> usize {
        if self <= 0 {
            0
        } else {
            (self as usize).min(last_index)
        }
    }
}

impl LutIndexSample for f32 {
    #[inline(always)]
    fn clipped_lut_index(self, last_index: usize) -> usize {
        if !self.is_finite() || self <= 0.0 {
            0
        } else if self >= last_index as Self {
            last_index
        } else {
            self.trunc() as usize
        }
    }
}

impl LutIndexSample for f64 {
    #[inline(always)]
    fn clipped_lut_index(self, last_index: usize) -> usize {
        if !self.is_finite() || self <= 0.0 {
            0
        } else if self >= last_index as Self {
            last_index
        } else {
            self.trunc() as usize
        }
    }
}

/// Maps samples through a LUT, matching libvips `maplut` single-band,
/// multiband, selected-band, and index-casting semantics.
///
/// The LUT is stored interleaved as `[index0_band0, index0_band1, …, index1_band0, …]`.
pub struct MapLut<I: BandFormat = U8, O: BandFormat = U8> {
    lut: Box<[O::Sample]>,
    lut_bands: u32,
    selected_band: Option<u32>,
    _input: PhantomData<I>,
}

impl<I: BandFormat, O: BandFormat> MapLut<I, O> {
    /// Creates this value from table.
    pub fn from_table(lut: Vec<O::Sample>, lut_bands: u32) -> Result<Self, ViprsError> {
        Self::from_table_with_band(lut, lut_bands, None)
    }

    /// Creates this value from table with band.
    pub fn from_table_with_band(
        lut: Vec<O::Sample>,
        lut_bands: u32,
        selected_band: Option<u32>,
    ) -> Result<Self, ViprsError> {
        if lut_bands == 0 {
            return Err(ViprsError::Scheduler(
                "MapLut lut_bands must be greater than zero".to_owned(),
            ));
        }
        if lut.is_empty() {
            return Err(ViprsError::Scheduler(
                "MapLut LUT must contain at least one sample".to_owned(),
            ));
        }
        if !lut.len().is_multiple_of(lut_bands as usize) {
            return Err(ViprsError::Scheduler(format!(
                "MapLut LUT length {} is not divisible by lut_bands {lut_bands}",
                lut.len()
            )));
        }
        if selected_band.is_some() && lut_bands != 1 {
            return Err(ViprsError::Scheduler(
                "MapLut selected_band requires a single-band LUT".to_owned(),
            ));
        }

        Ok(Self {
            lut: lut.into_boxed_slice(),
            lut_bands,
            selected_band,
            _input: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs lut bands.
    pub const fn lut_bands(&self) -> u32 {
        self.lut_bands
    }

    #[must_use]
    /// Returns or performs selected band.
    pub const fn selected_band(&self) -> Option<u32> {
        self.selected_band
    }

    #[must_use]
    /// Returns or performs lut size.
    pub fn lut_size(&self) -> usize {
        self.lut.len() / self.lut_bands as usize
    }

    /// Returns or performs output bands for input.
    pub fn output_bands_for_input(&self, input_bands: u32) -> Result<u32, ViprsError> {
        if input_bands == 0 {
            return Err(ViprsError::Scheduler(
                "MapLut input_bands must be greater than zero".to_owned(),
            ));
        }

        if let Some(selected_band) = self.selected_band
            && selected_band >= input_bands
        {
            return Err(ViprsError::Scheduler(format!(
                "MapLut selected_band {selected_band} is out of range for input_bands {input_bands}"
            )));
        }

        if self.lut_bands == 1 {
            return Ok(input_bands);
        }
        if input_bands == 1 {
            return Ok(self.lut_bands);
        }
        if input_bands == self.lut_bands {
            return Ok(input_bands);
        }

        Err(ViprsError::Scheduler(format!(
            "MapLut LUT bands {} are incompatible with input_bands {input_bands}",
            self.lut_bands
        )))
    }
}

impl<I, O> MapLut<I, O>
where
    I: BandFormat,
    O: BandFormat,
    I::Sample: LutIndexSample + bytemuck::Pod,
    O::Sample: FromF64 + LutIdentitySample + bytemuck::Pod,
{
    /// Returns or performs into bridge.
    pub fn into_bridge(self, input_bands: u32) -> Result<OperationBridge<Self>, ViprsError> {
        let output_bands = self.output_bands_for_input(input_bands)?;
        Ok(OperationBridge::with_dynamic_bands_pixel_local(
            self,
            input_bands,
            output_bands,
        ))
    }

    #[inline(always)]
    fn clipped_index(&self, sample: I::Sample) -> usize {
        sample.clipped_lut_index(self.lut_size() - 1)
    }

    #[inline(always)]
    fn identity_sample(index: usize) -> O::Sample {
        O::Sample::from_identity_index(index)
    }
}

impl MapLut<U8, U8> {
    /// Construct a single-band 256-entry U8→U8 LUT.
    #[must_use]
    pub fn new(lut: [u8; 256]) -> Self {
        Self {
            lut: Vec::from(lut).into_boxed_slice(),
            lut_bands: 1,
            selected_band: None,
            _input: PhantomData,
        }
    }

    /// Construct a selected-band 256-entry U8→U8 LUT.
    #[must_use]
    pub fn new_with_selected_band(lut: [u8; 256], selected_band: u32) -> Self {
        Self {
            lut: Vec::from(lut).into_boxed_slice(),
            lut_bands: 1,
            selected_band: Some(selected_band),
            _input: PhantomData,
        }
    }

    /// Construct an identity `MapLut` (`lut[i] = i` for all entries).
    #[must_use]
    pub fn identity() -> Self {
        let mut lut = [0u8; 256];
        for (index, entry) in lut.iter_mut().enumerate() {
            *entry = index as u8;
        }
        Self::new(lut)
    }
}

impl<I, O> Op for MapLut<I, O>
where
    I: BandFormat,
    O: BandFormat,
    I::Sample: LutIndexSample,
    O::Sample: FromF64 + LutIdentitySample,
{
    type Input = I;
    type Output = O;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<I>, output: &mut TileMut<O>) {
        let expected_output_bands_result = self.output_bands_for_input(input.bands);
        debug_assert!(
            expected_output_bands_result.is_ok(),
            "MapLut configuration must be valid before processing"
        );
        let expected_output_bands = if let Ok(bands) = expected_output_bands_result {
            bands
        } else {
            output.bands
        };

        debug_assert_eq!(input.region, output.region, "MapLut regions must match");
        debug_assert_eq!(
            output.bands, expected_output_bands,
            "MapLut output tile has wrong band count"
        );

        let input_bands = input.bands as usize;
        let output_bands = output.bands as usize;
        let lut_bands = self.lut_bands as usize;
        let pixel_count = input.region.pixel_count();

        if lut_bands == 1 {
            if let Some(selected_band) = self.selected_band.map(|band| band as usize) {
                for pixel in 0..pixel_count {
                    let src_base = pixel * input_bands;
                    let dst_base = pixel * output_bands;
                    for band in 0..input_bands {
                        let lut_index = self.clipped_index(input.data[src_base + band]);
                        output.data[dst_base + band] = if band == selected_band {
                            self.lut[lut_index]
                        } else {
                            Self::identity_sample(lut_index)
                        };
                    }
                }
            } else {
                for (src, dst) in input.data.iter().zip(output.data.iter_mut()) {
                    *dst = self.lut[self.clipped_index(*src)];
                }
            }
            return;
        }

        if input_bands == 1 {
            for pixel in 0..pixel_count {
                let lut_base = self.clipped_index(input.data[pixel]) * lut_bands;
                let dst_base = pixel * output_bands;
                output.data[dst_base..dst_base + lut_bands]
                    .copy_from_slice(&self.lut[lut_base..lut_base + lut_bands]);
            }
            return;
        }

        for pixel in 0..pixel_count {
            let src_base = pixel * input_bands;
            let dst_base = pixel * output_bands;
            for band in 0..input_bands {
                let lut_index = self.clipped_index(input.data[src_base + band]);
                output.data[dst_base + band] = self.lut[lut_index * lut_bands + band];
            }
        }
    }
}

impl<I, O> PixelLocalOp for MapLut<I, O>
where
    I: BandFormat,
    O: BandFormat,
    I::Sample: LutIndexSample,
    O::Sample: FromF64 + LutIdentitySample,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, F64, I16, U8, U16},
        image::Region,
        op::DynOperation,
    };

    prop_compose! {
        fn multiband_u8_pixels()(bands in 1u32..=4, pixel_count in 1usize..=32)
            (bands in Just(bands), pixels in proptest::collection::vec(0u8..=255u8, pixel_count * bands as usize))
            -> (u32, Vec<u8>) {
                (bands, pixels)
            }
    }

    fn make_region(pixel_count: usize) -> Region {
        Region::new(0, 0, pixel_count as u32, 1)
    }

    fn invert_lut() -> [u8; 256] {
        let mut lut = [0u8; 256];
        for (index, entry) in lut.iter_mut().enumerate() {
            *entry = 255 - index as u8;
        }
        lut
    }

    fn multiband_lut() -> Vec<u8> {
        vec![
            0, 10, 20, // index 0
            1, 11, 21, // index 1
            2, 12, 22, // index 2
            3, 13, 23, // index 3
        ]
    }

    fn identity_u8_table() -> Vec<u8> {
        (0u8..=u8::MAX).collect()
    }

    fn run_maplut<I, O>(
        op: &MapLut<I, O>,
        input_bands: u32,
        output_bands: u32,
        input_data: &[I::Sample],
    ) -> Vec<O::Sample>
    where
        I: BandFormat,
        O: BandFormat,
        I::Sample: LutIndexSample,
        O::Sample: FromF64 + LutIdentitySample + Copy,
    {
        let pixel_count = input_data.len() / input_bands as usize;
        let region = make_region(pixel_count);
        let input = Tile::<I>::new(region, input_bands, input_data);
        let mut output_data = vec![O::Sample::from_f64(0.0); pixel_count * output_bands as usize];
        let mut output = TileMut::<O>::new(region, output_bands, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_maplut_bridge(
        bridge: &dyn DynOperation,
        input_bands: u32,
        input_data: &[u8],
    ) -> Vec<u8> {
        let pixel_count = input_data.len() / input_bands as usize;
        let region = make_region(pixel_count);
        let mut output = vec![0u8; pixel_count * bridge.bands() as usize];
        let mut state = bridge.dyn_start();
        bridge.dyn_process_region(state.as_mut(), input_data, &mut output, region, region);
        output
    }

    #[test]
    fn identity_lut_is_no_op() {
        let op = MapLut::identity();
        let input = vec![0u8, 64, 128, 255];
        assert_eq!(run_maplut(&op, 1, 1, &input), input);
    }

    #[test]
    fn invert_lut_inverts_pixels() {
        let op = MapLut::new(invert_lut());
        assert_eq!(
            run_maplut(&op, 1, 1, &[0u8, 64, 128, 255]),
            vec![255, 191, 127, 0]
        );
    }

    #[test]
    fn single_band_lut_maps_every_input_band() {
        let op = MapLut::new(invert_lut());
        let input = [0u8, 10, 20, 30, 40, 50];
        assert_eq!(
            run_maplut(&op, 3, 3, &input),
            vec![255, 245, 235, 225, 215, 205]
        );
    }

    #[test]
    fn selected_band_only_maps_requested_band() {
        let op = MapLut::new_with_selected_band(invert_lut(), 1);
        let input = [10u8, 20, 30, 40, 50, 60];
        assert_eq!(
            run_maplut(&op, 3, 3, &input),
            vec![10, 235, 30, 40, 205, 60]
        );
    }

    #[test]
    fn matching_multiband_lut_maps_each_band_independently() {
        let op = MapLut::<U8, U8>::from_table(multiband_lut(), 3).unwrap();
        let input = [0u8, 1, 2, 3, 2, 1];
        assert_eq!(run_maplut(&op, 3, 3, &input), vec![0, 11, 22, 3, 12, 21]);
    }

    #[test]
    fn multiband_lut_expands_single_band_input() {
        let op = MapLut::<U8, U8>::from_table(multiband_lut(), 3).unwrap();
        let input = [0u8, 2, 3];
        assert_eq!(
            run_maplut(&op, 1, 3, &input),
            vec![0, 10, 20, 2, 12, 22, 3, 13, 23]
        );
    }

    #[test]
    fn shorter_lut_clips_u16_indices_to_last_entry() {
        let op = MapLut::<U16, U16>::from_table(vec![10u16, 20, 30], 1).unwrap();
        let input = [0u16, 1, 2, 3, u16::MAX];
        assert_eq!(run_maplut(&op, 1, 1, &input), vec![10, 20, 30, 30, 30]);
    }

    #[test]
    fn invalid_selected_band_is_rejected() {
        let op = MapLut::new_with_selected_band(invert_lut(), 3);
        let err = op.output_bands_for_input(3).unwrap_err();
        assert!(err.to_string().contains("selected_band 3 is out of range"));
    }

    #[test]
    fn invalid_table_shape_and_configuration_are_rejected() {
        let zero_bands = MapLut::<U8, U8>::from_table(vec![0u8], 0)
            .err()
            .expect("zero lut_bands must be rejected");
        assert!(
            zero_bands
                .to_string()
                .contains("lut_bands must be greater than zero")
        );

        let empty_lut = MapLut::<U8, U8>::from_table(Vec::new(), 1)
            .err()
            .expect("empty LUT must be rejected");
        assert!(
            empty_lut
                .to_string()
                .contains("must contain at least one sample")
        );

        let invalid_shape = MapLut::<U8, U8>::from_table(vec![1u8, 2, 3, 4, 5], 2)
            .err()
            .expect("non-divisible LUT shape must be rejected");
        assert!(
            invalid_shape
                .to_string()
                .contains("is not divisible by lut_bands 2")
        );

        let invalid_selected_band =
            MapLut::<U8, U8>::from_table_with_band(vec![1u8, 2, 3, 4], 2, Some(0))
                .err()
                .expect("selected_band with multiband LUT must be rejected");
        assert!(
            invalid_selected_band
                .to_string()
                .contains("selected_band requires a single-band LUT")
        );
    }

    #[test]
    fn zero_input_bands_and_incompatible_multiband_inputs_are_rejected() {
        let single_band = MapLut::<U8, U8>::from_table(vec![0u8, 1, 2], 1).unwrap();
        let zero_input = single_band.output_bands_for_input(0).unwrap_err();
        assert!(
            zero_input
                .to_string()
                .contains("input_bands must be greater than zero")
        );

        let multiband = MapLut::<U8, U8>::from_table(multiband_lut(), 3).unwrap();
        let incompatible = multiband.output_bands_for_input(2).unwrap_err();
        assert!(
            incompatible
                .to_string()
                .contains("LUT bands 3 are incompatible with input_bands 2")
        );
    }

    #[test]
    fn bridge_expands_single_band_input_with_dynamic_output_bands() {
        let bridge = MapLut::<U8, U8>::from_table(multiband_lut(), 3)
            .unwrap()
            .into_bridge(1)
            .unwrap();
        assert_eq!(bridge.bands(), 3);
        assert!(bridge.is_pixel_local());
        assert_eq!(
            run_maplut_bridge(&bridge, 1, &[0u8, 2, 3]),
            vec![0, 10, 20, 2, 12, 22, 3, 13, 23]
        );
    }

    #[test]
    fn non_u8_index_families_clip_and_truncate_exactly() {
        let i16_op = MapLut::<I16, U8>::from_table(vec![10u8, 20, 30], 1).unwrap();
        assert_eq!(
            run_maplut(&i16_op, 1, 1, &[i16::MIN, -5, 0, 1, 2, 3, i16::MAX]),
            vec![10, 10, 10, 20, 30, 30, 30]
        );

        let u32_op =
            MapLut::<viprs_core::format::U32, U8>::from_table(vec![10u8, 20, 30], 1).unwrap();
        assert_eq!(
            run_maplut(&u32_op, 1, 1, &[0u32, 1, 2, 3, u32::MAX]),
            vec![10, 20, 30, 30, 30]
        );

        let i32_op =
            MapLut::<viprs_core::format::I32, U8>::from_table(vec![10u8, 20, 30], 1).unwrap();
        assert_eq!(
            run_maplut(&i32_op, 1, 1, &[i32::MIN, -1, 0, 1, 2, 3, i32::MAX]),
            vec![10, 10, 10, 20, 30, 30, 30]
        );

        let f64_op = MapLut::<F64, U8>::from_table(vec![10u8, 20, 30], 1).unwrap();
        assert_eq!(
            run_maplut(
                &f64_op,
                1,
                1,
                &[
                    f64::NEG_INFINITY,
                    f64::NAN,
                    -1.5,
                    0.0,
                    1.9,
                    2.0,
                    99.0,
                    f64::INFINITY,
                ],
            ),
            vec![10, 10, 10, 10, 20, 30, 30, 10]
        );
    }

    proptest! {
        #[test]
        fn negative_signed_indices_map_to_first_entry(
            negative in i16::MIN..=-1,
            second in any::<u8>(),
            third in any::<u8>(),
        ) {
            let op = MapLut::<I16, U8>::from_table(vec![10u8, second, third], 1).unwrap();
            prop_assert_eq!(run_maplut(&op, 1, 1, &[negative]), vec![10]);
        }

        #[test]
        fn float_zero_maps_to_first_entry(index in Just(0.0_f32)) {
            let op = MapLut::<F32, U8>::from_table(vec![10u8, 20, 30], 1).unwrap();
            prop_assert_eq!(run_maplut(&op, 1, 1, &[index]), vec![10]);
        }

        #[test]
        fn float_255_point_9_truncates_for_u8_lut(index in Just(255.9_f32)) {
            let op = MapLut::<F32, U8>::from_table(identity_u8_table(), 1).unwrap();
            prop_assert_eq!(run_maplut(&op, 1, 1, &[index]), vec![255]);
        }

        #[test]
        fn out_of_bounds_indices_clip_to_last_entry(index in 3.0_f64..10_000.0_f64) {
            let op = MapLut::<F64, U8>::from_table(vec![10u8, 20, 30], 1).unwrap();
            prop_assert_eq!(run_maplut(&op, 1, 1, &[index]), vec![30]);
        }

        #[test]
        fn identity_lut_preserves_multiband_pixels((bands, pixels) in multiband_u8_pixels()) {
            let op = MapLut::identity();
            prop_assert_eq!(run_maplut(&op, bands, bands, &pixels), pixels);
        }

        #[test]
        fn inversion_lut_matches_255_minus_input((bands, pixels) in multiband_u8_pixels()) {
            let op = MapLut::new(invert_lut());
            let expected = pixels.iter().map(|sample| 255u8.wrapping_sub(*sample)).collect::<Vec<_>>();
            prop_assert_eq!(run_maplut(&op, bands, bands, &pixels), expected);
        }

        #[test]
        fn multiband_lut_expansion_matches_expected_rgb(pixels in proptest::collection::vec(0u8..=3u8, 1..=32)) {
            let op = MapLut::<U8, U8>::from_table(multiband_lut(), 3).unwrap();
            let mut expected = Vec::with_capacity(pixels.len() * 3);
            for sample in &pixels {
                expected.extend_from_slice(&[*sample, sample.saturating_add(10), sample.saturating_add(20)]);
            }
            prop_assert_eq!(run_maplut(&op, 1, 3, &pixels), expected);
        }
    }
}
