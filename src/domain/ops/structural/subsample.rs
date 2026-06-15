use crate::{
    domain::op::{NodeSpec, Op, OperationBridge},
    domain::{
        error::BuildError,
        format::BandFormat,
        image::{DemandHint, Region, Tile, TileMut},
    },
};
use std::marker::PhantomData;

fn validate_subsample_factors(xfac: u32, yfac: u32) -> Result<(), BuildError> {
    if xfac == 0 || yfac == 0 {
        return Err(BuildError::SourceHint {
            context: "subsample",
            message: "xfac and yfac must be >= 1".to_string(),
        });
    }

    Ok(())
}

/// Integer decimation by taking every `xfac`-th / `yfac`-th pixel.
pub struct Subsample<F: BandFormat> {
    xfac: u32,
    yfac: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> Subsample<F> {
    /// Creates a new `Subsample`.
    pub fn new(xfac: u32, yfac: u32) -> Result<Self, BuildError> {
        validate_subsample_factors(xfac, yfac)?;
        Ok(Self {
            xfac,
            yfac,
            _format: PhantomData,
        })
    }
}

impl<F: BandFormat> Op for Subsample<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        if output.is_empty() {
            return Region::new(
                output.x * self.xfac as i32,
                output.y * self.yfac as i32,
                0,
                0,
            );
        }

        Region::new(
            output.x * self.xfac as i32,
            output.y * self.yfac as i32,
            (output.width * self.xfac).saturating_sub(self.xfac - 1),
            (output.height * self.yfac).saturating_sub(self.yfac - 1),
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        if tile_w == 0 || tile_h == 0 {
            return NodeSpec {
                input_tile_w: 0,
                input_tile_h: 0,
                output_tile_w: tile_w,
                output_tile_h: tile_h,
                coordinate_driven_source: None,
            };
        }

        NodeSpec {
            input_tile_w: (tile_w * self.xfac).saturating_sub(self.xfac - 1),
            input_tile_h: (tile_h * self.yfac).saturating_sub(self.yfac - 1),
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let bands = input.bands as usize;
        let input_width = input.region.width as usize;
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let xfac = self.xfac as usize;
        let yfac = self.yfac as usize;

        for row in 0..out_h {
            let src_row = row * yfac;
            for col in 0..out_w {
                let src_col = col * xfac;
                let src = (src_row * input_width + src_col) * bands;
                let dst = (row * out_w + col) * bands;
                output.data[dst..dst + bands].copy_from_slice(&input.data[src..src + bands]);
            }
        }
    }
}

pub(crate) struct SubsampleBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<Subsample<F>>,
}

impl<F: BandFormat> SubsampleBridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    pub fn new(xfac: u32, yfac: u32, bands: u32) -> Result<Self, BuildError> {
        Ok(Self {
            inner: OperationBridge::new(Subsample::new(xfac, yfac)?, bands),
        })
    }
}

impl<F: BandFormat> crate::domain::op::DynOperation for SubsampleBridge<F>
where
    F::Sample: bytemuck::Pod + Copy + Send,
{
    fn input_format(&self) -> crate::domain::format::BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> crate::domain::format::BandFormatId {
        self.inner.output_format()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    fn demand_hint(&self) -> DemandHint {
        self.inner.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn output_width(&self, input_w: u32) -> u32 {
        input_w / self.inner.op.xfac
    }

    fn output_height(&self, input_h: u32) -> u32 {
        input_h / self.inner.op.yfac
    }

    fn dyn_start(&self) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start_with_tile(tile_w, tile_h)
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn std::any::Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        self.inner
            .dyn_process_region(state, input, output, input_region, output_region);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{pipeline::PipelineBuilder, sources::memory::MemorySource};
    use crate::domain::{
        error::BuildError,
        format::{BandFormatId, U8},
        image::{Region, Tile, TileMut},
        op::DynOperation,
    };
    use proptest::prelude::*;

    fn expected_subsample(
        input: &[u8],
        input_width: usize,
        bands: usize,
        output_width: usize,
        output_height: usize,
        xfac: usize,
        yfac: usize,
    ) -> Vec<u8> {
        let mut expected = vec![0u8; output_width * output_height * bands];

        for row in 0..output_height {
            let src_row = row * yfac;
            for col in 0..output_width {
                let src_col = col * xfac;
                let src = (src_row * input_width + src_col) * bands;
                let dst = (row * output_width + col) * bands;
                expected[dst..dst + bands].copy_from_slice(&input[src..src + bands]);
            }
        }

        expected
    }

    #[test]
    fn required_input_region_expands_by_factor() {
        let op = Subsample::<U8>::new(2, 3).unwrap();
        let output = Region::new(1, 1, 2, 2);
        let input = op.required_input_region(&output);
        assert_eq!(input, Region::new(2, 3, 3, 4));
    }

    #[test]
    fn required_input_region_is_empty_for_zero_sized_output() {
        let op = Subsample::<U8>::new(2, 3).unwrap();
        let input = op.required_input_region(&Region::new(1, 1, 0, 2));
        assert_eq!(input, Region::new(2, 3, 0, 0));
    }

    #[test]
    fn output_dimensions_round_down() {
        let bridge = SubsampleBridge::<U8>::new(2, 3, 1).unwrap();
        assert_eq!(bridge.output_width(7), 3);
        assert_eq!(bridge.output_height(8), 2);
    }

    #[test]
    fn structural_subsample_builder_zero_x_factor_returns_error() {
        let source = MemorySource::<U8>::new(4, 4, 1, (0u8..16).collect()).unwrap();
        let result = PipelineBuilder::from_source(source).subsample(0, 1);

        assert!(matches!(
            result,
            Err(BuildError::SourceHint {
                context: "subsample",
                ..
            })
        ));
    }

    #[test]
    fn node_spec_expands_input_tile_by_sampling_factor() {
        let op = Subsample::<U8>::new(3, 4).unwrap();
        let spec = op.node_spec(5, 6);
        assert_eq!(spec.input_tile_w, 13);
        assert_eq!(spec.input_tile_h, 21);
        assert_eq!(spec.output_tile_w, 5);
        assert_eq!(spec.output_tile_h, 6);
    }

    #[test]
    fn node_spec_uses_empty_input_tile_for_zero_sized_output() {
        let op = Subsample::<U8>::new(12, 5).unwrap();
        let spec = op.node_spec(0, 1);
        assert_eq!(spec.input_tile_w, 0);
        assert_eq!(spec.input_tile_h, 0);
        assert_eq!(spec.output_tile_w, 0);
        assert_eq!(spec.output_tile_h, 1);
    }

    #[test]
    fn bridge_metadata_and_dyn_dispatch_cover_multiband_tiles() {
        let bridge = SubsampleBridge::<U8>::new(2, 3, 2).unwrap();
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(bridge.bands(), 2);
        assert_eq!(bridge.demand_hint(), DemandHint::ThinStrip);

        let output_region = Region::new(0, 0, 2, 2);
        let input_region = bridge.required_input_region(&output_region);
        let input = vec![
            0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        ];
        let mut output = vec![0u8; 8];
        let mut state = bridge.dyn_start();
        bridge.dyn_process_region(
            state.as_mut(),
            &input,
            &mut output,
            input_region,
            output_region,
        );

        assert_eq!(output, vec![0u8, 1, 4, 5, 18, 19, 22, 23]);
    }

    #[test]
    fn process_region_decimates_integer_grid() {
        let op = Subsample::<U8>::new(2, 2).unwrap();
        let output_region = Region::new(0, 0, 2, 2);
        let input_region = op.required_input_region(&output_region);
        let input_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut output_data = vec![0u8; 4];
        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![1u8, 3, 7, 9]);
    }

    proptest! {
        #[test]
        fn subsample_factor_1_is_identity(rows in 1usize..=6, cols in 1usize..=6) {
            let pixels: Vec<u8> = (0..(rows * cols) as u8).collect();
            let output_region = Region::new(0, 0, cols as u32, rows as u32);
            let op = Subsample::<U8>::new(1, 1).unwrap();
            let input_region = op.required_input_region(&output_region);
            let mut output_data = vec![0u8; pixels.len()];
            let input = Tile::<U8>::new(input_region, 1, &pixels);
            let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixels);
        }

        #[test]
        fn subsample_single_pixel_input_survives_any_factor(
            pixel in prop::collection::vec(any::<u8>(), 1..=4),
            xfac in 1u32..=8,
            yfac in 1u32..=8,
        ) {
            let bands = pixel.len() as u32;
            let op = Subsample::<U8>::new(xfac, yfac).unwrap();
            let output_region = Region::new(0, 0, 1, 1);
            let input_region = op.required_input_region(&output_region);
            prop_assert_eq!(input_region, Region::new(0, 0, 1, 1));

            let mut output_data = vec![0u8; pixel.len()];
            let input = Tile::<U8>::new(input_region, bands, &pixel);
            let mut output = TileMut::<U8>::new(output_region, bands, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixel);
        }

        #[test]
        fn subsample_factor_1_is_identity_for_multiband_images(
            rows in 1usize..=4,
            cols in 1usize..=4,
            bands in 2usize..=4,
        ) {
            let pixels = (0..rows * cols * bands)
                .map(|idx| (idx % 251) as u8)
                .collect::<Vec<_>>();
            let output_region = Region::new(0, 0, cols as u32, rows as u32);
            let op = Subsample::<U8>::new(1, 1).unwrap();
            let input_region = op.required_input_region(&output_region);
            let mut output_data = vec![0u8; pixels.len()];
            let input = Tile::<U8>::new(input_region, bands as u32, &pixels);
            let mut output = TileMut::<U8>::new(output_region, bands as u32, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixels);
        }

        #[test]
        fn subsample_large_factors_pick_bottom_right_edge_samples(
            xfac in 2u32..=8,
            yfac in 2u32..=8,
            bands in 1usize..=4,
        ) {
            let input_width = (xfac + 1) as usize;
            let input_height = (yfac + 1) as usize;
            let pixels = (0..input_width * input_height * bands)
                .map(|idx| (idx % 251) as u8)
                .collect::<Vec<_>>();
            let output_region = Region::new(0, 0, 2, 2);
            let op = Subsample::<U8>::new(xfac, yfac).unwrap();
            let input_region = op.required_input_region(&output_region);
            let mut output_data = vec![0u8; 2 * 2 * bands];
            let input = Tile::<U8>::new(input_region, bands as u32, &pixels);
            let mut output = TileMut::<U8>::new(output_region, bands as u32, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);

            let expected = expected_subsample(
                &pixels,
                input_width,
                bands,
                2,
                2,
                xfac as usize,
                yfac as usize,
            );
            prop_assert_eq!(output_data, expected);
        }
    }

    #[test]
    fn one_pixel_wide_edge_case_uses_exact_required_rows() {
        let bands = 2usize;
        let op = Subsample::<U8>::new(1, 3).unwrap();
        let output_region = Region::new(0, 0, 1, 2);
        let input_region = op.required_input_region(&output_region);
        let pixels = (0..input_region.pixel_count() * bands)
            .map(|idx| (idx % 251) as u8)
            .collect::<Vec<_>>();
        let mut output_data = vec![0u8; output_region.pixel_count() * bands];
        let input = Tile::<U8>::new(input_region, bands as u32, &pixels);
        let mut output = TileMut::<U8>::new(output_region, bands as u32, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);

        let expected = expected_subsample(
            &pixels,
            input_region.width as usize,
            bands,
            output_region.width as usize,
            output_region.height as usize,
            1,
            3,
        );
        assert_eq!(output_data, expected);
    }

    #[test]
    fn one_pixel_tall_edge_case_uses_exact_required_columns() {
        let bands = 2usize;
        let op = Subsample::<U8>::new(3, 1).unwrap();
        let output_region = Region::new(0, 0, 2, 1);
        let input_region = op.required_input_region(&output_region);
        let pixels = (0..input_region.pixel_count() * bands)
            .map(|idx| (idx % 251) as u8)
            .collect::<Vec<_>>();
        let mut output_data = vec![0u8; output_region.pixel_count() * bands];
        let input = Tile::<U8>::new(input_region, bands as u32, &pixels);
        let mut output = TileMut::<U8>::new(output_region, bands as u32, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);

        let expected = expected_subsample(
            &pixels,
            input_region.width as usize,
            bands,
            output_region.width as usize,
            output_region.height as usize,
            3,
            1,
        );
        assert_eq!(output_data, expected);
    }
}
