use std::{any::Any, marker::PhantomData};

use crate::{
    domain::op::{DynOperation, NodeSpec, Op, OperationBridge},
    domain::{
        format::{BandFormat, BandFormatId},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enumerates the available angle values.
pub enum Angle {
    /// Uses the `D0` variant of `Angle`.
    D0,
    /// Uses the `D90` variant of `Angle`.
    D90,
    /// Uses the `D180` variant of `Angle`.
    D180,
    /// Uses the `D270` variant of `Angle`.
    D270,
}

/// Rotate by a multiple of 90 degrees clockwise.
pub struct Rot<F: BandFormat> {
    image_width: u32,
    image_height: u32,
    angle: Angle,
    _format: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> Rot<F> {
    #[must_use]
    /// Creates a new `Rot`.
    pub const fn new(image_width: u32, image_height: u32, angle: Angle) -> Self {
        Self {
            image_width,
            image_height,
            angle,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs angle.
    pub const fn angle(&self) -> Angle {
        self.angle
    }
}

impl<F: BandFormat> Op for Rot<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn preferred_tile_geometry(&self) -> DemandHint {
        match self.angle {
            Angle::D0 | Angle::D90 | Angle::D180 | Angle::D270 => DemandHint::SmallTile,
        }
    }

    fn required_input_region(&self, output: &Region) -> Region {
        match self.angle {
            Angle::D0 => *output,
            Angle::D90 => crate::domain::ops::structural::rotate90::Rotate90::<F>::new(
                self.image_width,
                self.image_height,
            )
            .required_input_region(output),
            Angle::D180 => crate::domain::ops::structural::rotate180::Rotate180::<F>::new(
                self.image_width,
                self.image_height,
            )
            .required_input_region(output),
            Angle::D270 => crate::domain::ops::structural::rotate270::Rotate270::<F>::new(
                self.image_width,
                self.image_height,
            )
            .required_input_region(output),
        }
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        match self.angle {
            Angle::D90 | Angle::D270 => NodeSpec {
                input_tile_w: tile_h,
                input_tile_h: tile_w,
                output_tile_w: tile_w,
                output_tile_h: tile_h,
                coordinate_driven_source: None,
            },
            Angle::D0 | Angle::D180 => NodeSpec::identity(tile_w, tile_h),
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        match self.angle {
            Angle::D0 => output.data.copy_from_slice(input.data),
            Angle::D90 => {
                crate::domain::ops::structural::rotate90::Rotate90::<F>::new(
                    self.image_width,
                    self.image_height,
                )
                .process_region(state, input, output);
            }
            Angle::D180 => {
                crate::domain::ops::structural::rotate180::Rotate180::<F>::new(
                    self.image_width,
                    self.image_height,
                )
                .process_region(state, input, output);
            }
            Angle::D270 => {
                crate::domain::ops::structural::rotate270::Rotate270::<F>::new(
                    self.image_width,
                    self.image_height,
                )
                .process_region(state, input, output);
            }
        }
    }
}

pub(crate) struct RotBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<Rot<F>>,
}

impl<F: BandFormat> RotBridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    pub fn new(image_width: u32, image_height: u32, angle: Angle, bands: u32) -> Self {
        Self {
            inner: OperationBridge::new(Rot::new(image_width, image_height, angle), bands),
        }
    }
}

impl<F: BandFormat> DynOperation for RotBridge<F>
where
    F::Sample: bytemuck::Pod + Copy + Send,
{
    fn input_format(&self) -> BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> BandFormatId {
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
        match self.inner.op.angle() {
            Angle::D90 | Angle::D270 => self.inner.op.image_height,
            Angle::D0 | Angle::D180 => input_w,
        }
    }

    fn output_height(&self, input_h: u32) -> u32 {
        match self.inner.op.angle() {
            Angle::D90 | Angle::D270 => self.inner.op.image_width,
            Angle::D0 | Angle::D180 => input_h,
        }
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn Any + Send> {
        self.inner.dyn_start_with_tile(tile_w, tile_h)
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn Any,
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
    use crate::domain::{
        format::{BandFormatId, U8},
        op::DynOperation,
    };
    use proptest::prelude::*;

    fn run_rot(op: &Rot<U8>, input_data: &[u8], output_region: Region) -> Vec<u8> {
        let input_region = op.required_input_region(&output_region);
        let mut output = vec![0u8; output_region.pixel_count()];
        let input = Tile::<U8>::new(input_region, 1, input_data);
        let mut output_tile = TileMut::<U8>::new(output_region, 1, &mut output);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    fn output_dimensions(angle: Angle, image_width: u32, image_height: u32) -> (u32, u32) {
        match angle {
            Angle::D0 | Angle::D180 => (image_width, image_height),
            Angle::D90 | Angle::D270 => (image_height, image_width),
        }
    }

    fn extract_region(data: &[u8], image_width: u32, region: Region) -> Vec<u8> {
        let start_x = region.x as usize;
        let start_y = region.y as usize;
        let width = region.width as usize;
        let height = region.height as usize;
        let stride = image_width as usize;
        let mut pixels = Vec::with_capacity(region.pixel_count());

        for row in 0..height {
            let offset = (start_y + row) * stride + start_x;
            pixels.extend_from_slice(&data[offset..offset + width]);
        }

        pixels
    }

    fn rotate_image(data: &[u8], image_width: u32, image_height: u32, angle: Angle) -> Vec<u8> {
        let (output_width, output_height) = output_dimensions(angle, image_width, image_height);
        let mut output = vec![0u8; output_width as usize * output_height as usize];

        for output_y in 0..output_height {
            for output_x in 0..output_width {
                let (input_x, input_y) = match angle {
                    Angle::D0 => (output_x, output_y),
                    Angle::D90 => (output_y, image_height - 1 - output_x),
                    Angle::D180 => (image_width - 1 - output_x, image_height - 1 - output_y),
                    Angle::D270 => (image_width - 1 - output_y, output_x),
                };
                let src = (input_y * image_width + input_x) as usize;
                let dst = (output_y * output_width + output_x) as usize;
                output[dst] = data[src];
            }
        }

        output
    }

    fn expected_region_pixels(
        image_width: u32,
        image_height: u32,
        pixels: &[u8],
        angle: Angle,
        output_region: Region,
    ) -> Vec<u8> {
        let rotated = rotate_image(pixels, image_width, image_height, angle);
        let (output_width, _) = output_dimensions(angle, image_width, image_height);
        extract_region(&rotated, output_width, output_region)
    }

    #[test]
    fn d90_required_input_region_matches_libvips_geometry() {
        let op = Rot::<U8>::new(8, 6, Angle::D90);
        assert_eq!(
            op.required_input_region(&Region::new(0, 0, 4, 3)),
            Region::new(0, 2, 3, 4)
        );
    }

    #[test]
    fn d90_bridge_transposes_dimensions() {
        let bridge = RotBridge::<U8>::new(8, 6, Angle::D90, 1);
        assert_eq!(bridge.output_width(8), 6);
        assert_eq!(bridge.output_height(6), 8);
        assert_eq!(bridge.demand_hint(), DemandHint::SmallTile);
    }

    #[test]
    fn d180_rotates_boundary_2x2() {
        let op = Rot::<U8>::new(2, 2, Angle::D180);
        assert_eq!(
            run_rot(&op, &[1u8, 2, 3, 4], Region::new(0, 0, 2, 2)),
            vec![4u8, 3, 2, 1]
        );
    }

    #[test]
    fn all_angles_map_partial_regions_for_non_square_images() {
        let image_width = 5;
        let image_height = 3;
        let pixels = (0..image_width * image_height)
            .map(|value| value as u8)
            .collect::<Vec<_>>();
        let edge_aligned = Region::new(2, 1, 3, 2);
        let edge_aligned_transposed = Region::new(1, 2, 2, 3);
        let transposed_spec = NodeSpec {
            input_tile_w: 2,
            input_tile_h: 4,
            output_tile_w: 4,
            output_tile_h: 2,
            coordinate_driven_source: None,
        };

        let cases = [
            (
                Angle::D0,
                edge_aligned,
                edge_aligned,
                DemandHint::SmallTile,
                NodeSpec::identity(4, 2),
            ),
            (
                Angle::D90,
                edge_aligned_transposed,
                Region::new(2, 0, 3, 2),
                DemandHint::SmallTile,
                transposed_spec,
            ),
            (
                Angle::D180,
                edge_aligned,
                Region::new(0, 0, 3, 2),
                DemandHint::SmallTile,
                NodeSpec::identity(4, 2),
            ),
            (
                Angle::D270,
                edge_aligned_transposed,
                Region::new(0, 1, 3, 2),
                DemandHint::SmallTile,
                transposed_spec,
            ),
        ];

        for (angle, output_region, expected_input_region, expected_hint, expected_node_spec) in
            cases
        {
            let op = Rot::<U8>::new(image_width, image_height, angle);

            assert_eq!(op.demand_hint(), expected_hint);
            assert_eq!(
                op.required_input_region(&output_region),
                expected_input_region
            );
            assert_eq!(op.node_spec(4, 2), expected_node_spec);

            let input_pixels = extract_region(&pixels, image_width, expected_input_region);
            let output_pixels = run_rot(&op, &input_pixels, output_region);

            assert_eq!(
                output_pixels,
                expected_region_pixels(image_width, image_height, &pixels, angle, output_region)
            );
        }
    }

    #[test]
    fn rot_bridge_covers_all_angles_and_region_contracts() {
        let image_width = 5;
        let image_height = 3;
        let pixels = (0..image_width * image_height)
            .map(|value| value as u8)
            .collect::<Vec<_>>();
        let edge_aligned = Region::new(2, 1, 3, 2);
        let edge_aligned_transposed = Region::new(1, 2, 2, 3);
        let transposed_spec = NodeSpec {
            input_tile_w: 2,
            input_tile_h: 4,
            output_tile_w: 4,
            output_tile_h: 2,
            coordinate_driven_source: None,
        };

        let cases = [
            (
                Angle::D0,
                edge_aligned,
                edge_aligned,
                DemandHint::SmallTile,
                NodeSpec::identity(4, 2),
                image_width,
                image_height,
                false,
            ),
            (
                Angle::D90,
                edge_aligned_transposed,
                Region::new(2, 0, 3, 2),
                DemandHint::SmallTile,
                transposed_spec,
                image_height,
                image_width,
                true,
            ),
            (
                Angle::D180,
                edge_aligned,
                Region::new(0, 0, 3, 2),
                DemandHint::SmallTile,
                NodeSpec::identity(4, 2),
                image_width,
                image_height,
                false,
            ),
            (
                Angle::D270,
                edge_aligned_transposed,
                Region::new(0, 1, 3, 2),
                DemandHint::SmallTile,
                transposed_spec,
                image_height,
                image_width,
                true,
            ),
        ];

        for (
            angle,
            output_region,
            expected_input_region,
            expected_hint,
            expected_node_spec,
            expected_output_width,
            expected_output_height,
            use_tile_start,
        ) in cases
        {
            let bridge = RotBridge::<U8>::new(image_width, image_height, angle, 1);
            let mut state = if use_tile_start {
                bridge.dyn_start_with_tile(4, 2)
            } else {
                bridge.dyn_start()
            };
            let mut output = vec![0u8; output_region.pixel_count()];

            assert_eq!(bridge.input_format(), BandFormatId::U8);
            assert_eq!(bridge.output_format(), BandFormatId::U8);
            assert_eq!(bridge.bands(), 1);
            assert_eq!(bridge.demand_hint(), expected_hint);
            assert_eq!(
                bridge.required_input_region(&output_region),
                expected_input_region
            );
            assert_eq!(bridge.node_spec(4, 2), expected_node_spec);
            assert_eq!(bridge.output_width(image_width), expected_output_width);
            assert_eq!(bridge.output_height(image_height), expected_output_height);

            let input = extract_region(&pixels, image_width, expected_input_region);
            bridge.dyn_process_region(
                state.as_mut(),
                &input,
                &mut output,
                expected_input_region,
                output_region,
            );

            assert_eq!(
                output,
                expected_region_pixels(image_width, image_height, &pixels, angle, output_region)
            );
        }
    }

    proptest! {
        #[test]
        fn d0_is_identity(rows in 1usize..=8, cols in 1usize..=8) {
            let pixels = (0..rows * cols).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let op = Rot::<U8>::new(cols as u32, rows as u32, Angle::D0);
            let result = run_rot(&op, &pixels, Region::new(0, 0, cols as u32, rows as u32));
            prop_assert_eq!(result, pixels);
        }

        #[test]
        fn four_clockwise_quarter_turns_are_identity(rows in 1usize..=6, cols in 1usize..=6) {
            let pixels = (0..rows * cols).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();

            let r1 = run_rot(
                &Rot::<U8>::new(cols as u32, rows as u32, Angle::D90),
                &pixels,
                Region::new(0, 0, rows as u32, cols as u32),
            );
            let r2 = run_rot(
                &Rot::<U8>::new(rows as u32, cols as u32, Angle::D90),
                &r1,
                Region::new(0, 0, cols as u32, rows as u32),
            );
            let r3 = run_rot(
                &Rot::<U8>::new(cols as u32, rows as u32, Angle::D90),
                &r2,
                Region::new(0, 0, rows as u32, cols as u32),
            );
            let r4 = run_rot(
                &Rot::<U8>::new(rows as u32, cols as u32, Angle::D90),
                &r3,
                Region::new(0, 0, cols as u32, rows as u32),
            );

            prop_assert_eq!(r4, pixels);
        }
    }
}
