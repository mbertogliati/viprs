use std::marker::PhantomData;

use crate::{
    domain::op::{NodeSpec, Op},
    domain::{
        format::BandFormat,
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Libvips-style 3D transpose for vertically stacked page strips.
///
/// Output scanline `yo` reads input scanline
/// `yi = (yo % output_page_height) * page_height + (yo / output_page_height)`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::conversion::transpose3d::Transpose3dOp;
///
/// let op = Transpose3dOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Transpose3dOp<F: BandFormat> {
    page_height: u32,
    output_page_height: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Transpose3dOp<F> {
    #[must_use]
    /// Creates a new `Transpose3dOp`.
    pub fn new(image_height: u32, page_height: u32) -> Self {
        debug_assert!(page_height > 0, "Transpose3dOp requires page_height >= 1");
        debug_assert!(
            image_height.is_multiple_of(page_height),
            "Transpose3dOp requires image_height to be a multiple of page_height"
        );

        Self {
            page_height,
            output_page_height: image_height / page_height,
            _format: PhantomData,
        }
    }

    #[inline(always)]
    fn input_y(&self, output_y: i32) -> i32 {
        debug_assert!(output_y >= 0, "Transpose3dOp expects non-negative output y");
        let output_y = output_y as u32;
        let output_page = output_y / self.output_page_height;
        let output_line = output_y % self.output_page_height;
        (output_line * self.page_height + output_page) as i32
    }

    fn bounding_input_y_range(&self, output_y: i32, output_height: u32) -> (i32, i32) {
        if output_height == 0 {
            let yi = self.input_y(output_y);
            return (yi, yi);
        }

        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;
        for row in 0..output_height {
            let yi = self.input_y(output_y + row as i32);
            min_y = min_y.min(yi);
            max_y = max_y.max(yi);
        }

        (min_y, max_y)
    }
}

impl<F> Op for Transpose3dOp<F>
where
    F: BandFormat,
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        // Exact transpose3d demand is a non-contiguous scanline set, so
        // the current rectangular scheduler must force one output row per tile.
        DemandHint::OneLine
    }

    fn required_input_region(&self, output: &Region) -> Region {
        let (min_y, max_y) = self.bounding_input_y_range(output.y, output.height);
        Region::new(
            output.x,
            min_y,
            output.width,
            output
                .height
                .checked_sub(1)
                .map_or(0, |_| (max_y - min_y + 1) as u32),
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w,
            input_tile_h: if tile_h == 0 {
                0
            } else {
                tile_h
                    .saturating_sub(1)
                    .saturating_mul(self.page_height)
                    .saturating_add(1)
            },
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(
            input.bands, output.bands,
            "Transpose3dOp preserves band count"
        );
        debug_assert_eq!(
            input.region.width, output.region.width,
            "Transpose3dOp requires matching packed input/output widths"
        );

        let bands = output.bands as usize;
        let output_width = output.region.width as usize;
        let input_width = input.region.width as usize;
        let row_bytes = output_width * bands;

        for row in 0..output.region.height as usize {
            let yo = output.region.y + row as i32;
            let yi = self.input_y(yo);
            let src_row = (yi - input.region.y) as usize;
            let src = src_row * input_width * bands;
            let dst = row * row_bytes;

            output.data[dst..dst + row_bytes].copy_from_slice(&input.data[src..src + row_bytes]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{format::U8, op::OperationBridge},
        ports::scheduler::TileScheduler,
    };
    use proptest::prelude::*;

    fn run_transpose3d(
        width: u32,
        image_height: u32,
        page_height: u32,
        pixels: Vec<u8>,
    ) -> Vec<u8> {
        let source = MemorySource::<U8>::new(width, image_height, 1, pixels).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(OperationBridge::new(
                Transpose3dOp::<U8>::new(image_height, page_height),
                1,
            )))
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(RayonScheduler::default_threads())
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();
        sink.into_buffer()
    }

    fn run_transpose3d_round_trip(
        width: u32,
        image_height: u32,
        page_height: u32,
        pixels: Vec<u8>,
    ) -> Vec<u8> {
        let output_page_height = image_height / page_height;
        let source = MemorySource::<U8>::new(width, image_height, 1, pixels).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(OperationBridge::new(
                Transpose3dOp::<U8>::new(image_height, page_height),
                1,
            )))
            .unwrap()
            .then(Box::new(OperationBridge::new(
                Transpose3dOp::<U8>::new(image_height, output_page_height),
                1,
            )))
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(RayonScheduler::default_threads())
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();
        sink.into_buffer()
    }

    #[test]
    fn required_input_region_bounds_cross_page_rows() {
        let op = Transpose3dOp::<U8>::new(12, 3);
        assert_eq!(
            op.required_input_region(&Region::new(4, 3, 5, 2)),
            Region::new(4, 1, 5, 9)
        );
    }

    #[test]
    fn small_example_matches_libvips_row_mapping() {
        let input = vec![
            10u8, 11, //
            20, 21, //
            30, 31, //
            40, 41, //
            50, 51, //
            60, 61, //
        ];
        let output = run_transpose3d(2, 6, 2, input);
        assert_eq!(
            output,
            vec![
                10u8, 11, //
                30, 31, //
                50, 51, //
                20, 21, //
                40, 41, //
                60, 61, //
            ]
        );
    }

    proptest! {
        #[test]
        fn single_page_transpose_is_identity(
            (width, image_height, pixels) in (1u32..=8, 1u32..=8).prop_flat_map(|(width, image_height)| {
                let len = (width * image_height) as usize;
                (Just(width), Just(image_height), proptest::collection::vec(any::<u8>(), len))
            })
        ) {
            let output = run_transpose3d(width, image_height, image_height, pixels.clone());
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn transpose_round_trip_restores_input(
            (width, page_height, page_count, pixels) in
                (1u32..=8, 1u32..=6, 1u32..=6).prop_flat_map(|(width, page_height, page_count)| {
                    let image_height = page_height * page_count;
                    let len = (width * image_height) as usize;
                    (
                        Just(width),
                        Just(page_height),
                        Just(page_count),
                        proptest::collection::vec(any::<u8>(), len),
                    )
                })
        ) {
            let image_height = page_height * page_count;
            let output = run_transpose3d_round_trip(width, image_height, page_height, pixels.clone());
            prop_assert_eq!(output, pixels);
        }
    }
}
