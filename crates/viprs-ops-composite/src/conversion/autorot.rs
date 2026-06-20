use std::{any::Any, marker::PhantomData};

use viprs_core::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, ImageMetadata, Region, Tile, TileMut},
    op::{DynOperation, NodeSpec, Op, OperationBridge},
};

/// Rotation angle reported by libvips autorot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutorotAngle {
    /// Uses the `D0` variant of `AutorotAngle`.
    D0,
    /// Uses the `D90` variant of `AutorotAngle`.
    D90,
    /// Uses the `D180` variant of `AutorotAngle`.
    D180,
    /// Uses the `D270` variant of `AutorotAngle`.
    D270,
}

/// EXIF-orientation driven rotate/flip transform.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::conversion::autorot::AutorotOp;
///
/// let op = AutorotOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct AutorotOp<F: BandFormat> {
    image_width: u32,
    image_height: u32,
    angle: AutorotAngle,
    flip: bool,
    _format: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> AutorotOp<F> {
    #[must_use]
    /// Creates a new `AutorotOp`.
    pub const fn new(image_width: u32, image_height: u32, orientation: u8) -> Self {
        let (angle, flip) = transform_for_orientation(orientation);
        Self::from_transform(image_width, image_height, angle, flip)
    }

    #[must_use]
    /// Creates this value from transform.
    pub const fn from_transform(
        image_width: u32,
        image_height: u32,
        angle: AutorotAngle,
        flip: bool,
    ) -> Self {
        Self {
            image_width,
            image_height,
            angle,
            flip,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs angle.
    pub const fn angle(&self) -> AutorotAngle {
        self.angle
    }

    #[must_use]
    /// Returns or performs flip.
    pub const fn flip(&self) -> bool {
        self.flip
    }

    #[must_use]
    /// Returns or performs output width.
    pub const fn output_width(&self) -> u32 {
        match self.angle {
            AutorotAngle::D0 | AutorotAngle::D180 => self.image_width,
            AutorotAngle::D90 | AutorotAngle::D270 => self.image_height,
        }
    }

    #[must_use]
    /// Returns or performs output height.
    pub const fn output_height(&self) -> u32 {
        match self.angle {
            AutorotAngle::D0 | AutorotAngle::D180 => self.image_height,
            AutorotAngle::D90 | AutorotAngle::D270 => self.image_width,
        }
    }

    #[must_use]
    /// Returns or performs apply metadata.
    pub fn apply_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        let mut metadata = source.clone();
        metadata.remove_orientation();
        metadata
    }

    const fn map_output_to_input(&self, output_x: i32, output_y: i32) -> (i32, i32) {
        let rotated_width = self.output_width() as i32;
        let (rot_x, rot_y) = if self.flip {
            (rotated_width - 1 - output_x, output_y)
        } else {
            (output_x, output_y)
        };

        match self.angle {
            AutorotAngle::D0 => (rot_x, rot_y),
            AutorotAngle::D90 => (rot_y, self.image_height as i32 - 1 - rot_x),
            AutorotAngle::D180 => (
                self.image_width as i32 - 1 - rot_x,
                self.image_height as i32 - 1 - rot_y,
            ),
            AutorotAngle::D270 => (self.image_width as i32 - 1 - rot_y, rot_x),
        }
    }
}

const fn transform_for_orientation(orientation: u8) -> (AutorotAngle, bool) {
    match orientation {
        2 => (AutorotAngle::D0, true),
        3 => (AutorotAngle::D180, false),
        4 => (AutorotAngle::D180, true),
        5 => (AutorotAngle::D90, true),
        6 => (AutorotAngle::D90, false),
        7 => (AutorotAngle::D270, true),
        8 => (AutorotAngle::D270, false),
        _ => (AutorotAngle::D0, false),
    }
}

impl<F: BandFormat> Op for AutorotOp<F>
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
            let (x, y) = self.map_output_to_input(output.x, output.y);
            return Region::new(x, y, 0, 0);
        }

        let right = output.x + output.width as i32 - 1;
        let bottom = output.y + output.height as i32 - 1;
        let corners = [
            self.map_output_to_input(output.x, output.y),
            self.map_output_to_input(right, output.y),
            self.map_output_to_input(output.x, bottom),
            self.map_output_to_input(right, bottom),
        ];

        let mut min_x = corners[0].0;
        let mut max_x = corners[0].0;
        let mut min_y = corners[0].1;
        let mut max_y = corners[0].1;

        for &(x, y) in &corners[1..] {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        Region::new(
            min_x,
            min_y,
            (max_x - min_x + 1) as u32,
            (max_y - min_y + 1) as u32,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        match self.angle {
            AutorotAngle::D90 | AutorotAngle::D270 => NodeSpec {
                input_tile_w: tile_h,
                input_tile_h: tile_w,
                output_tile_w: tile_w,
                output_tile_h: tile_h,
                coordinate_driven_source: None,
            },
            AutorotAngle::D0 | AutorotAngle::D180 => NodeSpec::identity(tile_w, tile_h),
        }
    }

    fn start(&self) {}

    fn transform_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        self.apply_metadata(source)
    }

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(input.bands, output.bands, "AutorotOp preserves band count");

        let bands = input.bands as usize;
        let input_width = input.region.width as usize;
        let output_width = output.region.width as usize;
        let output_height = output.region.height as usize;

        for row in 0..output_height {
            let output_y = output.region.y + row as i32;
            for col in 0..output_width {
                let output_x = output.region.x + col as i32;
                let (input_x, input_y) = self.map_output_to_input(output_x, output_y);
                debug_assert!(input_x >= input.region.x);
                debug_assert!(input_y >= input.region.y);
                let local_x = (input_x - input.region.x) as usize;
                let local_y = (input_y - input.region.y) as usize;
                let src = (local_y * input_width + local_x) * bands;
                let dst = (row * output_width + col) * bands;
                output.data[dst..dst + bands].copy_from_slice(&input.data[src..src + bands]);
            }
        }
    }
}

/// Dynamic bridge that reports autorot output dimensions.
pub struct AutorotBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<AutorotOp<F>>,
}

impl<F: BandFormat> AutorotBridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    #[must_use]
    /// Creates a new `AutorotBridge`.
    pub fn new(image_width: u32, image_height: u32, bands: u32, orientation: u8) -> Self {
        Self {
            inner: OperationBridge::new(
                AutorotOp::new(image_width, image_height, orientation),
                bands,
            ),
        }
    }

    #[must_use]
    /// Returns or performs apply metadata.
    pub fn apply_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        self.inner.op.apply_metadata(source)
    }
}

impl<F: BandFormat> DynOperation for AutorotBridge<F>
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

    fn output_width(&self, _input_w: u32) -> u32 {
        self.inner.op.output_width()
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        self.inner.op.output_height()
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

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use crate::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        ports::scheduler::TileScheduler,
    };
    use proptest::prelude::*;
    use viprs_core::{format::U8, image::ImageMetadata, op::DynOperation};

    fn run_autorot(
        width: u32,
        height: u32,
        orientation: u8,
        output_region: Region,
        input_data: &[u8],
    ) -> Vec<u8> {
        let op = AutorotOp::<U8>::new(width, height, orientation);
        let input_region = op.required_input_region(&output_region);
        let input = Tile::<U8>::new(input_region, 1, input_data);
        let mut output_data = vec![0u8; output_region.pixel_count()];
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn orientation_six_rotates_clockwise() {
        let input = [1u8, 2, 3, 4, 5, 6];
        let output = run_autorot(2, 3, 6, Region::new(0, 0, 3, 2), &input);
        assert_eq!(output, vec![5u8, 3, 1, 6, 4, 2]);
    }

    #[test]
    fn orientation_two_flips_horizontally() {
        let input = [1u8, 2, 3, 4, 5, 6];
        let output = run_autorot(3, 2, 2, Region::new(0, 0, 3, 2), &input);
        assert_eq!(output, vec![3u8, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn orientation_five_rotates_then_flips() {
        let input = [1u8, 2, 3, 4, 5, 6];
        let output = run_autorot(2, 3, 5, Region::new(0, 0, 3, 2), &input);
        assert_eq!(output, vec![1u8, 3, 5, 2, 4, 6]);
    }

    #[test]
    fn bridge_reports_orientation_outputs() {
        let bridge = AutorotBridge::<U8>::new(2, 3, 1, 6);
        assert_eq!(bridge.output_width(2), 3);
        assert_eq!(bridge.output_height(3), 2);
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(
            bridge.node_spec(4, 2),
            NodeSpec {
                input_tile_w: 2,
                input_tile_h: 4,
                output_tile_w: 4,
                output_tile_h: 2,
                coordinate_driven_source: None,
            }
        );
    }

    #[test]
    fn pipeline_applies_orientation_six() {
        let source = MemorySource::<U8>::new(2, 3, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(AutorotBridge::<U8>::new(2, 3, 1, 6)))
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(1)
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();

        assert_eq!(pipeline.width, 3);
        assert_eq!(pipeline.height, 2);
        assert_eq!(sink.into_buffer(), vec![5u8, 3, 1, 6, 4, 2]);
    }

    #[test]
    fn apply_metadata_removes_structured_and_raw_exif_orientation() {
        let op = AutorotOp::<U8>::new(1, 1, 6);
        let exif = decode_hex_fixture(include_str!(
            "../../../../tests/fixtures/autorot/exif_ifd0_orientation_6.hex"
        ));
        let source = ImageMetadata {
            orientation: Some(6),
            exif: Some(exif),
            ..ImageMetadata::default()
        };
        let output = op.apply_metadata(&source);
        assert_eq!(output.orientation, None);
        assert_eq!(
            output.exif,
            Some(decode_hex_fixture(include_str!(
                "../../../../tests/fixtures/autorot/exif_ifd0_without_orientation.hex"
            )))
        );
    }

    proptest! {
        #[test]
        fn orientation_one_is_identity(width in 1usize..=8, height in 1usize..=8) {
            let len = width * height;
            let pixels = (0..len).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let output = run_autorot(
                width as u32,
                height as u32,
                1,
                Region::new(0, 0, width as u32, height as u32),
                &pixels,
            );
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn orientation_three_twice_is_identity(width in 1usize..=8, height in 1usize..=8) {
            let len = width * height;
            let pixels = (0..len).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let region = Region::new(0, 0, width as u32, height as u32);
            let once = run_autorot(width as u32, height as u32, 3, region, &pixels);
            let twice = run_autorot(width as u32, height as u32, 3, region, &once);
            prop_assert_eq!(twice, pixels);
        }
    }

    fn decode_hex_fixture(source: &str) -> Vec<u8> {
        source
            .split_ascii_whitespace()
            .map(|byte| u8::from_str_radix(byte, 16).unwrap())
            .collect()
    }
}
