use std::marker::PhantomData;

use crate::domain::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    ops::resample::sample_conv::FromF64,
};

/// Generate a libvips-style cosine zone plate.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::zone::ZoneOp;
///
/// let op = ZoneOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug)]
pub struct ZoneOp<F: BandFormat> {
    width: u32,
    height: u32,
    uchar: bool,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Copy for ZoneOp<F> {}

impl<F: BandFormat> Clone for ZoneOp<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: BandFormat> ZoneOp<F> {
    /// Creates a new `ZoneOp`.
    pub fn new(width: u32, height: u32) -> Result<Self, ViprsError> {
        if width == 0 || height == 0 {
            return Err(ViprsError::Scheduler(format!(
                "ZoneOp width and height must be > 0, got {width}x{height}"
            )));
        }

        Ok(Self {
            width,
            height,
            uchar: false,
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns this value configured with uchar.
    pub const fn with_uchar(mut self, uchar: bool) -> Self {
        self.uchar = uchar;
        self
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        self.height
    }
}

impl<F> Op for ZoneOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(output.bands, 1, "ZoneOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let hwidth = f64::from(self.width / 2);
        let hheight = f64::from(self.height / 2);
        let c = std::f64::consts::PI / f64::from(self.width.max(1));
        let region_width = output.region.width as usize;

        for row in 0..output.region.height as usize {
            let y = f64::from(output.region.y as u32 + row as u32);
            for col in 0..region_width {
                let x = f64::from(output.region.x as u32 + col as u32);
                let dx = x - hwidth;
                let dy = y - hheight;
                let value = (c * (dx * dx + dy * dy)).cos();
                let sample = if self.uchar {
                    ((value.clamp(-1.0, 1.0) + 1.0) * 0.5) * 255.0
                } else {
                    value
                };
                output.data[row * region_width + col] = F::Sample::from_f64(sample);
            }
        }
    }
}

impl<F> PixelLocalOp for ZoneOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::F32,
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn render(op: ZoneOp<F32>) -> Vec<f32> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn constructor_rejects_zero_dimensions() {
        assert!(ZoneOp::<F32>::new(0, 4).is_err());
        assert!(ZoneOp::<F32>::new(4, 0).is_err());
    }

    #[test]
    fn centre_of_odd_sized_zone_plate_is_one() {
        let samples = render(ZoneOp::<F32>::new(5, 5).unwrap());
        assert!((samples[2 * 5 + 2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn uchar_mode_scales_to_byte_range() {
        let samples = render(ZoneOp::<F32>::new(7, 7).unwrap().with_uchar(true));
        assert!(samples.iter().all(|sample| (0.0..=255.0).contains(sample)));
    }

    #[test]
    fn partial_region_matches_full_render_slice() {
        let op = ZoneOp::<F32>::new(6, 6).unwrap();
        let full = render(op);
        let region = Region::new(2, 1, 3, 2);
        let input_region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; input_region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(input_region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        let full_width = op.width() as usize;
        assert_eq!(
            output_data,
            vec![
                full[full_width + 2],
                full[full_width + 3],
                full[full_width + 4],
                full[full_width * 2 + 2],
                full[full_width * 2 + 3],
                full[full_width * 2 + 4],
            ]
        );
    }

    #[test]
    fn accessors_report_requested_geometry() {
        let op = ZoneOp::<F32>::new(6, 8).unwrap().with_uchar(true);
        assert_eq!(op.width(), 6);
        assert_eq!(op.height(), 8);
        assert_eq!(op.demand_hint(), DemandHint::Any);
    }

    proptest! {
        #[test]
        fn prop_output_has_expected_dimensions_and_range(
            width in 1u32..=32,
            height in 1u32..=32,
            uchar in any::<bool>(),
        ) {
            let op = ZoneOp::<F32>::new(width, height).unwrap().with_uchar(uchar);
            let samples = render(op);

            prop_assert_eq!(samples.len(), width as usize * height as usize);
            for sample in samples {
                if uchar {
                    prop_assert!((0.0..=255.0).contains(&sample));
                } else {
                    prop_assert!((-1.0 - 1e-6..=1.0 + 1e-6).contains(&sample));
                }
            }
        }
    }
}
