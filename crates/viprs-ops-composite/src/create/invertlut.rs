#![allow(clippy::missing_fields_in_debug)]
// REASON: the custom debug view intentionally omits large generated tables to keep logs readable.

use std::{cmp::Ordering, fmt, marker::PhantomData};

use viprs_core::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::FromF64,
};

/// Generate the inverse LUT for a normalized XY table, matching libvips `invertlut`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::invertlut::InvertlutOp;
///
/// let op = InvertlutOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct InvertlutOp<F: BandFormat> {
    size: u32,
    bands: u32,
    lut: Box<[f64]>,
    _format: PhantomData<F>,
}

impl<F: BandFormat> fmt::Debug for InvertlutOp<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InvertlutOp")
            .field("size", &self.size)
            .field("bands", &self.bands)
            .finish()
    }
}

impl<F: BandFormat> InvertlutOp<F> {
    /// Creates a new `InvertlutOp`.
    pub fn new(table: &[f64], rows: usize, cols: usize, size: u32) -> Result<Self, ViprsError> {
        if rows == 0 || cols < 2 || table.len() != rows * cols {
            return Err(ViprsError::Scheduler(format!(
                "InvertlutOp expects a {rows}x{cols} matrix with at least 1 row and 2 cols"
            )));
        }
        if !(1..=65_536).contains(&size) {
            return Err(ViprsError::Scheduler(format!(
                "InvertlutOp size must be in [1, 65536], got {size}"
            )));
        }

        let bands = cols - 1;
        let mut row_refs = (0..rows)
            .map(|row| &table[row * cols..(row + 1) * cols])
            .collect::<Vec<_>>();

        for (index, value) in table.iter().enumerate() {
            if !(0.0..=1.0).contains(value) {
                let x = index % cols;
                let y = index / cols;
                return Err(ViprsError::Scheduler(format!(
                    "InvertlutOp element ({x}, {y}) is {value}, outside range [0,1]"
                )));
            }
        }

        row_refs.sort_by(|lhs, rhs| lhs[0].partial_cmp(&rhs[0]).unwrap_or(Ordering::Equal));

        let lut = build_inverted_lut(&row_refs, size as usize, bands);

        Ok(Self {
            size,
            bands: bands as u32,
            lut,
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        self.size
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        1
    }

    #[must_use]
    /// Returns or performs bands.
    pub const fn bands(&self) -> u32 {
        self.bands
    }
}

fn build_inverted_lut(rows: &[&[f64]], size: usize, bands: usize) -> Box<[f64]> {
    let height = rows.len();
    let mut buffer = vec![0.0; size * bands];

    for band in 0..bands {
        let first = (rows[0][band + 1] * (size - 1) as f64) as usize;
        let last = (rows[height - 1][band + 1] * (size - 1) as f64) as usize;

        for k in 0..first {
            let fac = if first == 0 {
                0.0
            } else {
                rows[0][0] / first as f64
            };
            buffer[band + k * bands] = k as f64 * fac;
        }

        for k in last..size {
            let denom = (size - 1).saturating_sub(last);
            let fac = if denom == 0 {
                0.0
            } else {
                (1.0 - rows[height - 1][0]) / denom as f64
            };
            buffer[band + k * bands] = ((k - last) as f64).mul_add(fac, rows[height - 1][0]);
        }

        for k in first..=last {
            let ki = if size == 1 {
                0.0
            } else {
                k as f64 / (size - 1) as f64
            };

            let mut j = height as isize - 1;
            while j >= 0 && rows[j as usize][band + 1] >= ki {
                j -= 1;
            }
            if j < 0 {
                j = 0;
            }

            if height > 1 && (j as usize + 1) < height {
                let lower = rows[j as usize];
                let upper = rows[j as usize + 1];
                let irange = upper[band + 1] - lower[band + 1];
                let orange = upper[0] - lower[0];

                buffer[band + k * bands] = if irange.abs() <= f64::EPSILON {
                    lower[0]
                } else {
                    orange.mul_add((ki - lower[band + 1]) / irange, lower[0])
                };
            } else {
                buffer[band + k * bands] = rows[j as usize][0];
            }
        }
    }

    buffer.into_boxed_slice()
}

impl<F> Op for InvertlutOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(output.bands, self.bands);
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width());
        debug_assert!(output.region.y as u32 + output.region.height <= self.height());

        let bands = self.bands as usize;
        let region_width = output.region.width as usize;
        let image_width = self.width() as usize;

        for row in 0..output.region.height as usize {
            let src_row = (output.region.y as usize + row) * image_width * bands;
            let src_col = output.region.x as usize * bands;
            let dst_row = row * region_width * bands;

            for col in 0..region_width {
                let src_base = src_row + src_col + col * bands;
                let dst_base = dst_row + col * bands;
                for band in 0..bands {
                    output.data[dst_base + band] = F::Sample::from_f64(self.lut[src_base + band]);
                }
            }
        }
    }
}

impl<F> PixelLocalOp for InvertlutOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U8},
        image::{Region, Tile, TileMut},
    };

    fn render_f32(op: &InvertlutOp<F32>) -> Vec<f32> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; region.pixel_count() * op.bands() as usize];
        let mut output_data = vec![0.0f32; region.pixel_count() * op.bands() as usize];
        let input = Tile::<F32>::new(region, op.bands(), &input_data);
        let mut output = TileMut::<F32>::new(region, op.bands(), &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn render_u8(op: &InvertlutOp<U8>) -> Vec<u8> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0u8; region.pixel_count() * op.bands() as usize];
        let mut output_data = vec![0u8; region.pixel_count() * op.bands() as usize];
        let input = Tile::<U8>::new(region, op.bands(), &input_data);
        let mut output = TileMut::<U8>::new(region, op.bands(), &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn render_f32_region(op: &InvertlutOp<F32>, output_region: Region) -> Vec<f32> {
        let input_region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; input_region.pixel_count() * op.bands() as usize];
        let mut output_data = vec![0.0f32; output_region.pixel_count() * op.bands() as usize];
        let input = Tile::<F32>::new(input_region, op.bands(), &input_data);
        let mut output = TileMut::<F32>::new(output_region, op.bands(), &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn dimensions_match_requested_size_and_bands() {
        let table = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let op = InvertlutOp::<F32>::new(&table, 2, 3, 16).unwrap();
        assert_eq!(op.width(), 16);
        assert_eq!(op.height(), 1);
        assert_eq!(op.bands(), 2);
    }

    #[test]
    fn identity_table_round_trips() {
        let table = [0.0, 0.0, 1.0, 1.0];
        let op = InvertlutOp::<F32>::new(&table, 2, 2, 8).unwrap();
        let rendered = render_f32(&op);

        for (index, sample) in rendered.iter().enumerate() {
            assert!((*sample - index as f32 / 7.0).abs() < 1e-6);
        }
    }

    #[test]
    fn output_can_be_quantized_to_u8() {
        let table = [0.0, 0.0, 1.0, 1.0];
        let op = InvertlutOp::<U8>::new(&table, 2, 2, 4).unwrap();
        let rendered = render_u8(&op);
        assert_eq!(rendered, vec![0, 0, 1, 1]);
    }

    #[test]
    fn rejects_invalid_matrices() {
        assert!(InvertlutOp::<F32>::new(&[], 0, 0, 8).is_err());
        assert!(InvertlutOp::<F32>::new(&[0.0, 1.2], 1, 2, 8).is_err());
        assert!(InvertlutOp::<F32>::new(&[0.0, 0.0, 1.0, 1.0], 2, 2, 0).is_err());
    }

    #[test]
    fn size_one_table_uses_first_control_point_without_dividing_by_zero() {
        let table = [0.25, 0.75, 1.0, 1.0];
        let op = InvertlutOp::<F32>::new(&table, 2, 2, 1).unwrap();

        let rendered = render_f32(&op);
        assert_eq!(rendered.len(), 1);
        assert!(rendered[0].is_finite());
    }

    #[test]
    fn repeated_output_coordinate_uses_lower_segment_value() {
        let table = [0.2, 0.4, 0.8, 0.4];
        let op = InvertlutOp::<F32>::new(&table, 2, 2, 8).unwrap();
        let rendered = render_f32(&op);

        assert!((rendered[2] - 0.2).abs() < 1e-6);
        assert!(rendered[0] < rendered[2]);
        assert!(rendered[7] > rendered[2]);
    }

    #[test]
    fn process_region_honours_partial_output_region_and_multiband_layout() {
        let table = [
            0.0, 0.0, 0.2, //
            0.5, 0.5, 0.6, //
            1.0, 1.0, 1.0,
        ];
        let op = InvertlutOp::<F32>::new(&table, 3, 3, 5).unwrap();

        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(
            op.required_input_region(&Region::new(1, 0, 2, 1)),
            Region::new(1, 0, 2, 1)
        );

        let partial = render_f32_region(&op, Region::new(1, 0, 2, 1));
        let full = render_f32(&op);

        assert_eq!(partial, full[2..6].to_vec());
    }

    proptest! {
        #[test]
        fn prop_monotone_identity_stays_in_range(
            size in 2u32..=128,
            midpoint_x in 0.1f64..=0.9,
            midpoint_y in 0.1f64..=0.9,
        ) {
            let table = [0.0, 0.0, midpoint_x, midpoint_y, 1.0, 1.0];
            let op = InvertlutOp::<F32>::new(&table, 3, 2, size).unwrap();
            let rendered = render_f32(&op);

            prop_assert_eq!(rendered.len(), size as usize);
            for window in rendered.windows(2) {
                prop_assert!(window[0] >= 0.0 && window[0] <= 1.0);
                prop_assert!(window[1] >= 0.0 && window[1] <= 1.0);
                prop_assert!(window[0] <= window[1] + 1e-6);
            }
        }
    }
}
