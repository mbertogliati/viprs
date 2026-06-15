use crate::domain::{
    error::ViprsError,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    ops::resample::sample_conv::FromF64,
};

#[derive(Clone, Copy, Debug)]
struct Gradient {
    x: f32,
    y: f32,
}

/// Generate tiled Perlin noise using precomputed cell gradients.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::perlin::PerlinOp;
///
/// let op = PerlinOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug)]
pub struct PerlinOp {
    width: u32,
    height: u32,
    cell_size: u32,
    cells_across: u32,
    cells_down: u32,
    gradients: Box<[Gradient]>,
}

impl PerlinOp {
    /// Associated constant for default cell size.
    pub const DEFAULT_CELL_SIZE: u32 = 256;

    /// Creates a new `PerlinOp`.
    pub fn new(width: u32, height: u32, cell_size: u32, seed: u32) -> Result<Self, ViprsError> {
        if width == 0 || height == 0 {
            return Err(ViprsError::Scheduler(format!(
                "PerlinOp width and height must be > 0, got {width}x{height}"
            )));
        }
        if cell_size == 0 {
            return Err(ViprsError::Scheduler(
                "PerlinOp cell_size must be > 0".to_owned(),
            ));
        }

        let cells_across = width.div_ceil(cell_size);
        let cells_down = height.div_ceil(cell_size);
        let mut gradients = Vec::with_capacity(cells_across as usize * cells_down as usize);

        for cell_y in 0..cells_down {
            for cell_x in 0..cells_across {
                let hash = hash_cell(seed, cell_x as i32, cell_y as i32);
                let angle = ((hash ^ (hash >> 8) ^ (hash >> 16)) & 0xff) as f32
                    * std::f32::consts::TAU
                    / 256.0;
                gradients.push(Gradient {
                    x: angle.cos(),
                    y: angle.sin(),
                });
            }
        }

        Ok(Self {
            width,
            height,
            cell_size,
            cells_across,
            cells_down,
            gradients: gradients.into_boxed_slice(),
        })
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

    #[inline(always)]
    fn gradient_at(&self, cell_x: u32, cell_y: u32) -> Gradient {
        let wrapped_x = cell_x % self.cells_across;
        let wrapped_y = cell_y % self.cells_down;
        self.gradients[(wrapped_y * self.cells_across + wrapped_x) as usize]
    }
}

#[inline(always)]
fn vips_random_add(mut hash: u32, value: i32) -> u32 {
    for shift in [0, 8, 16, 24] {
        hash = (hash ^ ((value >> shift) as u32 & 0xff)).wrapping_mul(16_777_619);
    }
    hash
}

#[inline(always)]
fn hash_cell(seed: u32, cell_x: i32, cell_y: i32) -> u32 {
    let mixed = vips_random_add(seed, cell_y);
    vips_random_add(mixed, cell_x)
}

#[inline(always)]
fn smootherstep(x: f32) -> f32 {
    x * x * x * x.mul_add(x.mul_add(6.0, -15.0), 10.0)
}

impl Op for PerlinOp {
    type Input = F32;
    type Output = F32;
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
    fn process_region(&self, _state: &mut (), _input: &Tile<F32>, output: &mut TileMut<F32>) {
        debug_assert_eq!(output.bands, 1, "PerlinOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let region_width = output.region.width as usize;
        let cell_size = self.cell_size as f32;

        for row in 0..output.region.height as usize {
            let y = output.region.y as u32 + row as u32;
            let cell_y = y / self.cell_size;
            let dy = (y % self.cell_size) as f32 / cell_size;
            let sy = smootherstep(dy);

            for col in 0..region_width {
                let x = output.region.x as u32 + col as u32;
                let cell_x = x / self.cell_size;
                let dx = (x % self.cell_size) as f32 / cell_size;
                let sx = smootherstep(dx);

                let g00 = self.gradient_at(cell_x, cell_y);
                let g10 = self.gradient_at(cell_x + 1, cell_y);
                let g01 = self.gradient_at(cell_x, cell_y + 1);
                let g11 = self.gradient_at(cell_x + 1, cell_y + 1);

                let n00 = (-dx).mul_add(g00.x, -dy * g00.y);
                let n10 = (1.0 - dx).mul_add(g10.x, -dy * g10.y);
                let ix0 = sx.mul_add(n10 - n00, n00);

                let n01 = (-dx).mul_add(g01.x, (1.0 - dy) * g01.y);
                let n11 = (1.0 - dx).mul_add(g11.x, (1.0 - dy) * g11.y);
                let ix1 = sx.mul_add(n11 - n01, n01);

                let noise = sy.mul_add(ix1 - ix0, ix0);
                output.data[row * region_width + col] = f32::from_f64(
                    ((f64::from(noise.clamp(-1.0, 1.0)) + 1.0) * 0.5).clamp(0.0, 1.0),
                );
            }
        }
    }
}

impl PixelLocalOp for PerlinOp {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::{Region, Tile, TileMut};
    use proptest::prelude::*;

    fn render_region(op: &PerlinOp, region: Region) -> Vec<f32> {
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn render_full(op: &PerlinOp) -> Vec<f32> {
        render_region(op, Region::new(0, 0, op.width(), op.height()))
    }

    #[test]
    fn constructor_rejects_invalid_geometry() {
        assert!(PerlinOp::new(0, 8, 4, 1).is_err());
        assert!(PerlinOp::new(8, 0, 4, 1).is_err());
        assert!(PerlinOp::new(8, 8, 0, 1).is_err());
    }

    #[test]
    fn output_is_deterministic_for_same_seed() {
        let first = PerlinOp::new(16, 16, 4, 7).unwrap();
        let second = PerlinOp::new(16, 16, 4, 7).unwrap();
        assert_eq!(render_full(&first), render_full(&second));
    }

    #[test]
    fn partial_tiles_match_full_render() {
        let op = PerlinOp::new(16, 16, 4, 42).unwrap();
        let full = render_full(&op);
        let region = Region::new(5, 6, 4, 3);
        let partial = render_region(&op, region);

        for row in 0..region.height as usize {
            for col in 0..region.width as usize {
                let full_index =
                    (row + region.y as usize) * op.width() as usize + col + region.x as usize;
                let partial_index = row * region.width as usize + col;
                assert_eq!(partial[partial_index], full[full_index]);
            }
        }
    }

    proptest! {
        #[test]
        fn prop_output_stays_in_unit_interval(
            width in 1u32..=32,
            height in 1u32..=32,
            cell_size in 1u32..=16,
            seed in any::<u32>(),
        ) {
            let op = PerlinOp::new(width, height, cell_size, seed).unwrap();
            let output = render_full(&op);
            prop_assert_eq!(output.len(), width as usize * height as usize);
            prop_assert!(output.iter().all(|sample| sample.is_finite()));
            prop_assert!(output.iter().all(|sample| (0.0..=1.0).contains(sample)));
        }
    }
}
