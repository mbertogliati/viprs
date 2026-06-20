use std::marker::PhantomData;
use std::time::{SystemTime, UNIX_EPOCH};

use viprs_core::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::FromF64,
};

/// Generate deterministic Gaussian noise via libvips' 12-sample approximation.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::gaussnoise::GaussnoiseOp;
///
/// let op = GaussnoiseOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug)]
pub struct GaussnoiseOp<F: BandFormat> {
    mean: f64,
    sigma: f64,
    seed: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Copy for GaussnoiseOp<F> {}

impl<F: BandFormat> Clone for GaussnoiseOp<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: BandFormat> GaussnoiseOp<F> {
    /// Creates a new `GaussnoiseOp`.
    pub fn new(mean: f64, sigma: f64) -> Result<Self, ViprsError> {
        if !mean.is_finite() {
            return Err(ViprsError::Scheduler(format!(
                "GaussnoiseOp mean must be finite, got {mean}"
            )));
        }
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(ViprsError::Scheduler(format!(
                "GaussnoiseOp sigma must be finite and >= 0, got {sigma}"
            )));
        }

        Ok(Self {
            mean,
            sigma,
            seed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.subsec_nanos()),
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns this value configured with seed.
    pub const fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
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
fn vips_random(seed: u32) -> u32 {
    vips_random_add(2_166_136_261, seed as i32)
}

#[inline(always)]
fn gaussian_sample(seed: u32, x: i32, y: i32, mean: f64, sigma: f64) -> f64 {
    let mut mixed = seed;
    mixed = vips_random_add(mixed, x);
    mixed = vips_random_add(mixed, y);

    let mut sum = 0.0;
    for _ in 0..12 {
        mixed = vips_random(mixed);
        sum += f64::from(mixed) / f64::from(u32::MAX);
    }

    (sum - 6.0).mul_add(sigma, mean)
}

impl<F> Op for GaussnoiseOp<F>
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
        debug_assert_eq!(output.bands, 1, "GaussnoiseOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);

        let region_width = output.region.width as usize;
        for row in 0..output.region.height as usize {
            let y = output.region.y + row as i32;
            for col in 0..region_width {
                let x = output.region.x + col as i32;
                let value = gaussian_sample(self.seed, x, y, self.mean, self.sigma);
                output.data[row * region_width + col] = F::Sample::from_f64(value);
            }
        }
    }
}

impl<F> PixelLocalOp for GaussnoiseOp<F>
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
        format::F32,
        image::{Region, Tile, TileMut},
    };

    fn render(width: u32, height: u32, op: GaussnoiseOp<F32>) -> Vec<f32> {
        let region = Region::new(0, 0, width, height);
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn constructor_rejects_invalid_parameters() {
        assert!(GaussnoiseOp::<F32>::new(f64::NAN, 1.0).is_err());
        assert!(GaussnoiseOp::<F32>::new(0.0, -1.0).is_err());
    }

    #[test]
    fn output_is_deterministic_for_the_same_seed() {
        let op = GaussnoiseOp::<F32>::new(10.0, 3.0).unwrap().with_seed(42);
        let first = render(8, 8, op);
        let second = render(8, 8, op);

        assert_eq!(first, second);
    }

    #[test]
    fn different_regions_share_the_same_seeded_pixel_values() {
        let op = GaussnoiseOp::<F32>::new(0.0, 1.0).unwrap().with_seed(77);

        let full = render(8, 8, op);
        let region = Region::new(3, 2, 2, 3);
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut partial = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut partial);
        op.process_region(&mut (), &input, &mut output);

        for row in 0..region.height as usize {
            for col in 0..region.width as usize {
                let full_idx = (row + region.y as usize) * 8 + (col + region.x as usize);
                let partial_idx = row * region.width as usize + col;
                assert_eq!(partial[partial_idx], full[full_idx]);
            }
        }
    }

    proptest! {
        #[test]
        fn prop_output_has_expected_dimensions_and_finite_values(
            width in 1u32..=32,
            height in 1u32..=32,
            mean in -100.0f64..=100.0,
            sigma in 0.0f64..=50.0,
            seed in any::<u32>(),
        ) {
            let op = GaussnoiseOp::<F32>::new(mean, sigma).unwrap().with_seed(seed);
            let samples = render(width, height, op);

            prop_assert_eq!(samples.len(), width as usize * height as usize);
            prop_assert!(samples.iter().all(|sample| sample.is_finite()));
        }
    }
}
