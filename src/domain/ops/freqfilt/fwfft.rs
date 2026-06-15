use std::{marker::PhantomData, sync::Arc};

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use crate::domain::{
    error::{FreqfiltError, ViprsError},
    format::{BandFormat, F32},
    image::{DemandHint, Region, Tile, TileMut},
    op::Op,
    ops::{freqfilt::COMPLEX_BANDS, resample::sample_conv::ToF64},
};

#[doc(hidden)]
pub struct Fft2dScratch {
    pub(crate) spectrum: Vec<Complex32>,
    pub(crate) column: Vec<Complex32>,
    /// Pre-allocated scratch for `Fft::process_with_scratch` on rows, avoiding
    /// the per-call `Vec` allocation that `Fft::process` performs internally.
    pub(crate) row_scratch: Vec<Complex32>,
    /// Pre-allocated scratch for `Fft::process_with_scratch` on columns.
    pub(crate) col_scratch: Vec<Complex32>,
}

impl Clone for Fft2dScratch {
    fn clone(&self) -> Self {
        Self {
            spectrum: self.spectrum.clone(),
            column: self.column.clone(),
            row_scratch: self.row_scratch.clone(),
            col_scratch: self.col_scratch.clone(),
        }
    }
}

impl Fft2dScratch {
    pub(crate) fn new(
        width: u32,
        height: u32,
        row_scratch_len: usize,
        col_scratch_len: usize,
    ) -> Result<Self, ViprsError> {
        let pixel_count = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| n.try_into().ok())
            .ok_or(ViprsError::ImageTooLarge {
                width,
                height,
                bands: 1,
                bytes: u128::from(width) * u128::from(height),
                limit_bytes: usize::MAX as u128,
                details: "fft scratch pixel count exceeds addressable memory",
            })?;
        let column_len = usize::try_from(height).map_err(|_| ViprsError::ImageTooLarge {
            width,
            height,
            bands: COMPLEX_BANDS,
            bytes: u128::from(height) * std::mem::size_of::<Complex32>() as u128,
            limit_bytes: usize::MAX as u128,
            details: "fft scratch column exceeds addressable memory",
        })?;
        let mut spectrum = Vec::new();
        spectrum
            .try_reserve_exact(pixel_count)
            .map_err(|_| ViprsError::ImageTooLarge {
                width,
                height,
                bands: COMPLEX_BANDS,
                bytes: u128::from(width)
                    * u128::from(height)
                    * std::mem::size_of::<Complex32>() as u128,
                limit_bytes: usize::MAX as u128,
                details: "fft scratch spectrum exceeds addressable memory",
            })?;
        spectrum.resize(pixel_count, Complex32::default());

        let mut column = Vec::new();
        column
            .try_reserve_exact(column_len)
            .map_err(|_| ViprsError::ImageTooLarge {
                width,
                height,
                bands: COMPLEX_BANDS,
                bytes: u128::from(height) * std::mem::size_of::<Complex32>() as u128,
                limit_bytes: usize::MAX as u128,
                details: "fft scratch column exceeds addressable memory",
            })?;
        column.resize(column_len, Complex32::default());

        let mut row_scratch = Vec::new();
        row_scratch
            .try_reserve_exact(row_scratch_len)
            .map_err(|_| ViprsError::ImageTooLarge {
                width,
                height,
                bands: COMPLEX_BANDS,
                bytes: row_scratch_len as u128 * std::mem::size_of::<Complex32>() as u128,
                limit_bytes: usize::MAX as u128,
                details: "fft row scratch exceeds addressable memory",
            })?;
        row_scratch.resize(row_scratch_len, Complex32::default());

        let mut col_scratch = Vec::new();
        col_scratch
            .try_reserve_exact(col_scratch_len)
            .map_err(|_| ViprsError::ImageTooLarge {
                width,
                height,
                bands: COMPLEX_BANDS,
                bytes: col_scratch_len as u128 * std::mem::size_of::<Complex32>() as u128,
                limit_bytes: usize::MAX as u128,
                details: "fft column scratch exceeds addressable memory",
            })?;
        col_scratch.resize(col_scratch_len, Complex32::default());

        Ok(Self {
            spectrum,
            column,
            row_scratch,
            col_scratch,
        })
    }
}

pub(crate) type Fft2dState = Fft2dScratch;

// STATIC DISPATCH IMPOSSIBLE: rustfft's `FftPlanner` selects specialised FFT
// algorithm implementations (Radix4, MixedRadix, Bluestein, etc.) at runtime
// based on the transform length. The concrete type is not known at compile time
// and is not exposed publicly by rustfft. The dyn dispatch cost is amortised:
// plans are created once at Op construction, and the per-call overhead is a
// single vtable lookup per row/column — dominated by the O(N log N) FFT work.
pub(crate) type FftPlan = Arc<dyn Fft<f32>>;

/// Applies a 2D FFT in-place using pre-allocated scratch buffers.
///
/// Generic over `R` and `C` to enable monomorphization when the caller has a
/// concrete `Fft<f32>` type. Current callers pass `dyn Fft<f32>` (from rustfft's
/// opaque planner output), which still dispatches through the vtable — but the
/// per-call heap allocation that `Fft::process()` performs internally is avoided
/// by using `process_with_scratch` with pre-allocated buffers.
pub(crate) fn apply_fft_2d_in_place<R: Fft<f32> + ?Sized, C: Fft<f32> + ?Sized>(
    buffer: &mut [Complex32],
    column: &mut [Complex32],
    width: usize,
    height: usize,
    row_fft: &R,
    column_fft: &C,
    row_scratch: &mut [Complex32],
    col_scratch: &mut [Complex32],
) {
    for row in buffer.chunks_exact_mut(width) {
        row_fft.process_with_scratch(row, row_scratch);
    }

    for x in 0..width {
        for y in 0..height {
            column[y] = buffer[y * width + x];
        }
        column_fft.process_with_scratch(column, col_scratch);
        for y in 0..height {
            buffer[y * width + x] = column[y];
        }
    }
}

/// Applies the `forward FFT` frequency-domain operation to the image. Use it for FFT-driven
/// filtering, spectrum analysis, or complex-domain transforms.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::freqfilt::fwfft::FwFftOp;
///
/// let op = FwFftOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Clone)]
pub struct FwFftOp<F: BandFormat> {
    width: usize,
    height: usize,
    // STATIC DISPATCH IMPOSSIBLE: see FftPlan type alias comment above.
    row_fft: Option<FftPlan>,
    column_fft: Option<FftPlan>,
    scratch_template: Arc<Fft2dScratch>,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> FwFftOp<F> {
    pub fn new(width: u32, height: u32) -> Result<Self, ViprsError> {
        let width_usize = width as usize;
        let height_usize = height as usize;
        let (row_fft, column_fft, row_scratch_len, col_scratch_len) =
            if width_usize == 0 || height_usize == 0 {
                (None, None, 0, 0)
            } else {
                let mut planner = FftPlanner::<f32>::new();
                let row = planner.plan_fft_forward(width_usize);
                let col = planner.plan_fft_forward(height_usize);
                let rsl = row.get_inplace_scratch_len();
                let csl = col.get_inplace_scratch_len();
                (Some(row), Some(col), rsl, csl)
            };

        Ok(Self {
            width: width_usize,
            height: height_usize,
            row_fft,
            column_fft,
            scratch_template: Arc::new(Fft2dScratch::new(
                width,
                height,
                row_scratch_len,
                col_scratch_len,
            )?),
            _phantom: PhantomData,
        })
    }
}

impl<F: BandFormat> std::fmt::Debug for FwFftOp<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FwFftOp")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl<F> Op for FwFftOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + bytemuck::Pod,
{
    type Input = F;
    type Output = F32;
    type State = Fft2dState;

    const OUTPUT_BANDS: Option<usize> = Some(COMPLEX_BANDS as usize);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::FullImage
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) -> Self::State {
        self.scratch_template.as_ref().clone()
    }

    fn validate_region_contract(
        &self,
        input_region: Region,
        input_bands: u32,
        output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        let expected_region = Region::new(0, 0, self.width as u32, self.height as u32);
        if input_bands != 1
            || output_bands != COMPLEX_BANDS
            || input_region != expected_region
            || output_region != expected_region
        {
            return Err(ViprsError::Freqfilt(FreqfiltError::FwfftContract {
                input_bands,
                output_bands,
                input_region,
                output_region,
            }));
        }

        Ok(())
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<F>, output: &mut TileMut<F32>) {
        if self
            .validate_region_contract(input.region, input.bands, output.region, output.bands)
            .is_err()
        {
            return;
        }

        let Some(row_fft) = self.row_fft.as_deref() else {
            return;
        };
        let Some(column_fft) = self.column_fft.as_deref() else {
            return;
        };

        for (complex, sample) in state.spectrum.iter_mut().zip(input.data.iter()) {
            *complex = Complex32::new(sample.to_f64() as f32, 0.0);
        }

        let Fft2dScratch {
            spectrum,
            column,
            row_scratch,
            col_scratch,
        } = state;
        apply_fft_2d_in_place(
            spectrum,
            column,
            self.width,
            self.height,
            row_fft,
            column_fft,
            row_scratch,
            col_scratch,
        );

        let scale = 1.0 / (self.width * self.height) as f32;
        for (pixel, value) in spectrum.iter().enumerate() {
            let offset = pixel * COMPLEX_BANDS as usize;
            output.data[offset] = value.re * scale;
            output.data[offset + 1] = value.im * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        error::{FreqfiltError, ViprsError},
        format::{F32, U8},
        image::{DemandHint, Region},
        op::{NodeSpec, Op},
    };
    use proptest::prelude::*;
    fn run_fwfft<F>(width: u32, height: u32, input_data: &[F::Sample]) -> Vec<f32>
    where
        F: BandFormat,
        F::Sample: ToF64 + bytemuck::Pod,
    {
        let op = FwFftOp::<F>::new(width, height).expect("FwFftOp should construct");
        let region = Region::new(0, 0, width, height);
        let input = Tile::<F>::new(region, 1, input_data);
        let mut output_data =
            vec![0.0f32; width as usize * height as usize * COMPLEX_BANDS as usize];
        let mut output = TileMut::<F32>::new(region, COMPLEX_BANDS, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    proptest! {
        #[test]
        fn constant_signal_maps_all_energy_to_dc(
            width in 1_u32..=8,
            height in 1_u32..=8,
            value in -32.0f32..32.0f32,
        ) {
            let input = vec![value; (width * height) as usize];
            let output = run_fwfft::<F32>(width, height, &input);

            prop_assert!((output[0] - value).abs() <= 1e-4);
            prop_assert!(output[1].abs() <= 1e-4);

            for chunk in output[2..].chunks_exact(COMPLEX_BANDS as usize) {
                prop_assert!(chunk[0].abs() <= 1e-4);
                prop_assert!(chunk[1].abs() <= 1e-4);
            }
        }
    }

    #[test]
    fn single_pixel_forward_fft_is_real_only() {
        let output = run_fwfft::<U8>(1, 1, &[7]);

        assert_eq!(output, vec![7.0, 0.0]);
    }

    #[test]
    fn pure_sine_wave_maps_to_unit_frequency_spikes_with_forward_normalization() {
        let samples = (0..8)
            .map(|x| (2.0 * std::f32::consts::PI * x as f32 / 8.0).sin())
            .collect::<Vec<_>>();

        let output = run_fwfft::<F32>(8, 1, &samples);

        for (index, pair) in output.chunks_exact(COMPLEX_BANDS as usize).enumerate() {
            match index {
                1 => {
                    assert!(pair[0].abs() <= 1e-5);
                    assert!((pair[1] + 0.5).abs() <= 1e-4);
                }
                7 => {
                    assert!(pair[0].abs() <= 1e-5);
                    assert!((pair[1] - 0.5).abs() <= 1e-4);
                }
                _ => {
                    assert!(pair[0].abs() <= 1e-5);
                    assert!(pair[1].abs() <= 1e-5);
                }
            }
        }
    }

    #[test]
    fn reports_full_image_complex_contract() {
        let op = FwFftOp::<F32>::new(5, 7).expect("FwFftOp should construct");
        let region = Region::new(0, 0, 5, 7);

        assert_eq!(op.demand_hint(), DemandHint::FullImage);
        assert_eq!(op.required_input_region(&region), region);
        assert!(
            op.validate_region_contract(region, 1, region, COMPLEX_BANDS)
                .is_ok()
        );
        assert_eq!(op.node_spec(3, 4), NodeSpec::identity(3, 4));
        assert_eq!(
            <FwFftOp<F32> as Op>::OUTPUT_BANDS,
            Some(COMPLEX_BANDS as usize)
        );
        assert_eq!(format!("{op:?}"), "FwFftOp { width: 5, height: 7 }");
    }

    #[test]
    fn invalid_contract_surfaces_a_typed_error() {
        let op = FwFftOp::<F32>::new(2, 2).expect("FwFftOp should construct");
        let err = op
            .validate_region_contract(Region::new(0, 0, 1, 2), 1, Region::new(0, 0, 2, 2), 2)
            .unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::FwfftContract {
                input_bands: 1,
                output_bands: 2,
                input_region,
                output_region,
            }) if input_region == Region::new(0, 0, 1, 2)
                && output_region == Region::new(0, 0, 2, 2)
        ));
    }

    #[test]
    fn invalid_contract_execute_region_returns_typed_error_and_process_region_leaves_output_unchanged()
     {
        let op = FwFftOp::<F32>::new(2, 2).expect("FwFftOp should construct");
        let input = Tile::<F32>::new(Region::new(0, 0, 1, 2), 1, &[1.0, 2.0]);
        let mut output_data = vec![7.5f32; 2 * 2 * COMPLEX_BANDS as usize];
        let mut output =
            TileMut::<F32>::new(Region::new(0, 0, 2, 2), COMPLEX_BANDS, &mut output_data);
        let mut state = op.start();
        let err = op
            .execute_region(&mut state, &input, &mut output)
            .unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::FwfftContract {
                input_bands: 1,
                output_bands: COMPLEX_BANDS,
                input_region,
                output_region,
            }) if input_region == Region::new(0, 0, 1, 2)
                && output_region == Region::new(0, 0, 2, 2)
        ));
        drop(output);

        let mut output_data = vec![7.5f32; 2 * 2 * COMPLEX_BANDS as usize];
        let mut output =
            TileMut::<F32>::new(Region::new(0, 0, 2, 2), COMPLEX_BANDS, &mut output_data);
        op.process_region(&mut state, &input, &mut output);
        drop(output);

        assert!(
            output_data
                .iter()
                .all(|sample| (*sample - 7.5).abs() <= f32::EPSILON)
        );
    }

    #[test]
    fn start_returns_independent_preallocated_scratch_buffers() {
        let op = FwFftOp::<F32>::new(3, 2).expect("FwFftOp should construct");
        let state = op.start();
        let second = op.start();

        assert_ne!(
            state.spectrum.as_ptr(),
            second.spectrum.as_ptr(),
            "start() must return distinct scratch storage"
        );
        assert_eq!(state.spectrum.len(), 6);
        assert_eq!(state.column.len(), 2);
        assert_eq!(second.spectrum.len(), 6);
        assert_eq!(second.column.len(), 2);
    }

    #[test]
    fn concurrent_starts_keep_mutable_scratch_isolated() {
        let op = FwFftOp::<F32>::new(2, 2).expect("FwFftOp should construct");
        let first = op.start();
        let second = op.start();
        let first_worker = std::thread::spawn(move || {
            let mut state = first;
            state.spectrum[0] = Complex32::new(1.0, 10.0);
            state.column[0] = Complex32::new(2.0, 20.0);
            state
        });
        let second_worker = std::thread::spawn(move || {
            let mut state = second;
            state.spectrum[0] = Complex32::new(3.0, 30.0);
            state.column[0] = Complex32::new(4.0, 40.0);
            state
        });

        let first_state = first_worker.join().expect("first worker should complete");
        let second_state = second_worker.join().expect("second worker should complete");

        assert_eq!(first_state.spectrum[0], Complex32::new(1.0, 10.0));
        assert_eq!(first_state.column[0], Complex32::new(2.0, 20.0));
        assert_eq!(second_state.spectrum[0], Complex32::new(3.0, 30.0));
        assert_eq!(second_state.column[0], Complex32::new(4.0, 40.0));
    }

    #[test]
    fn scratch_allocator_rejects_oversized_dimensions() {
        let err = match Fft2dScratch::new(u32::MAX, u32::MAX, 0, 0) {
            Ok(_) => panic!("expected oversized scratch allocation to fail"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge { width, height, .. }
                if width == u32::MAX && height == u32::MAX
        ));
    }

    #[test]
    fn zero_sized_fft_has_empty_state_and_output() {
        let op = FwFftOp::<F32>::new(0, 0).expect("FwFftOp should construct");
        let region = Region::new(0, 0, 0, 0);
        let input = Tile::<F32>::new(region, 1, &[]);
        let mut output_data = Vec::new();
        let mut output = TileMut::<F32>::new(region, COMPLEX_BANDS, &mut output_data);
        let mut state = op.start();

        assert!(state.spectrum.is_empty());
        assert!(state.column.is_empty());

        op.process_region(&mut state, &input, &mut output);

        assert!(output_data.is_empty());
    }
}
