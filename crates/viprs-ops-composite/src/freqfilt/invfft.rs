use std::{marker::PhantomData, sync::Arc};

use rustfft::{FftPlanner, num_complex::Complex32};

use crate::freqfilt::{
    COMPLEX_BANDS,
    fwfft::{Fft2dScratch, Fft2dState, FftPlan, apply_fft_2d_in_place},
};

use viprs_core::{
    error::{FreqfiltError, ViprsError},
    format::{BandFormat, F32},
    image::{DemandHint, Region, Tile, TileMut},
    op::Op,
    shared_ops::sample_conv::ToF64,
};

/// Applies the `inverse FFT` frequency-domain operation to the image. Use it for FFT-driven
/// filtering, spectrum analysis, or complex-domain transforms.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::freqfilt::invfft::InvFftOp;
///
/// let op = InvFftOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Clone)]
pub struct InvFftOp<F: BandFormat> {
    width: usize,
    height: usize,
    // STATIC DISPATCH IMPOSSIBLE: see FftPlan type alias comment in fwfft.rs.
    row_fft: Option<FftPlan>,
    column_fft: Option<FftPlan>,
    scratch_template: Arc<Fft2dScratch>,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> InvFftOp<F> {
    /// Creates an inverse FFT operation for the given dimensions.
    pub fn new(width: u32, height: u32) -> Result<Self, ViprsError> {
        let width_usize = width as usize;
        let height_usize = height as usize;
        let (row_fft, column_fft, row_scratch_len, col_scratch_len) =
            if width_usize == 0 || height_usize == 0 {
                (None, None, 0, 0)
            } else {
                let mut planner = FftPlanner::<f32>::new();
                let row = planner.plan_fft_inverse(width_usize);
                let col = planner.plan_fft_inverse(height_usize);
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

impl<F: BandFormat> std::fmt::Debug for InvFftOp<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvFftOp")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl<F> Op for InvFftOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + bytemuck::Pod,
{
    type Input = F;
    type Output = F32;
    type State = Fft2dState;

    const OUTPUT_BANDS: Option<usize> = Some(1);

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
        if input_bands != COMPLEX_BANDS
            || output_bands != 1
            || input_region != expected_region
            || output_region != expected_region
        {
            return Err(ViprsError::Freqfilt(FreqfiltError::InvfftContract {
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

        for (complex, chunk) in state
            .spectrum
            .iter_mut()
            .zip(input.data.chunks_exact(COMPLEX_BANDS as usize))
        {
            *complex = Complex32::new(chunk[0].to_f64() as f32, chunk[1].to_f64() as f32);
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

        for (sample, value) in output.data.iter_mut().zip(spectrum.iter()) {
            *sample = value.re;
        }
    }
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use crate::freqfilt::{COMPLEX_BANDS, FwFftOp};

    use proptest::prelude::*;
    use viprs_core::{
        error::{FreqfiltError, ViprsError},
        format::{F32, U8},
        image::{DemandHint, Region},
        op::{NodeSpec, Op},
    };
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

    fn run_invfft(width: u32, height: u32, input_data: &[f32]) -> Vec<f32> {
        let op = InvFftOp::<F32>::new(width, height).expect("InvFftOp should construct");
        let region = Region::new(0, 0, width, height);
        let input = Tile::<F32>::new(region, COMPLEX_BANDS, input_data);
        let mut output_data = vec![0.0f32; width as usize * height as usize];
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    prop_compose! {
        fn mono_images()
            (width in 1_u32..=8, height in 1_u32..=8)
            (
                width in Just(width),
                height in Just(height),
                pixels in prop::collection::vec(-32.0f32..32.0f32, (width * height) as usize),
            ) -> (u32, u32, Vec<f32>) {
                (width, height, pixels)
            }
    }

    proptest! {
        #[test]
        fn inverse_of_forward_round_trips_real_signal((width, height, pixels) in mono_images()) {
            let spectrum = run_fwfft::<F32>(width, height, &pixels);
            let reconstructed = run_invfft(width, height, &spectrum);

            for (expected, actual) in pixels.iter().zip(reconstructed.iter()) {
                prop_assert!((*expected - *actual).abs() <= 1e-3);
            }
        }
    }

    #[test]
    fn inverse_fft_of_single_dc_sample_is_constant_image() {
        let reconstructed = run_invfft(1, 1, &[9.0, 0.0]);

        assert_eq!(reconstructed, vec![9.0]);
    }

    #[test]
    fn integer_input_round_trip_returns_original_values_as_f32() {
        let spectrum = run_fwfft::<U8>(2, 2, &[1, 2, 3, 4]);
        let reconstructed = run_invfft(2, 2, &spectrum);

        assert_eq!(reconstructed, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn reports_full_image_real_contract() {
        let op = InvFftOp::<F32>::new(6, 5).expect("InvFftOp should construct");
        let region = Region::new(0, 0, 6, 5);

        assert_eq!(op.demand_hint(), DemandHint::FullImage);
        assert_eq!(op.required_input_region(&region), region);
        assert!(
            op.validate_region_contract(region, COMPLEX_BANDS, region, 1)
                .is_ok()
        );
        assert_eq!(op.node_spec(4, 3), NodeSpec::identity(4, 3));
        assert_eq!(<InvFftOp<F32> as Op>::OUTPUT_BANDS, Some(1));
        assert_eq!(format!("{op:?}"), "InvFftOp { width: 6, height: 5, .. }");
    }

    #[test]
    fn invalid_contract_surfaces_a_typed_error() {
        let op = InvFftOp::<F32>::new(2, 2).expect("InvFftOp should construct");
        let err = op
            .validate_region_contract(
                Region::new(0, 0, 2, 1),
                COMPLEX_BANDS,
                Region::new(0, 0, 2, 2),
                1,
            )
            .unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::InvfftContract {
                input_bands: 2,
                output_bands: 1,
                input_region,
                output_region,
            }) if input_region == Region::new(0, 0, 2, 1)
                && output_region == Region::new(0, 0, 2, 2)
        ));
    }

    #[test]
    fn invalid_contract_execute_region_returns_typed_error_and_process_region_leaves_output_unchanged()
     {
        let op = InvFftOp::<F32>::new(2, 2).expect("InvFftOp should construct");
        let input = Tile::<F32>::new(
            Region::new(0, 0, 2, 1),
            COMPLEX_BANDS,
            &[1.0, 0.0, 2.0, 0.0],
        );
        let mut output_data = vec![7.5f32; 2 * 2];
        let mut output = TileMut::<F32>::new(Region::new(0, 0, 2, 2), 1, &mut output_data);
        let mut state = op.start();
        let err = op
            .execute_region(&mut state, &input, &mut output)
            .unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::InvfftContract {
                input_bands: COMPLEX_BANDS,
                output_bands: 1,
                input_region,
                output_region,
            }) if input_region == Region::new(0, 0, 2, 1)
                && output_region == Region::new(0, 0, 2, 2)
        ));
        drop(output);

        let mut output_data = vec![7.5f32; 2 * 2];
        let mut output = TileMut::<F32>::new(Region::new(0, 0, 2, 2), 1, &mut output_data);
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
        let op = InvFftOp::<F32>::new(3, 2).expect("InvFftOp should construct");
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
        let op = InvFftOp::<F32>::new(2, 2).expect("InvFftOp should construct");
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

        let first = first_worker.join().expect("first worker should complete");
        let second = second_worker.join().expect("second worker should complete");
        assert_eq!(first.spectrum[0], Complex32::new(1.0, 10.0));
        assert_eq!(first.column[0], Complex32::new(2.0, 20.0));
        assert_eq!(second.spectrum[0], Complex32::new(3.0, 30.0));
        assert_eq!(second.column[0], Complex32::new(4.0, 40.0));
    }

    #[test]
    fn zero_sized_inverse_fft_has_empty_state_and_output() {
        let op = InvFftOp::<F32>::new(0, 0).expect("InvFftOp should construct");
        let region = Region::new(0, 0, 0, 0);
        let input = Tile::<F32>::new(region, COMPLEX_BANDS, &[]);
        let mut output_data = Vec::new();
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = op.start();

        assert!(state.spectrum.is_empty());
        assert!(state.column.is_empty());

        op.process_region(&mut state, &input, &mut output);

        assert!(output_data.is_empty());
    }
}
