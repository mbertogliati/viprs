#![cfg(feature = "fft")]

use viprs::{COMPLEX_BANDS, F32, FwFftOp, InvFftOp, Op, Region, Tile, TileMut};

#[test]
fn test_fft_roundtrip_pipeline() {
    let width = 4;
    let height = 4;
    let input_data = vec![
        0.0f32, 1.0, -1.0, 2.0, //
        3.0, -2.0, 0.5, 4.0, //
        -3.5, 2.5, 1.5, -0.5, //
        6.0, -4.0, 2.25, 1.25,
    ];
    let region = Region::new(0, 0, width, height);

    let spectrum = {
        let fwfft = FwFftOp::<F32>::new(width, height).expect("FwFftOp should construct");
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut spectrum = vec![0.0f32; input_data.len() * COMPLEX_BANDS as usize];
        let mut output = TileMut::<F32>::new(region, COMPLEX_BANDS, &mut spectrum);
        let mut state = fwfft.start();
        fwfft.process_region(&mut state, &input, &mut output);
        spectrum
    };

    let reconstructed = {
        let invfft = InvFftOp::<F32>::new(width, height).expect("InvFftOp should construct");
        let input = Tile::<F32>::new(region, COMPLEX_BANDS, &spectrum);
        let mut reconstructed = vec![0.0f32; input_data.len()];
        let mut output = TileMut::<F32>::new(region, 1, &mut reconstructed);
        let mut state = invfft.start();
        invfft.process_region(&mut state, &input, &mut output);
        reconstructed
    };

    let tolerance = f32::EPSILON * input_data.len() as f32;
    for (expected, actual) in input_data.iter().zip(reconstructed.iter()) {
        assert!((expected - actual).abs() <= tolerance);
    }
}
