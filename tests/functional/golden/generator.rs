use super::support as golden;

use bytemuck::cast_slice;
use viprs::domain::ops::create::IdentityOp;
use viprs::{
    BandFormat, BlackSource, EyeOp, F32, F64, GaussmatOp, GaussmatPrecision, ImageSource, Op,
    Region, SinesOp, Tile, TileMut, TonelutOp, U8, U16,
};

const OUTPUT_PLACEHOLDER: &str = "{output}";

fn render_source_bytes<S: ImageSource>(source: &S) -> Vec<u8> {
    let region = Region::new(0, 0, source.width(), source.height());
    let sample_size = std::mem::size_of::<<<S as ImageSource>::Format as BandFormat>::Sample>();
    let mut output = vec![0u8; region.pixel_count() * source.bands() as usize * sample_size];
    source.read_region(region, &mut output).unwrap();
    output
}

fn render_op<F, O>(op: &O, width: u32, height: u32, bands: u32) -> Vec<F::Sample>
where
    F: BandFormat,
    F::Sample: Copy + Default,
    O: Op<Input = F, Output = F, State = ()>,
{
    let region = Region::new(0, 0, width, height);
    let len = region.pixel_count() * bands as usize;
    let input_data = vec![F::Sample::default(); len];
    let mut output_data = vec![F::Sample::default(); len];
    let input = Tile::<F>::new(region, bands, &input_data);
    let mut output = TileMut::<F>::new(region, bands, &mut output_data);
    op.process_region(&mut (), &input, &mut output);
    output_data
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn decode_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn decode_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn black_matches_libvips_and_is_all_zero() {
    let actual = render_source_bytes(&BlackSource::new(100, 100, 3));

    assert_eq!(actual.len(), 100 * 100 * 3);
    assert!(actual.iter().all(|sample| *sample == 0));

    if golden::skip_without_vips() {
        return;
    }

    golden::assert_golden_libvips(
        "create_black_libvips",
        "100x100_bands3",
        &actual,
        &["black", OUTPUT_PLACEHOLDER, "100", "100", "--bands", "3"],
    );
}

#[test]
fn gaussmat_matches_libvips_and_normalized_weights_sum_to_one() {
    let op = GaussmatOp::<F64>::new(3.0, 0.5)
        .unwrap()
        .with_precision(GaussmatPrecision::Float);
    let actual = render_op::<F64, _>(&op, op.width(), op.height(), 1);
    let normalized_sum: f64 = actual.iter().sum();
    let normalized = actual
        .iter()
        .map(|value| value / normalized_sum)
        .collect::<Vec<_>>();
    let expected_first_row = [
        0.011_297_249_358,
        0.014_914_547_131,
        0.017_619_455_556,
        0.018_626_015_316,
        0.017_619_455_556,
        0.014_914_547_131,
        0.011_297_249_358,
    ];

    assert!((normalized.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    for (got, want) in normalized[..expected_first_row.len()]
        .iter()
        .zip(expected_first_row)
    {
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }

    if golden::skip_without_vips() {
        return;
    }

    let expected = decode_f64(&golden::generate_vips_golden(
        "create_gaussmat_libvips",
        "sigma3_min0_5_float",
        &[
            "gaussmat",
            OUTPUT_PLACEHOLDER,
            "3",
            "0.5",
            "--precision",
            "float",
        ],
    ));
    assert_eq!(actual.len(), expected.len());
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "gaussmat mismatch at {index}: got {got}, want {want}"
        );
    }
}

#[test]
fn identity_matches_libvips_and_emits_byte_indices() {
    let op = IdentityOp::<U8>::new(false);
    let actual = render_op::<U8, _>(&op, op.width(), op.height(), 1);

    assert_eq!(actual.len(), 256);
    assert!(
        actual
            .iter()
            .enumerate()
            .all(|(index, sample)| *sample == index as u8)
    );

    if golden::skip_without_vips() {
        return;
    }

    golden::assert_golden_libvips(
        "create_identity_libvips",
        "uchar_default",
        cast_slice(&actual),
        &["identity", OUTPUT_PLACEHOLDER],
    );
}

#[test]
fn tonelut_matches_libvips_for_known_curve() {
    let op = TonelutOp::<U16>::new(255, 255, 0.0, 100.0, 0.2, 0.5, 0.8, 12.0, -8.0, 16.0).unwrap();
    let actual = render_op::<U16, _>(&op, op.width(), op.height(), 1);

    assert_eq!(actual[0], 0);
    assert_eq!(actual[16], 23);
    assert_eq!(actual[64], 90);
    assert_eq!(actual[128], 107);
    assert_eq!(actual[192], 228);
    assert_eq!(actual[255], 254);

    if golden::skip_without_vips() {
        return;
    }

    let expected = decode_u16(&golden::generate_vips_golden(
        "create_tonelut_libvips",
        "custom_curve_u16",
        &[
            "tonelut",
            OUTPUT_PLACEHOLDER,
            "--in-max=255",
            "--out-max=255",
            "--Lb=0",
            "--Lw=100",
            "--Ps=0.2",
            "--Pm=0.5",
            "--Ph=0.8",
            "--S=12",
            "--M=-8",
            "--H=16",
        ],
    ));
    assert_eq!(actual, expected);
}

#[test]
fn sines_matches_libvips_for_known_frequencies() {
    let op = SinesOp::<F32>::with_frequencies(64, 64, 0.25, 0.0);
    let actual = render_op::<F32, _>(&op, 64, 64, 1);
    let expected_prefix = [
        1.0,
        0.999_698_8,
        0.998_795_45,
        0.997_290_43,
        0.995_184_7,
        0.992_479_56,
        0.989_176_5,
        0.985_277_65,
    ];

    for (got, want) in actual
        .iter()
        .take(expected_prefix.len())
        .zip(expected_prefix)
    {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }

    if golden::skip_without_vips() {
        return;
    }

    let expected = decode_f32(&golden::generate_vips_golden(
        "create_sines_libvips",
        "64x64_h0_25_v0",
        &[
            "sines",
            OUTPUT_PLACEHOLDER,
            "64",
            "64",
            "--hfreq",
            "0.25",
            "--vfreq",
            "0.0",
        ],
    ));
    assert_eq!(actual.len(), expected.len());
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "sines mismatch at {index}: got {got}, want {want}"
        );
    }
}

#[test]
fn eye_matches_libvips_and_preserves_first_column_ramp() {
    let op = EyeOp::<F32>::new(64, 64, 0.5).unwrap();
    let actual = render_op::<F32, _>(&op, 64, 64, 1);
    let max_y_sq = 63.0_f32 * 63.0_f32;

    assert!((actual[0] - 0.0).abs() < 1e-6);
    assert!((actual[64] - (1.0 / max_y_sq)).abs() < 1e-6);
    assert!((actual[63 * 64] - 1.0).abs() < 1e-6);

    if golden::skip_without_vips() {
        return;
    }

    let expected = decode_f32(&golden::generate_vips_golden(
        "create_eye_libvips",
        "64x64_factor0_5",
        &["eye", OUTPUT_PLACEHOLDER, "64", "64", "--factor", "0.5"],
    ));
    assert_eq!(actual.len(), expected.len());
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 4e-6,
            "eye mismatch at {index}: got {got}, want {want}"
        );
    }
}
