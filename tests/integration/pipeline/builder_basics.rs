//! Integration tests for the public `ImagePipeline` operation DSL.
//!
//! These tests exercise the first-class public vocabulary with in-memory inputs
//! and an explicit raw-pixel output contract.

use viprs::{BandFormatId, Format, ImagePipeline, Input, Sink, U8};

fn run_u8_pipeline(
    pixels: Vec<u8>,
    configure: impl FnOnce(ImagePipeline) -> Result<ImagePipeline, viprs::BuildError>,
) -> Vec<u8> {
    let input = Input::memory::<U8>(4, 4, 1, pixels).unwrap();
    let output = configure(ImagePipeline::from_input(input))
        .unwrap()
        .raw_pixels()
        .run_blocking(Sink::memory())
        .unwrap();

    assert_eq!(output.width(), 4);
    assert_eq!(output.height(), 4);
    assert_eq!(output.bands(), 1);
    assert_eq!(output.format(), Format::U8);
    output.as_bytes().to_vec()
}

#[test]
fn memory_input_raw_pixels_end_to_end() {
    let output =
        ImagePipeline::from_input(Input::memory::<U8>(4, 4, 1, (0u8..16).collect()).unwrap())
            .raw_pixels()
            .run_blocking(Sink::memory())
            .unwrap();

    assert_eq!(output.width(), 4);
    assert_eq!(output.height(), 4);
    assert_eq!(output.bands(), 1);
    assert_eq!(output.format(), Format::U8);
    assert_eq!(output.as_bytes(), &(0u8..16).collect::<Vec<_>>());
}

#[test]
fn fluent_invert_end_to_end() {
    let output = run_u8_pipeline(vec![0u8; 16], |pipeline| pipeline.invert());

    assert!(
        output.iter().all(|&sample| sample == 255),
        "Invert(0) must be 255, got: {output:?}"
    );
}

#[test]
fn fluent_linear_end_to_end() {
    let input = Input::memory::<viprs::F32>(4, 1, 1, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let output = ImagePipeline::from_input(input)
        .linear(3.0, 1.0)
        .unwrap()
        .raw_pixels()
        .run_blocking(Sink::memory())
        .unwrap();

    let floats: &[f32] = bytemuck::cast_slice(output.as_bytes());
    assert_eq!(floats, &[4.0, 7.0, 10.0, 13.0]);
}

#[test]
fn fluent_cast_u8_to_f32_end_to_end() {
    let input = [0u8, 64, 127, 255];
    let expected = [0.0f32, 64.0 / 255.0, 127.0 / 255.0, 1.0];
    let output = ImagePipeline::from_input(Input::memory::<U8>(4, 1, 1, input.to_vec()).unwrap())
        .cast(BandFormatId::F32)
        .unwrap()
        .raw_pixels()
        .run_blocking(Sink::memory())
        .unwrap();

    assert_eq!(output.format(), Format::F32);
    let floats: &[f32] = bytemuck::cast_slice(output.as_bytes());
    for (index, (&got, &expected)) in floats.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-6,
            "pixel {index}: expected {expected}, got {got}"
        );
    }
}

#[test]
fn chained_invert_twice_is_identity() {
    let input: Vec<u8> = (0u8..16).collect();
    let output = run_u8_pipeline(input.clone(), |pipeline| pipeline.invert()?.invert());

    assert_eq!(output, input, "Double-invert must be identity");
}
