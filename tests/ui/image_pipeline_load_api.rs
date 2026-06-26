use viprs::{ImagePipeline, Input, Sink, U8};

fn main() -> Result<(), viprs::ViprsError> {
    let output = ImagePipeline::load(Input::memory::<U8>(2, 1, 1, vec![0, 10])?)
        .invert()?
        .linear(1.0, 0.0)?
        .raw_pixels()
        .run_blocking(Sink::memory())?;

    assert_eq!(output.as_bytes(), &[255, 245]);
    Ok(())
}
