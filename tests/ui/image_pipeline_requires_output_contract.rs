use viprs::{ImagePipeline, Input, Sink, U8};

fn main() -> Result<(), viprs::ViprsError> {
    let pipeline = ImagePipeline::from_input(Input::memory::<U8>(1, 1, 1, vec![0])?);
    let _output = pipeline.run_blocking(Sink::memory())?;
    Ok(())
}
