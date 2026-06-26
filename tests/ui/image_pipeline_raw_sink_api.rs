use viprs::{ImagePipeline, Input, Sink, U8};

fn main() -> Result<(), viprs::ViprsError> {
    let memory = ImagePipeline::load(Input::memory::<U8>(2, 1, 1, vec![1, 2])?)
        .raw_pixels()
        .run_blocking(Sink::memory())?;
    assert_eq!(memory.as_bytes(), &[1, 2]);

    let writer = Vec::<u8>::new();
    let writer_output = ImagePipeline::load(Input::memory::<U8>(2, 1, 1, vec![3, 4])?)
        .raw_pixels()
        .run_blocking(Sink::writer(writer))?;
    assert_eq!(writer_output.as_bytes(), &[3, 4]);

    let raw_output_path = std::env::temp_dir().join("trybuild-raw-sink-api.raw");
    let path_output = ImagePipeline::load(Input::memory::<U8>(2, 1, 1, vec![5, 6])?)
        .raw_pixels()
        .run_blocking(Sink::path(raw_output_path))?;
    assert_eq!(path_output.as_bytes(), &[5, 6]);

    Ok(())
}
