use std::io::Cursor;

use viprs::{ImagePipeline, Input, Sink};

fn main() -> Result<(), viprs::ViprsError> {
    let encoded = include_bytes!("../fixtures/images/sample.png").to_vec();

    let from_bytes = ImagePipeline::load(Input::bytes(encoded.clone())?)
        .raw_pixels()
        .run_blocking(Sink::memory())?;

    let from_reader = ImagePipeline::load(Input::reader(Cursor::new(encoded))?)
        .raw_pixels()
        .run_blocking(Sink::memory())?;

    assert_eq!(from_bytes.width(), from_reader.width());
    assert_eq!(from_bytes.height(), from_reader.height());
    assert_eq!(from_bytes.as_bytes(), from_reader.as_bytes());
    Ok(())
}
