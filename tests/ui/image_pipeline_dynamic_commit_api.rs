use viprs::{ImagePipeline, Input, Sink, U8};

enum Adjustment {
    Identity,
    Invert,
    Linear,
}

fn main() -> Result<(), viprs::ViprsError> {
    let input = Input::memory::<U8>(2, 1, 1, vec![0, 10])?;
    let adjustment = if std::env::args().len() > 1 {
        Adjustment::Invert
    } else {
        Adjustment::Linear
    };
    let _ = Adjustment::Identity;

    let pipeline = match adjustment {
        Adjustment::Identity => ImagePipeline::load(input).commit()?,
        Adjustment::Invert => ImagePipeline::load(input).invert()?.commit()?,
        Adjustment::Linear => ImagePipeline::load(input).linear(1.0, 0.0)?.commit()?,
    };

    let output = pipeline.raw_pixels().run_blocking(Sink::memory())?;

    assert_eq!(output.width(), 2);
    Ok(())
}
