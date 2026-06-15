use viprs::prelude::*;

fn usage(program: &str) -> ! {
    eprintln!("Usage: {program} <input.jpg> <output.jpg> [width]");
    eprintln!("Example: cargo run --example thumbnail --features jpeg -- in.jpg out.jpg 400");
    std::process::exit(1);
}

fn parse_width(value: Option<String>) -> Result<u32, ViprsError> {
    match value {
        Some(raw) => raw.parse::<u32>().map_err(|err| {
            ViprsError::Codec(format!("thumbnail width must be a positive integer: {err}"))
        }),
        None => Ok(400),
    }
}

fn main() -> Result<(), ViprsError> {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "thumbnail".to_owned());
    let input = args.next().unwrap_or_else(|| usage(&program));
    let output = args.next().unwrap_or_else(|| usage(&program));
    let target_width = parse_width(args.next())?;

    ImageApi::open(&input)?
        .thumbnail(target_width)?
        .save(&output)?;
    println!("Saved thumbnail to {output}");
    Ok(())
}
