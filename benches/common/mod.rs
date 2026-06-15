use criterion::{BenchmarkId, Criterion, black_box};
use viprs::domain::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::Op,
};

pub const STANDARD_SIZES: [u32; 3] = [512, 2048, 8192];

pub fn tile_region(hint: DemandHint, size: u32) -> Region {
    Region::new(0, 0, hint.tile_width(size), hint.tile_height(size, size))
}

pub fn sample_count(region: Region, bands: u32) -> usize {
    region.pixel_count() * bands as usize
}

pub fn direct_tile_regions<O: Op>(op: &O, size: u32) -> (Region, Region) {
    let output = tile_region(op.demand_hint(), size);
    let input = op.required_input_region(&output);
    (input, output)
}

pub fn full_image_regions<O: Op>(op: &O, size: u32) -> (Region, Region) {
    let output = Region::new(0, 0, size, size);
    let input = op.required_input_region(&output);
    (input, output)
}

pub fn colour_convert_tile_regions<C, FromCs, ToCs>(converter: &C, size: u32) -> (Region, Region)
where
    C: viprs::domain::colour::ColourConvert<FromCs, ToCs>,
    FromCs: viprs::domain::colorspace::Colorspace,
    ToCs: viprs::domain::colorspace::Colorspace,
{
    let output = tile_region(converter.demand_hint(), size);
    let input = converter.required_input_region(&output);
    (input, output)
}

pub fn bench_direct_op_with_regions<O, In, Out, MakeOp, MakeRegions, MakeInput>(
    c: &mut Criterion,
    group_name: &str,
    input_bands: u32,
    output_bands: u32,
    make_op: MakeOp,
    make_regions: MakeRegions,
    make_input: MakeInput,
) where
    O: Op<Input = In, Output = Out>,
    In: BandFormat,
    Out: BandFormat,
    In::Sample: Copy,
    Out::Sample: Copy + Default,
    MakeOp: Fn() -> O,
    MakeRegions: Fn(&O, u32) -> (Region, Region),
    MakeInput: Fn(usize) -> Vec<In::Sample>,
{
    let mut group = c.benchmark_group(group_name);

    for &size in &STANDARD_SIZES {
        let op = make_op();
        let (input_region, output_region) = make_regions(&op, size);
        let input = make_input(sample_count(input_region, input_bands));
        let mut output = vec![Out::Sample::default(); sample_count(output_region, output_bands)];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let input_tile = Tile::<In>::new(input_region, input_bands, &input);
                let mut output_tile = TileMut::<Out>::new(output_region, output_bands, &mut output);
                let mut state = op.start();
                op.process_region(&mut state, &input_tile, &mut output_tile);
                black_box(&output);
            });
        });
    }

    group.finish();
}

pub fn bench_colour_convert_with_regions<
    C,
    FromCs,
    ToCs,
    In,
    Out,
    MakeConverter,
    MakeRegions,
    MakeInput,
>(
    c: &mut Criterion,
    group_name: &str,
    input_bands: u32,
    output_bands: u32,
    make_converter: MakeConverter,
    make_regions: MakeRegions,
    make_input: MakeInput,
) where
    C: viprs::domain::colour::ColourConvert<FromCs, ToCs, InputFormat = In, OutputFormat = Out>,
    FromCs: viprs::domain::colorspace::Colorspace,
    ToCs: viprs::domain::colorspace::Colorspace,
    In: BandFormat,
    Out: BandFormat,
    In::Sample: Copy,
    Out::Sample: Copy + Default,
    MakeConverter: Fn() -> C,
    MakeRegions: Fn(&C, u32) -> (Region, Region),
    MakeInput: Fn(usize) -> Vec<In::Sample>,
{
    let mut group = c.benchmark_group(group_name);

    for &size in &STANDARD_SIZES {
        let converter = make_converter();
        let (input_region, output_region) = make_regions(&converter, size);
        let input = make_input(sample_count(input_region, input_bands));
        let mut output = vec![Out::Sample::default(); sample_count(output_region, output_bands)];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let input_tile = Tile::<In>::new(input_region, input_bands, &input);
                let mut output_tile = TileMut::<Out>::new(output_region, output_bands, &mut output);
                let mut state = converter.start();
                converter.convert_region(&mut state, &input_tile, &mut output_tile);
                black_box(&output);
            });
        });
    }

    group.finish();
}
