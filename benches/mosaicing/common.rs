use viprs::{
    adapters::sources::memory::MemorySource,
    domain::{format::U8, image::Region, op::DynOperation},
    ports::source::ImageSource,
};

pub const SIZES: [u32; 3] = [512, 2048, 8192];
pub const BANDS: u32 = 1;
pub const BLEND_WIDTH: u32 = 64;

pub struct BenchBuffers {
    pub reference: Vec<u8>,
    pub secondary: Vec<u8>,
    pub output: Vec<u8>,
    pub input_regions: [Region; 2],
    pub output_region: Region,
}

pub fn prepare_two_source_bench(
    op: &dyn DynOperation,
    ref_width: u32,
    ref_height: u32,
    sec_width: u32,
    sec_height: u32,
    bands: u32,
) -> BenchBuffers {
    let reference_source = MemorySource::<U8>::new(
        ref_width,
        ref_height,
        bands,
        patterned_pixels(ref_width, ref_height, bands, 17),
    )
    .unwrap();
    let secondary_source = MemorySource::<U8>::new(
        sec_width,
        sec_height,
        bands,
        patterned_pixels(sec_width, sec_height, bands, 101),
    )
    .unwrap();

    let output_region = Region::new(
        0,
        0,
        op.output_width(ref_width),
        op.output_height(ref_height),
    );
    let input_regions = [
        op.required_input_region_slot(&output_region, 0),
        op.required_input_region_slot(&output_region, 1),
    ];

    let mut reference = vec![0u8; input_regions[0].pixel_count() * bands as usize];
    let mut secondary = vec![0u8; input_regions[1].pixel_count() * bands as usize];
    reference_source
        .read_region(input_regions[0], &mut reference)
        .unwrap();
    secondary_source
        .read_region(input_regions[1], &mut secondary)
        .unwrap();

    let output = vec![0u8; output_region.pixel_count() * bands as usize];

    BenchBuffers {
        reference,
        secondary,
        output,
        input_regions,
        output_region,
    }
}

fn patterned_pixels(width: u32, height: u32, bands: u32, seed: u8) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * bands as usize);
    for y in 0..height {
        for x in 0..width {
            for band in 0..bands {
                let value = x
                    .wrapping_mul(31)
                    .wrapping_add(y.wrapping_mul(17))
                    .wrapping_add(band.wrapping_mul(13))
                    .wrapping_add(u32::from(seed));
                pixels.push((value % 251) as u8 + 1);
            }
        }
    }
    pixels
}
