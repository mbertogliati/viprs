use super::support_core::*;

pub(crate) fn run_two_input_u8(
    op: &dyn DynOperation,
    reference: &[u8],
    secondary: &[u8],
    output_region: Region,
) -> Vec<u8> {
    let inputs = [reference, secondary];
    let input_regions = [
        op.required_input_region_slot(&output_region, 0),
        op.required_input_region_slot(&output_region, 1),
    ];
    let mut output = vec![0u8; output_region.pixel_count() * op.bands() as usize];
    let mut state = op.dyn_start();
    op.dyn_process_region_multi(
        state.as_mut(),
        &inputs,
        &mut output,
        &input_regions,
        output_region,
    );
    output
}

pub(crate) fn rgb_source(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 29 + y * 7 + 3) % 256) as u8);
            pixels.push(((x * 11 + y * 23 + 17) % 256) as u8);
            pixels.push(((x * 5 + y * 13 + 29) % 256) as u8);
        }
    }
    pixels
}

pub(crate) fn colour_lab_source() -> Vec<f32> {
    const PIXELS: [[f32; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [53.232_883, 80.109_33, 67.220_02],
        [87.737_04, -86.184_64, 83.181_17],
        [32.302_586, 79.196_66, -107.863_686],
        [60.0, -20.0, 30.0],
        [75.0, 10.0, -40.0],
        [25.0, 40.0, 20.0],
    ];

    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
    for idx in 0..(WIDTH * HEIGHT) as usize {
        pixels.extend_from_slice(&PIXELS[idx % PIXELS.len()]);
    }
    pixels
}

pub(crate) fn colour_xyz_source() -> Vec<f32> {
    const PIXELS: [[f32; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [0.950_47, 1.0, 1.088_83],
        [0.412_456_4, 0.212_672_9, 0.019_333_9],
        [0.357_576_1, 0.715_152_2, 0.119_192],
        [0.180_437_5, 0.072_175, 0.950_304_1],
        [0.203_44, 0.214_04, 0.233_09],
        [0.538_01, 0.787_33, 0.131_78],
        [0.114_0, 0.082_0, 0.401_0],
    ];

    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
    for idx in 0..(WIDTH * HEIGHT) as usize {
        pixels.extend_from_slice(&PIXELS[idx % PIXELS.len()]);
    }
    pixels
}

pub(crate) fn colour_hsv_source() -> Vec<f32> {
    const PIXELS: [[f32; 3]; 8] = [
        [0.0, 1.0, 1.0],
        [120.0, 1.0, 1.0],
        [240.0, 1.0, 1.0],
        [60.0, 1.0, 1.0],
        [300.0, 1.0, 1.0],
        [0.0, 0.0, 0.5],
        [210.0, 0.5, 0.8],
        [330.0, 0.25, 0.4],
    ];

    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
    for idx in 0..(WIDTH * HEIGHT) as usize {
        pixels.extend_from_slice(&PIXELS[idx % PIXELS.len()]);
    }
    pixels
}

pub(crate) fn colour_srgb_hsv_source() -> Vec<u8> {
    const PIXELS: [[u8; 3]; 8] = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
        [128, 128, 128],
    ];

    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
    for idx in 0..(WIDTH * HEIGHT) as usize {
        pixels.extend_from_slice(&PIXELS[idx % PIXELS.len()]);
    }
    pixels
}

pub(crate) fn scale_f32_pixels(pixels: &[f32], factor: f32) -> Vec<f32> {
    pixels.iter().map(|value| value * factor).collect()
}

pub(crate) fn encode_vips_hsv_input(pixels: &[f32]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(pixels.len());
    for pixel in pixels.chunks_exact(3) {
        let hue = pixel[0].rem_euclid(360.0);
        encoded.push(((hue / 360.0) * 255.0).round() as u8);
        encoded.push((pixel[1].clamp(0.0, 1.0) * 255.0).round() as u8);
        encoded.push((pixel[2].clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    encoded
}

pub(crate) fn delta_e_known_pairs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    const PAIRS: [([f32; 3], [f32; 3]); 8] = [
        ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485]),
        ([50.0, 3.1571, -77.2803], [50.0, 0.0, -82.7485]),
        ([50.0, 2.8361, -74.02], [50.0, 0.0, -82.7485]),
        ([50.0, -1.3802, -84.2814], [50.0, 0.0, -82.7485]),
        ([50.0, -1.1848, -84.8006], [50.0, 0.0, -82.7485]),
        ([50.0, -0.9009, -85.5211], [50.0, 0.0, -82.7485]),
        ([50.0, 0.0, 0.0], [50.0, -1.0, 2.0]),
        ([50.0, 2.49, -0.001], [50.0, -2.49, 0.001]),
    ];

    let mut left = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
    let mut right = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
    let mut combined = Vec::with_capacity((WIDTH * HEIGHT * 6) as usize);

    for idx in 0..(WIDTH * HEIGHT) as usize {
        let (lhs, rhs) = PAIRS[idx % PAIRS.len()];
        left.extend_from_slice(&lhs);
        right.extend_from_slice(&rhs);
        combined.extend_from_slice(&lhs);
        combined.extend_from_slice(&rhs);
    }

    (left, right, combined)
}

pub(crate) fn run_bandmean_u8(
    source_pixels: &[u8],
    width: u32,
    height: u32,
    bands: u32,
) -> Vec<u8> {
    let region = Region::new(0, 0, width, height);
    let op = BandMean::<U8>::new(bands as usize);
    let input = viprs::Tile::<U8>::new(region, bands, source_pixels);
    let mut output = vec![0u8; (width * height) as usize];
    let mut output_tile = viprs::TileMut::<U8>::new(region, 1, &mut output);
    let mut state = op.start();
    op.process_region(&mut state, &input, &mut output_tile);
    output
}
