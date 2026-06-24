#![allow(clippy::manual_clamp)]
// REASON: the explicit branch form matches libvips reference math and keeps boundary handling obvious.

use std::marker::PhantomData;

use viprs_core::shared_ops::gauss_kernel::gaussian_kernel_1d as libvips_gaussian_kernel_1d;
use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Image, Region},
    op::{ViewBridge, ViewOp},
};

/// libvips-style smartcrop strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interesting {
    /// Uses the `None` variant of `Interesting`.
    None,
    /// Uses the `Centre` variant of `Interesting`.
    Centre,
    /// Uses the `Entropy` variant of `Interesting`.
    Entropy,
    #[default]
    /// Uses the `Attention` variant of `Interesting`.
    Attention,
    /// Uses the `Low` variant of `Interesting`.
    Low,
    /// Uses the `High` variant of `Interesting`.
    High,
    /// Uses the `All` variant of `Interesting`.
    All,
    /// Uses the `Specific` variant of `Interesting`.
    Specific {
        /// Horizontal factor associated with this condition.
        x: u32,
        /// Vertical factor associated with this condition.
        y: u32,
    },
}

/// Smartcrop view with a precomputed crop origin.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::smartcrop::SmartcropOp;
///
/// let op = SmartcropOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SmartcropOp<F: BandFormat> {
    /// Stores the `target_width` value for this item.
    pub target_width: u32,
    /// Stores the `target_height` value for this item.
    pub target_height: u32,
    source_width: u32,
    source_height: u32,
    crop_left: u32,
    crop_top: u32,
    interesting: Interesting,
    attention_x: u32,
    attention_y: u32,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> SmartcropOp<F> {
    #[must_use]
    /// Creates a new `SmartcropOp`.
    pub fn new(
        source_width: u32,
        source_height: u32,
        target_width: u32,
        target_height: u32,
    ) -> Self {
        Self {
            target_width,
            target_height,
            source_width,
            source_height,
            crop_left: 0,
            crop_top: 0,
            interesting: Interesting::default(),
            attention_x: 0,
            attention_y: 0,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs crop left.
    pub const fn crop_left(&self) -> u32 {
        self.crop_left
    }

    #[must_use]
    /// Returns or performs crop top.
    pub const fn crop_top(&self) -> u32 {
        self.crop_top
    }

    #[must_use]
    /// Returns or performs interesting.
    pub const fn interesting(&self) -> Interesting {
        self.interesting
    }

    #[must_use]
    /// Returns or performs attention x.
    pub const fn attention_x(&self) -> u32 {
        self.attention_x
    }

    #[must_use]
    /// Returns or performs attention y.
    pub const fn attention_y(&self) -> u32 {
        self.attention_y
    }
}

#[allow(private_bounds)] // SampleToBin is an internal conversion helper for built-in band formats.
impl<F> SmartcropOp<F>
where
    F: BandFormat,
    F::Sample: SampleToBin + Copy,
{
    #[must_use]
    /// Returns or performs analyze.
    pub fn analyze(image: &Image<F>, target_width: u32, target_height: u32) -> Self {
        Self::analyze_with_interesting(image, target_width, target_height, Interesting::Attention)
    }

    #[must_use]
    /// Returns or performs analyze with interesting.
    pub fn analyze_with_interesting(
        image: &Image<F>,
        target_width: u32,
        target_height: u32,
        interesting: Interesting,
    ) -> Self {
        let source_width = image.width();
        let source_height = image.height();
        let bands = image.bands() as usize;
        let bounded_width = target_width.min(source_width).max(1);
        let bounded_height = target_height.min(source_height).max(1);

        let (target_width, target_height, crop_left, crop_top, attention_x, attention_y) =
            match interesting {
                Interesting::None | Interesting::Low => (bounded_width, bounded_height, 0, 0, 0, 0),
                Interesting::Centre => (
                    bounded_width,
                    bounded_height,
                    (source_width - bounded_width) / 2,
                    (source_height - bounded_height) / 2,
                    0,
                    0,
                ),
                Interesting::High => (
                    bounded_width,
                    bounded_height,
                    source_width - bounded_width,
                    source_height - bounded_height,
                    0,
                    0,
                ),
                Interesting::All => (source_width, source_height, 0, 0, 0, 0),
                Interesting::Specific { x, y } => {
                    let crop_left = clamp_crop_origin(x, bounded_width, source_width);
                    let crop_top = clamp_crop_origin(y, bounded_height, source_height);
                    (bounded_width, bounded_height, crop_left, crop_top, x, y)
                }
                Interesting::Entropy => {
                    let reduced_w = source_width.min(64).max(1) as usize;
                    let reduced_h = source_height.min(64).max(1) as usize;
                    let gray = downsample_to_gray(
                        image.pixels(),
                        source_width as usize,
                        source_height as usize,
                        bands,
                        reduced_w,
                        reduced_h,
                    );
                    let (crop_left, crop_top) = entropy_crop(
                        &gray,
                        reduced_w,
                        reduced_h,
                        source_width,
                        source_height,
                        bounded_width,
                        bounded_height,
                    );
                    (bounded_width, bounded_height, crop_left, crop_top, 0, 0)
                }
                Interesting::Attention => {
                    let reduced_w = ATTENTION_REDUCED_SIZE;
                    let reduced_h = ATTENTION_REDUCED_SIZE;
                    let xyz = downsample_attention_xyz(
                        image.pixels(),
                        source_width as usize,
                        source_height as usize,
                        bands,
                        reduced_w,
                        reduced_h,
                    );
                    let (crop_left, crop_top, attention_x, attention_y) = attention_crop(
                        &xyz,
                        reduced_w,
                        reduced_h,
                        source_width,
                        source_height,
                        bounded_width,
                        bounded_height,
                    );
                    (
                        bounded_width,
                        bounded_height,
                        crop_left,
                        crop_top,
                        attention_x,
                        attention_y,
                    )
                }
            };

        Self {
            target_width,
            target_height,
            source_width,
            source_height,
            crop_left,
            crop_top,
            interesting,
            attention_x,
            attention_y,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs into bridge.
    pub const fn into_bridge(self, bands: u32) -> ViewBridge<Self> {
        ViewBridge::new(self, bands)
    }
}

impl<F: BandFormat> ViewOp for SmartcropOp<F> {
    type Format = F;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x + self.crop_left as i32,
            output.y + self.crop_top as i32,
            output.width,
            output.height,
        )
    }

    fn output_width(&self, _input_width: u32) -> u32 {
        self.target_width.min(self.source_width)
    }

    fn output_height(&self, _input_height: u32) -> u32 {
        self.target_height.min(self.source_height)
    }
}

fn clamp_crop_origin(centre: u32, target: u32, source: u32) -> u32 {
    centre.saturating_sub(target / 2).min(source - target)
}

const ATTENTION_REDUCED_SIZE: usize = 32;
const SKIN_VECTOR_OFFSET: [f64; 3] = [-0.78, -0.57, -0.44];
const LAB_WHITE_POINT: [f64; 3] = [95.047, 100.0, 108.883];

fn scaled_window(target: u32, reduced: usize, source: u32) -> usize {
    ((target as usize * reduced).div_ceil(source as usize)).clamp(1, reduced)
}

fn map_reduced_origin(origin: usize, reduced: usize, source: u32, target: u32) -> u32 {
    ((origin * source as usize) / reduced).min((source - target) as usize) as u32
}

fn map_reduced_position(index: usize, reduced: usize, source: u32) -> u32 {
    ((index * source as usize) / reduced).min(source.saturating_sub(1) as usize) as u32
}

fn downsample_to_gray<T: SampleToBin + Copy>(
    pixels: &[T],
    source_width: usize,
    source_height: usize,
    bands: usize,
    reduced_w: usize,
    reduced_h: usize,
) -> Vec<u8> {
    let mut gray = vec![0u8; reduced_w * reduced_h];
    for y in 0..reduced_h {
        let src_y = y * source_height / reduced_h;
        for x in 0..reduced_w {
            let src_x = x * source_width / reduced_w;
            let base = (src_y * source_width + src_x) * bands;
            let mut sum = 0u32;
            for band in 0..bands {
                sum += u32::from(pixels[base + band].to_bin());
            }
            gray[y * reduced_w + x] = (sum / bands as u32) as u8;
        }
    }
    gray
}

fn downsample_attention_xyz<T: SampleToBin + Copy>(
    pixels: &[T],
    source_width: usize,
    source_height: usize,
    bands: usize,
    reduced_w: usize,
    reduced_h: usize,
) -> Vec<[f64; 3]> {
    let mut xyz = vec![[0.0; 3]; reduced_w * reduced_h];
    let x_scale = source_width as f64 / reduced_w as f64;
    let y_scale = source_height as f64 / reduced_h as f64;

    for y in 0..reduced_h {
        for x in 0..reduced_w {
            let idx = y * reduced_w + x;
            let src_x = (x as f64 + 0.5)
                .mul_add(x_scale, -0.5)
                .clamp(0.0, source_width as f64 - 1.0);
            let src_y = (y as f64 + 0.5)
                .mul_add(y_scale, -0.5)
                .clamp(0.0, source_height as f64 - 1.0);
            xyz[idx] = sample_xyz_pixel(pixels, source_width, source_height, bands, src_x, src_y);
        }
    }

    xyz
}

fn entropy_crop(
    gray: &[u8],
    reduced_w: usize,
    reduced_h: usize,
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
) -> (u32, u32) {
    let target_w = scaled_window(target_width, reduced_w, source_width);
    let target_h = scaled_window(target_height, reduced_h, source_height);
    let mut left = 0usize;
    let mut top = 0usize;
    let mut width = reduced_w;
    let mut height = reduced_h;
    let max_slice = ((reduced_w.saturating_sub(target_w)).div_ceil(8))
        .max((reduced_h.saturating_sub(target_h)).div_ceil(8))
        .max(1);

    while width > target_w || height > target_h {
        let slice_width = width.saturating_sub(target_w).min(max_slice);
        let slice_height = height.saturating_sub(target_h).min(max_slice);

        if slice_width > 0 {
            let left_score = window_entropy(gray, reduced_w, left, top, slice_width, height);
            let right_score = window_entropy(
                gray,
                reduced_w,
                left + width - slice_width,
                top,
                slice_width,
                height,
            );
            width -= slice_width;
            if left_score < right_score {
                left += slice_width;
            }
        }

        if slice_height > 0 {
            let top_score = window_entropy(gray, reduced_w, left, top, width, slice_height);
            let bottom_score = window_entropy(
                gray,
                reduced_w,
                left,
                top + height - slice_height,
                width,
                slice_height,
            );
            height -= slice_height;
            if top_score < bottom_score {
                top += slice_height;
            }
        }
    }

    (
        map_reduced_origin(left, reduced_w, source_width, target_width),
        map_reduced_origin(top, reduced_h, source_height, target_height),
    )
}

fn attention_crop(
    xyz: &[[f64; 3]],
    reduced_w: usize,
    reduced_h: usize,
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
) -> (u32, u32, u32, u32) {
    let hscale = reduced_w as f64 / f64::from(source_width);
    let vscale = reduced_h as f64 / f64::from(source_height);
    let sigma = ((f64::from(target_width) * hscale).hypot(f64::from(target_height) * vscale)
        / 10.0)
        .max(1.0);
    let mut score = vec![0.0; reduced_w * reduced_h];

    for y in 0..reduced_h {
        for x in 0..reduced_w {
            let idx = y * reduced_w + x;
            let edge = laplacian_y(xyz, reduced_w, reduced_h, x, y).abs() * 5.0;
            score[idx] = edge + skin_score(xyz[idx]) + saturation_score(xyz[idx]);
        }
    }

    // Matches libvips smartcrop attention scoring: edge + skin + saturation,
    // blurred according to the requested crop extent before locating the max.
    let blurred = gaussian_blur(&score, reduced_w, reduced_h, sigma);
    let mut best_sum = f64::NEG_INFINITY;
    let mut best_x = 0usize;
    let mut best_y = 0usize;

    for y in 0..reduced_h {
        for x in 0..reduced_w {
            let value = blurred[y * reduced_w + x];
            if value > best_sum {
                best_sum = value;
                best_x = x;
                best_y = y;
            }
        }
    }

    let attention_x = map_reduced_position(best_x, reduced_w, source_width);
    let attention_y = map_reduced_position(best_y, reduced_h, source_height);

    (
        clamp_crop_origin(attention_x, target_width, source_width),
        clamp_crop_origin(attention_y, target_height, source_height),
        attention_x,
        attention_y,
    )
}

fn sample_xyz_pixel<T: SampleToBin + Copy>(
    pixels: &[T],
    source_width: usize,
    source_height: usize,
    bands: usize,
    x: f64,
    y: f64,
) -> [f64; 3] {
    let x0 = x.floor().clamp(0.0, source_width as f64 - 1.0) as usize;
    let y0 = y.floor().clamp(0.0, source_height as f64 - 1.0) as usize;
    let x1 = (x0 + 1).min(source_width - 1);
    let y1 = (y0 + 1).min(source_height - 1);
    let dx = (x - x0 as f64).clamp(0.0, 1.0);
    let dy = (y - y0 as f64).clamp(0.0, 1.0);

    let top_left = load_rgb_sample(pixels, source_width, bands, x0, y0);
    let top_right = load_rgb_sample(pixels, source_width, bands, x1, y0);
    let bottom_left = load_rgb_sample(pixels, source_width, bands, x0, y1);
    let bottom_right = load_rgb_sample(pixels, source_width, bands, x1, y1);

    let mut rgb = [0.0; 3];
    for channel in 0..3 {
        let top = top_right[channel].mul_add(dx, top_left[channel] * (1.0 - dx));
        let bottom = bottom_right[channel].mul_add(dx, bottom_left[channel] * (1.0 - dx));
        rgb[channel] = top * (1.0 - dy) + bottom * dy;
    }

    srgb_to_xyz(rgb)
}

fn load_rgb_sample<T: SampleToBin + Copy>(
    pixels: &[T],
    source_width: usize,
    bands: usize,
    x: usize,
    y: usize,
) -> [f64; 3] {
    let base = (y * source_width + x) * bands;
    let (mut red, mut green, mut blue) = if bands >= 3 {
        (
            f64::from(pixels[base].to_bin()),
            f64::from(pixels[base + 1].to_bin()),
            f64::from(pixels[base + 2].to_bin()),
        )
    } else {
        let value = f64::from(pixels[base].to_bin());
        (value, value, value)
    };

    let alpha = match bands {
        0 | 1 | 3 => 255.0,
        2 => f64::from(pixels[base + 1].to_bin()),
        _ => f64::from(pixels[base + 3].to_bin()),
    } / 255.0;

    red *= alpha;
    green *= alpha;
    blue *= alpha;

    [red, green, blue]
}

fn srgb_to_xyz(rgb: [f64; 3]) -> [f64; 3] {
    let red = srgb_to_linear(rgb[0] / 255.0);
    let green = srgb_to_linear(rgb[1] / 255.0);
    let blue = srgb_to_linear(rgb[2] / 255.0);

    [
        0.180_437_5f64.mul_add(blue, 0.357_576_1f64.mul_add(green, 0.412_456_4 * red)) * 100.0,
        0.072_175f64.mul_add(blue, 0.715_152_2f64.mul_add(green, 0.212_672_9 * red)) * 100.0,
        0.950_304_1f64.mul_add(blue, 0.119_192f64.mul_add(green, 0.019_333_9 * red)) * 100.0,
    ]
}

fn srgb_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn laplacian_y(xyz: &[[f64; 3]], width: usize, height: usize, x: usize, y: usize) -> f64 {
    let centre = xyz[y * width + x][1];
    let left = xyz[y * width + x.saturating_sub(1)][1];
    let right = xyz[y * width + (x + 1).min(width - 1)][1];
    let up = xyz[y.saturating_sub(1) * width + x][1];
    let down = xyz[(y + 1).min(height - 1) * width + x][1];
    4.0f64.mul_add(centre, -left) - right - up - down
}

fn skin_score(xyz: [f64; 3]) -> f64 {
    if xyz[1] <= 5.0 {
        return 0.0;
    }

    let magnitude = xyz[2]
        .mul_add(xyz[2], xyz[1].mul_add(xyz[1], xyz[0] * xyz[0]))
        .sqrt();
    if magnitude <= f64::EPSILON {
        return 0.0;
    }

    let diff_x = xyz[0] / magnitude + SKIN_VECTOR_OFFSET[0];
    let diff_y = xyz[1] / magnitude + SKIN_VECTOR_OFFSET[1];
    let diff_z = xyz[2] / magnitude + SKIN_VECTOR_OFFSET[2];
    let distance = diff_z
        .mul_add(diff_z, diff_y.mul_add(diff_y, diff_x * diff_x))
        .sqrt();
    100.0f64.mul_add(-distance, 100.0)
}

fn saturation_score(xyz: [f64; 3]) -> f64 {
    if xyz[1] <= 5.0 {
        return 0.0;
    }

    let x = lab_f(xyz[0] / LAB_WHITE_POINT[0]);
    let y = lab_f(xyz[1] / LAB_WHITE_POINT[1]);
    500.0 * (x - y)
}

fn lab_f(value: f64) -> f64 {
    const EPSILON: f64 = 216.0 / 24_389.0;
    const KAPPA: f64 = 24_389.0 / 27.0;

    if value > EPSILON {
        value.cbrt()
    } else {
        KAPPA.mul_add(value, 16.0) / 116.0
    }
}

fn gaussian_blur(values: &[f64], width: usize, height: usize, sigma: f64) -> Vec<f64> {
    let kernel = libvips_gaussian_kernel_1d(sigma as f32);
    let radius = kernel.len() / 2;
    let mut horizontal = vec![0.0; width * height];
    let mut output = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut total = 0.0;
            for (kernel_idx, weight) in kernel.iter().enumerate() {
                let offset = kernel_idx as isize - radius as isize;
                let sample_x = (x as isize + offset).clamp(0, width as isize - 1) as usize;
                total += values[y * width + sample_x] * weight;
            }
            horizontal[y * width + x] = total;
        }
    }

    for y in 0..height {
        for x in 0..width {
            let mut total = 0.0;
            for (kernel_idx, weight) in kernel.iter().enumerate() {
                let offset = kernel_idx as isize - radius as isize;
                let sample_y = (y as isize + offset).clamp(0, height as isize - 1) as usize;
                total += horizontal[sample_y * width + x] * weight;
            }
            output[y * width + x] = total;
        }
    }

    output
}

fn window_entropy(
    gray: &[u8],
    width: usize,
    x0: usize,
    y0: usize,
    win_w: usize,
    win_h: usize,
) -> f64 {
    let mut counts = [0u32; 256];
    for y in y0..y0 + win_h {
        let row = y * width;
        for x in x0..x0 + win_w {
            counts[gray[row + x] as usize] += 1;
        }
    }

    let total = (win_w * win_h) as f64;
    let mut entropy = 0.0;
    for count in counts {
        if count == 0 {
            continue;
        }
        let p = f64::from(count) / total;
        entropy = p.mul_add(-p.log2(), entropy);
    }
    entropy
}

trait SampleToBin {
    fn to_bin(self) -> u8;
}

impl SampleToBin for u8 {
    fn to_bin(self) -> u8 {
        self
    }
}

impl SampleToBin for u16 {
    fn to_bin(self) -> u8 {
        (self >> 8) as u8
    }
}

impl SampleToBin for i16 {
    fn to_bin(self) -> u8 {
        ((i32::from(self) - i32::from(Self::MIN)) >> 8) as u8
    }
}

impl SampleToBin for u32 {
    fn to_bin(self) -> u8 {
        (self >> 24) as u8
    }
}

impl SampleToBin for i32 {
    fn to_bin(self) -> u8 {
        ((i64::from(self) - i64::from(Self::MIN)) >> 24) as u8
    }
}

impl SampleToBin for f32 {
    fn to_bin(self) -> u8 {
        (self.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}

impl SampleToBin for f64 {
    fn to_bin(self) -> u8 {
        (self.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use std::{fs, path::Path, process::Command};

    use super::*;
    use viprs_core::format::U8;
    fn write_ppm_rgb(path: &Path, width: u32, height: u32, pixels: &[u8]) {
        let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
        bytes.extend_from_slice(pixels);
        fs::write(path, bytes).unwrap();
    }

    fn write_pam_rgba(path: &Path, width: u32, height: u32, pixels: &[u8]) {
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
        )
        .into_bytes();
        bytes.extend_from_slice(pixels);
        fs::write(path, bytes).unwrap();
    }

    fn attention_fixture(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = vec![0u8; width as usize * height as usize * 3];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                pixels[idx] = 14 + ((x * 3 + y * 2) % 5) as u8;
                pixels[idx + 1] = 12 + ((x * 5 + y) % 5) as u8;
                pixels[idx + 2] = 10 + ((x + y * 7) % 5) as u8;
            }
        }

        for y in 18..42 {
            for x in 56..82 {
                let idx = (y * width as usize + x) * 3;
                let border = x == 56 || x == 81 || y == 18 || y == 41 || x == 69 || y == 30;
                if border {
                    pixels[idx..idx + 3].copy_from_slice(&[8, 8, 8]);
                } else if (x + y) % 3 == 0 {
                    pixels[idx..idx + 3].copy_from_slice(&[230, 186, 150]);
                } else {
                    pixels[idx..idx + 3].copy_from_slice(&[36, 232, 242]);
                }
            }
        }

        pixels
    }

    fn attention_fixture_rgba(width: u32, height: u32) -> Vec<u8> {
        let rgb = attention_fixture(width, height);
        let mut pixels = vec![0u8; width as usize * height as usize * 4];

        for (rgba, rgb) in pixels.chunks_exact_mut(4).zip(rgb.chunks_exact(3)) {
            rgba[..3].copy_from_slice(rgb);
            rgba[3] = 255;
        }

        for y in 6..34 {
            for x in 4..30 {
                let idx = (y * width as usize + x) * 4;
                pixels[idx..idx + 4].copy_from_slice(&[255, 24, 248, 0]);
            }
        }

        pixels
    }

    fn parse_header_offset(header_text: &str, field: &str) -> u32 {
        header_text
            .lines()
            .find_map(|line| line.strip_prefix(field))
            .and_then(|value| value.parse::<i32>().ok())
            .map(i32::unsigned_abs)
            .expect("expected offset field in vipsheader output")
    }

    fn run_vips_smartcrop_offsets(
        input_path: &Path,
        crop_width: u32,
        crop_height: u32,
        stem: &str,
    ) -> Option<(u32, u32)> {
        let vips = Path::new("/opt/homebrew/bin/vips");
        let vipsheader = Path::new("/opt/homebrew/bin/vipsheader");
        if !vips.exists() || !vipsheader.exists() {
            return None;
        }

        let workdir = Path::new("target/smartcrop-golden");
        fs::create_dir_all(workdir).unwrap();
        let output_v_path = workdir.join(format!("{stem}-{crop_width}x{crop_height}.v"));
        let crop_width_arg = crop_width.to_string();
        let crop_height_arg = crop_height.to_string();

        let output = Command::new(vips)
            .args([
                "smartcrop",
                input_path.to_str().unwrap(),
                output_v_path.to_str().unwrap(),
                crop_width_arg.as_str(),
                crop_height_arg.as_str(),
                "--interesting",
                "attention",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "vips smartcrop failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let header = Command::new(vipsheader)
            .args(["-a", output_v_path.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            header.status.success(),
            "vipsheader failed: {}",
            String::from_utf8_lossy(&header.stderr)
        );
        let header_text = String::from_utf8_lossy(&header.stdout);
        let offsets = (
            parse_header_offset(&header_text, "xoffset: "),
            parse_header_offset(&header_text, "yoffset: "),
        );

        let _ = fs::remove_file(output_v_path);
        Some(offsets)
    }

    #[test]
    fn default_strategy_matches_libvips_attention_default() {
        let image = Image::<U8>::from_buffer(8, 8, 1, vec![0u8; 64]).unwrap();
        let op = SmartcropOp::analyze(&image, 4, 4);
        assert_eq!(op.interesting(), Interesting::Attention);
    }

    #[test]
    fn uniform_image_yields_valid_crop() {
        let image = Image::<U8>::from_buffer(8, 8, 1, vec![0u8; 64]).unwrap();
        let op = SmartcropOp::analyze(&image, 4, 4);
        assert!(op.crop_left() <= 4);
        assert!(op.crop_top() <= 4);
        assert_eq!(op.output_width(image.width()), 4);
        assert_eq!(op.output_height(image.height()), 4);
    }

    #[test]
    fn specific_interest_clips_crop_around_requested_point() {
        let image = Image::<U8>::from_buffer(10, 6, 1, vec![0u8; 60]).unwrap();
        let op = SmartcropOp::analyze_with_interesting(
            &image,
            4,
            4,
            Interesting::Specific { x: 9, y: 5 },
        );

        assert_eq!(op.crop_left(), 6);
        assert_eq!(op.crop_top(), 2);
        assert_eq!(op.attention_x(), 9);
        assert_eq!(op.attention_y(), 5);
    }

    #[test]
    fn all_interest_returns_full_image_extent() {
        let image = Image::<U8>::from_buffer(7, 5, 1, vec![0u8; 35]).unwrap();
        let op = SmartcropOp::analyze_with_interesting(&image, 3, 2, Interesting::All);
        assert_eq!(op.output_width(image.width()), 7);
        assert_eq!(op.output_height(image.height()), 5);
        assert_eq!(op.crop_left(), 0);
        assert_eq!(op.crop_top(), 0);
    }

    #[test]
    fn entropy_and_attention_select_different_regions() {
        let mut pixels = vec![0u8; 12 * 4 * 3];

        for y in 0..4 {
            for x in 0..4 {
                let idx = (y * 12 + x) * 3;
                if (x + y) % 2 == 0 {
                    pixels[idx..idx + 3].copy_from_slice(&[255, 0, 0]);
                } else {
                    pixels[idx..idx + 3].copy_from_slice(&[0, 255, 0]);
                }
            }
        }

        let entropy_patch = [
            0u8, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 255, 36, 84, 132, 180,
        ];
        for y in 0..4 {
            for x in 0..4 {
                let idx = (y * 12 + (x + 8)) * 3;
                let value = entropy_patch[y * 4 + x];
                pixels[idx..idx + 3].copy_from_slice(&[value, value, value]);
            }
        }

        let image = Image::<U8>::from_buffer(12, 4, 3, pixels).unwrap();
        let attention = SmartcropOp::analyze_with_interesting(&image, 4, 4, Interesting::Attention);
        let entropy = SmartcropOp::analyze_with_interesting(&image, 4, 4, Interesting::Entropy);

        assert!(attention.crop_left() < 4);
        assert!(entropy.crop_left() >= 4);
        assert_ne!(attention.crop_left(), entropy.crop_left());
    }

    #[test]
    fn entropy_crop_contains_high_entropy_region() {
        let mut pixels = vec![0u8; 8 * 8];
        let noisy_block = [
            0u8, 255, 32, 223, 64, 191, 96, 159, 128, 127, 160, 95, 192, 63, 224, 31,
        ];
        for y in 0..4 {
            for x in 0..4 {
                pixels[(y + 2) * 8 + (x + 3)] = noisy_block[y * 4 + x];
            }
        }

        let image = Image::<U8>::from_buffer(8, 8, 1, pixels.clone()).unwrap();
        let op = SmartcropOp::analyze_with_interesting(&image, 4, 4, Interesting::Entropy);

        assert!(op.crop_left() <= 6);
        assert!(op.crop_top() <= 4);
        assert!(op.crop_left() + 4 > 3);
        assert!(op.crop_top() + 4 > 2);
    }

    #[test]
    fn attention_crop_biases_towards_salient_patch() {
        let image = Image::<U8>::from_buffer(96, 64, 3, attention_fixture(96, 64)).unwrap();
        let op = SmartcropOp::analyze_with_interesting(&image, 20, 20, Interesting::Attention);

        assert!(op.crop_left() <= 82);
        assert!(op.crop_left() + 20 > 56);
        assert!(op.crop_top() <= 42);
        assert!(op.crop_top() + 20 > 18);
    }

    #[test]
    fn attention_crop_matches_vips_cli_when_available() {
        let vips = Path::new("/opt/homebrew/bin/vips");
        if !vips.exists() {
            return;
        }

        let width = 96u32;
        let height = 64u32;
        let crop_width = 24u32;
        let crop_height = 24u32;
        let workdir = Path::new("target/smartcrop-golden");
        fs::create_dir_all(workdir).unwrap();
        let input_path = workdir.join("attention-input.ppm");
        let pixels = attention_fixture(width, height);
        write_ppm_rgb(&input_path, width, height, &pixels);
        let (expected_left, expected_top) =
            run_vips_smartcrop_offsets(&input_path, crop_width, crop_height, "attention-rgb")
                .expect("vips should be available");
        let expected_attention_x = expected_left + crop_width / 2;
        let expected_attention_y = expected_top + crop_height / 2;

        let image = Image::<U8>::from_buffer(width, height, 3, pixels).unwrap();
        let actual = SmartcropOp::analyze_with_interesting(
            &image,
            crop_width,
            crop_height,
            Interesting::Attention,
        );

        assert_eq!(
            (actual.crop_left(), actual.crop_top()),
            (expected_left, expected_top)
        );
        assert_eq!(
            (actual.attention_x(), actual.attention_y()),
            (expected_attention_x, expected_attention_y)
        );

        let _ = fs::remove_file(input_path);
    }

    #[test]
    fn rgba_attention_matches_vips_cli_when_available() {
        let width = 96u32;
        let height = 64u32;
        let crop_width = 24u32;
        let crop_height = 24u32;
        let workdir = Path::new("target/smartcrop-golden");
        fs::create_dir_all(workdir).unwrap();
        let input_path = workdir.join("attention-rgba-input.pam");

        let pixels = attention_fixture_rgba(width, height);
        write_pam_rgba(&input_path, width, height, &pixels);

        let (expected_left, expected_top) = match run_vips_smartcrop_offsets(
            &input_path,
            crop_width,
            crop_height,
            "attention-rgba",
        ) {
            Some(offsets) => offsets,
            None => return,
        };
        let expected_attention_x = expected_left + crop_width / 2;
        let expected_attention_y = expected_top + crop_height / 2;

        let image = Image::<U8>::from_buffer(width, height, 4, pixels).unwrap();
        let actual = SmartcropOp::analyze_with_interesting(
            &image,
            crop_width,
            crop_height,
            Interesting::Attention,
        );

        assert_eq!(
            (actual.crop_left(), actual.crop_top()),
            (expected_left, expected_top)
        );
        assert_eq!(
            (actual.attention_x(), actual.attention_y()),
            (expected_attention_x, expected_attention_y)
        );

        let _ = fs::remove_file(input_path);
    }
}
