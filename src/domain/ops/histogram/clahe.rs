use std::marker::PhantomData;

use crate::domain::{
    error::ViprsError,
    format::{BandFormat, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

/// Sliding-window local histogram equalization with optional contrast limiting.
///
/// This mirrors libvips `hist_local`: each output pixel is derived from the
/// cumulative histogram of a local window centered on that pixel.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::histogram::clahe::ClaheOp;
///
/// let op = ClaheOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ClaheOp<F: BandFormat> {
    /// Stores the `tile_width` value for this item.
    pub tile_width: u32,
    /// Stores the `tile_height` value for this item.
    pub tile_height: u32,
    /// Stores the `clip_limit` value for this item.
    pub clip_limit: f64,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> ClaheOp<F> {
    /// Creates a new `ClaheOp`.
    pub fn new(tile_width: u32, tile_height: u32, clip_limit: f64) -> Result<Self, ViprsError> {
        if tile_width == 0 || tile_height == 0 {
            return Err(ViprsError::Scheduler(
                "ClaheOp window dimensions must be greater than zero".into(),
            ));
        }
        if !clip_limit.is_finite() || clip_limit < 0.0 {
            return Err(ViprsError::Scheduler(
                "ClaheOp clip_limit must be finite and non-negative".into(),
            ));
        }

        Ok(Self {
            tile_width,
            tile_height,
            clip_limit,
            _phantom: PhantomData,
        })
    }
}

impl<F: BandFormat> ClaheOp<F> {
    #[inline(always)]
    const fn radius_x(&self) -> u32 {
        self.tile_width / 2
    }

    #[inline(always)]
    const fn radius_y(&self) -> u32 {
        self.tile_height / 2
    }

    #[inline(always)]
    fn effective_max_slope(&self) -> u32 {
        if self.clip_limit <= 1.0 {
            0
        } else {
            self.clip_limit.round() as u32
        }
    }
}

impl Op for ClaheOp<U8> {
    type Input = U8;
    type Output = U8;
    type State = Vec<[u32; 256]>;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::FatStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.radius_x() as i32,
            output.y - self.radius_y() as i32,
            output.width + 2 * self.radius_x(),
            output.height + 2 * self.radius_y(),
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius_x(),
            input_tile_h: tile_h + 2 * self.radius_y(),
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {
        Vec::new()
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<U8>, output: &mut TileMut<U8>) {
        let bands = input.bands as usize;
        if state.len() != bands {
            state.clear();
            state.resize(bands, [0u32; 256]);
        }

        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let win_w = self.tile_width as usize;
        let win_h = self.tile_height as usize;
        let radius_x = self.radius_x() as usize;
        let radius_y = self.radius_y() as usize;
        let max_slope = self.effective_max_slope();
        let window_area = u64::from(self.tile_width * self.tile_height);

        for oy in 0..out_h {
            for hist in state.iter_mut() {
                hist.fill(0);
            }

            for wy in 0..win_h {
                let row_base = (oy + wy) * in_w * bands;
                for wx in 0..win_w {
                    let pixel_base = row_base + wx * bands;
                    for (band, hist) in state.iter_mut().enumerate().take(bands) {
                        let sample = input.data[pixel_base + band] as usize;
                        hist[sample] += 1;
                    }
                }
            }

            for ox in 0..out_w {
                let center_base = ((oy + radius_y) * in_w + (ox + radius_x)) * bands;
                let out_base = (oy * out_w + ox) * bands;

                for (band, hist) in state.iter().enumerate().take(bands) {
                    let target = input.data[center_base + band] as usize;
                    output.data[out_base + band] =
                        equalize_local_bin(hist, target, max_slope, window_area);
                }

                if ox + 1 < out_w {
                    let left_x = ox;
                    let right_x = ox + win_w;
                    for wy in 0..win_h {
                        let row_base = (oy + wy) * in_w * bands;
                        let left_base = row_base + left_x * bands;
                        let right_base = row_base + right_x * bands;
                        for (band, hist) in state.iter_mut().enumerate().take(bands) {
                            let left = input.data[left_base + band] as usize;
                            let right = input.data[right_base + band] as usize;
                            hist[left] -= 1;
                            hist[right] += 1;
                        }
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn equalize_local_bin(hist: &[u32; 256], target: usize, max_slope: u32, window_area: u64) -> u8 {
    let mut sum = 0u64;

    if max_slope > 0 {
        let mut sum_over = 0u64;

        for &bin in hist.iter().take(target + 1) {
            if bin > max_slope {
                sum_over += u64::from(bin - max_slope);
                sum += u64::from(max_slope);
            } else {
                sum += u64::from(bin);
            }
        }

        for &bin in hist.iter().skip(target + 1) {
            if bin > max_slope {
                sum_over += u64::from(bin - max_slope);
            }
        }

        sum += ((target as u64) + 1) * sum_over / 256;
    } else {
        for &bin in hist.iter().take(target + 1) {
            sum += u64::from(bin);
        }
    }

    ((255 * sum) / window_area) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::Region;
    use proptest::prelude::*;

    fn mirror_index(index: i64, size: i64) -> usize {
        if size <= 1 {
            return 0;
        }

        let period = 2 * size - 2;
        let reflected = index.rem_euclid(period);
        let reflected = if reflected >= size {
            period - reflected
        } else {
            reflected
        };

        reflected as usize
    }

    fn expanded_input(
        data: &[u8],
        width: usize,
        height: usize,
        window_w: u32,
        window_h: u32,
    ) -> Vec<u8> {
        let radius_x = (window_w / 2) as i32;
        let radius_y = (window_h / 2) as i32;
        let out_w = width + 2 * radius_x as usize;
        let out_h = height + 2 * radius_y as usize;
        let mut expanded = vec![0u8; out_w * out_h];

        for y in 0..out_h {
            for x in 0..out_w {
                let src_x = mirror_index(i64::from(x as i32 - radius_x), width as i64);
                let src_y = mirror_index(i64::from(y as i32 - radius_y), height as i64);
                expanded[y * out_w + x] = data[src_y * width + src_x];
            }
        }

        expanded
    }

    fn naive_clahe(
        data: &[u8],
        width: usize,
        height: usize,
        window_w: u32,
        window_h: u32,
        clip_limit: f64,
    ) -> Vec<u8> {
        let radius_x = (window_w / 2) as i32;
        let radius_y = (window_h / 2) as i32;
        let max_slope = if clip_limit <= 1.0 {
            0
        } else {
            clip_limit.round() as u32
        };
        let window_area = u64::from(window_w) * u64::from(window_h);
        let mut output = vec![0u8; width * height];

        for y in 0..height {
            for x in 0..width {
                let mut hist = [0u32; 256];
                for wy in -radius_y..=(window_h as i32 - radius_y - 1) {
                    for wx in -radius_x..=(window_w as i32 - radius_x - 1) {
                        let src_x = mirror_index(i64::from(x as i32 + wx), width as i64);
                        let src_y = mirror_index(i64::from(y as i32 + wy), height as i64);
                        hist[data[src_y * width + src_x] as usize] += 1;
                    }
                }
                output[y * width + x] =
                    equalize_local_bin(&hist, data[y * width + x] as usize, max_slope, window_area);
            }
        }

        output
    }

    fn run_op(
        data: &[u8],
        width: usize,
        height: usize,
        window_w: u32,
        window_h: u32,
        clip_limit: f64,
    ) -> Vec<u8> {
        let op = ClaheOp::<U8>::new(window_w, window_h, clip_limit).unwrap();
        let input_region = Region::new(
            0,
            0,
            width as u32 + 2 * (window_w / 2),
            height as u32 + 2 * (window_h / 2),
        );
        let output_region = Region::new(0, 0, width as u32, height as u32);
        let expanded = expanded_input(data, width, height, window_w, window_h);
        let input = Tile::<U8>::new(input_region, 1, &expanded);
        let mut output_data = vec![0u8; width * height];
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn clahe_matches_naive_reference_without_clipping() {
        let data = vec![
            0u8, 32, 64, 96, 16, 48, 80, 112, 32, 64, 96, 128, 48, 80, 112, 144,
        ];
        let output = run_op(&data, 4, 4, 3, 3, 0.0);
        let expected = naive_clahe(&data, 4, 4, 3, 3, 0.0);
        assert_eq!(output, expected);
    }

    #[test]
    fn clahe_matches_naive_reference_with_contrast_limit() {
        let data = vec![
            0u8, 0, 255, 255, 0, 0, 255, 255, 16, 16, 240, 240, 32, 32, 224, 224,
        ];
        let output = run_op(&data, 4, 4, 3, 3, 3.0);
        let expected = naive_clahe(&data, 4, 4, 3, 3, 3.0);
        assert_eq!(output, expected);
    }

    #[test]
    fn clahe_top_left_border_uses_mirror_reflection() {
        let data = vec![0u8, 64, 128, 32, 96, 160, 48, 112, 176];
        let output = run_op(&data, 3, 3, 3, 3, 0.0);
        assert_eq!(output[0], 28);
    }

    #[test]
    fn clahe_constructor_and_metadata_validate_arguments() {
        assert!(ClaheOp::<U8>::new(0, 3, 0.0).is_err());
        assert!(ClaheOp::<U8>::new(3, 0, 0.0).is_err());
        assert!(ClaheOp::<U8>::new(3, 3, f64::NAN).is_err());
        assert!(ClaheOp::<U8>::new(3, 3, -0.5).is_err());

        let op = ClaheOp::<U8>::new(5, 3, 1.0).unwrap();
        let output = Region::new(10, 20, 4, 2);

        assert_eq!(op.demand_hint(), DemandHint::FatStrip);
        assert_eq!(op.required_input_region(&output), Region::new(8, 19, 8, 4));
        assert_eq!(op.node_spec(4, 2).input_tile_w, 8);
        assert_eq!(op.node_spec(4, 2).input_tile_h, 4);
    }

    #[test]
    fn clahe_process_region_resizes_state_to_match_band_count() {
        let op = ClaheOp::<U8>::new(3, 3, 2.0).unwrap();
        let input_region = Region::new(0, 0, 3, 3);
        let output_region = Region::new(0, 0, 1, 1);
        let input = Tile::<U8>::new(
            input_region,
            2,
            &[
                0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
            ],
        );
        let mut output_data = [0u8; 2];
        let mut output = TileMut::<U8>::new(output_region, 2, &mut output_data);
        let mut state = vec![[9u32; 256]];

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(state.len(), 2);
        assert!(output_data.iter().all(|&sample| sample > 0));
    }

    proptest! {
        #[test]
        fn clahe_matches_naive_reference_prop(data in proptest::collection::vec(0u8..=255u8, 16)) {
            let output = run_op(&data, 4, 4, 3, 3, 0.0);
            let expected = naive_clahe(&data, 4, 4, 3, 3, 0.0);
            prop_assert_eq!(output, expected);
        }

        #[test]
        fn clahe_uniform_images_remain_uniform(
            value in 0u8..=255u8,
            width in 1usize..=6usize,
            height in 1usize..=6usize,
            clip_limit in prop_oneof![Just(0.0f64), Just(3.0f64)],
        ) {
            let data = vec![value; width * height];
            let output = run_op(&data, width, height, 3, 3, clip_limit);
            let expected = naive_clahe(&data, width, height, 3, 3, clip_limit);

            prop_assert_eq!(&output, &expected);
            prop_assert!(output.iter().all(|&sample| sample == output[0]));
        }

        #[test]
        fn clahe_large_window_matches_naive_reference_prop(
            width in 1usize..=4usize,
            height in 1usize..=4usize,
            data in proptest::collection::vec(0u8..=255u8, 1..=16),
        ) {
            let len = width * height;
            prop_assume!(data.len() >= len);
            let data = data[..len].to_vec();

            let output = run_op(&data, width, height, 7, 7, 3.0);
            let expected = naive_clahe(&data, width, height, 7, 7, 3.0);

            prop_assert_eq!(output, expected);
        }
    }
}
