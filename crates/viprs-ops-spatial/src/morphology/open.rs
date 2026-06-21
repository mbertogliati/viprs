//! Binary morphological opening — erosion followed by dilation.

use crate::morphology::{
    dilate::{Dilate, DilateState},
    erode::{Erode, ErodeState},
};

use viprs_core::{
    error::ViprsError,
    format::U8,
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

/// Binary morphological opening with a flat structuring element.
///
/// This applies the libvips-style erosion kernel followed by the matching dilation
/// kernel using the same `{0, 128, 255}` mask semantics.
pub struct Open {
    radius_x: u32,
    radius_y: u32,
    erode: Erode,
    dilate: Dilate,
}

/// Represents an open state.
pub struct OpenState {
    intermediate: Vec<u8>,
    erode_state: ErodeState,
    dilate_state: DilateState,
}

impl Open {
    /// Creates a new `Open`.
    #[allow(clippy::needless_pass_by_value)]
    // REASON: public API stability for morphology builders that already own the mask.
    pub fn new(mask: Vec<Vec<u8>>) -> Result<Self, &'static str> {
        if mask.is_empty() {
            return Err("Open: mask must not be empty");
        }
        let mask_h = mask.len() as u32;
        let mask_w = mask[0].len() as u32;
        if mask_w == 0 {
            return Err("Open: mask rows must not be empty");
        }
        for row in &mask {
            if row.len() as u32 != mask_w {
                return Err("Open: mask must be rectangular");
            }
            for &value in row {
                if value != 0 && value != 128 && value != 255 {
                    return Err("Open: mask values must be 0, 128, or 255");
                }
            }
        }

        let erode = Erode::new(mask.clone())?;
        let dilate = Dilate::new(mask)?;

        Ok(Self {
            radius_x: mask_w / 2,
            radius_y: mask_h / 2,
            erode,
            dilate,
        })
    }

    /// Returns or performs rect.
    pub fn rect(n: u32) -> Result<Self, &'static str> {
        if n == 0 {
            return Err("Open::rect: size must be >= 1");
        }
        let row = vec![255u8; n as usize];
        Self::new(vec![row; n as usize])
    }

    #[inline]
    fn checked_intermediate_len(region: Region, bands: u32) -> Result<usize, ViprsError> {
        region
            .checked_pixel_count()
            .and_then(|n| n.checked_mul(bands as usize))
            .ok_or_else(|| ViprsError::ImageTooLarge {
                width: region.width,
                height: region.height,
                bands,
                bytes: u128::from(region.width) * u128::from(region.height) * u128::from(bands),
                limit_bytes: usize::MAX as u128,
                details: "open intermediate scratch exceeds addressable memory",
            })
    }
}

impl Op for Open {
    type Input = U8;
    type Output = U8;
    type State = OpenState;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - (2 * self.radius_x) as i32,
            output.y - (2 * self.radius_y) as i32,
            output.width + 4 * self.radius_x,
            output.height + 4 * self.radius_y,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 4 * self.radius_x,
            input_tile_h: tile_h + 4 * self.radius_y,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {
        OpenState {
            intermediate: Vec::new(),
            erode_state: self.erode.start(),
            dilate_state: self.dilate.start(),
        }
    }

    fn validate_region_contract(
        &self,
        input_region: Region,
        input_bands: u32,
        output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        let _ = (input_region, output_bands);
        let intermediate_region = Region::new(
            output_region.x - self.radius_x as i32,
            output_region.y - self.radius_y as i32,
            output_region.width + 2 * self.radius_x,
            output_region.height + 2 * self.radius_y,
        );
        Self::checked_intermediate_len(intermediate_region, input_bands).map(|_| ())
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<U8>, output: &mut TileMut<U8>) {
        let intermediate_region = Region::new(
            output.region.x - self.radius_x as i32,
            output.region.y - self.radius_y as i32,
            output.region.width + 2 * self.radius_x,
            output.region.height + 2 * self.radius_y,
        );
        debug_assert_eq!(
            input.region,
            self.erode.required_input_region(&intermediate_region)
        );
        debug_assert_eq!(
            intermediate_region,
            self.dilate.required_input_region(&output.region)
        );

        let Ok(intermediate_len) = Self::checked_intermediate_len(intermediate_region, input.bands)
        else {
            debug_assert!(false, "Open intermediate scratch overflow");
            return;
        };
        if state.intermediate.len() < intermediate_len {
            state.intermediate.resize(intermediate_len, 0);
        }

        {
            let mut intermediate = TileMut::<U8>::new(
                intermediate_region,
                input.bands,
                &mut state.intermediate[..intermediate_len],
            );
            self.erode
                .process_region(&mut state.erode_state, input, &mut intermediate);
        }

        {
            let intermediate = Tile::<U8>::new(
                intermediate_region,
                input.bands,
                &state.intermediate[..intermediate_len],
            );
            self.dilate
                .process_region(&mut state.dilate_state, &intermediate, output);
        }
    }
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;

    use proptest::prelude::*;
    use viprs_core::{
        error::ViprsError,
        image::{Tile, TileMut},
    };

    fn copy_extend(data: &[u8], width: u32, height: u32, pad_x: u32, pad_y: u32) -> Vec<u8> {
        let expanded_w = width + 2 * pad_x;
        let expanded_h = height + 2 * pad_y;
        let mut expanded = vec![0u8; (expanded_w * expanded_h) as usize];

        for y in 0..expanded_h {
            let src_y = (y.saturating_sub(pad_y)).min(height.saturating_sub(1)) as usize;
            for x in 0..expanded_w {
                let src_x = (x.saturating_sub(pad_x)).min(width.saturating_sub(1)) as usize;
                expanded[(y * expanded_w + x) as usize] = data[src_y * width as usize + src_x];
            }
        }

        expanded
    }

    fn run_full_image(op: &Open, width: u32, height: u32, data: &[u8]) -> Vec<u8> {
        let pad_x = 2 * op.radius_x;
        let pad_y = 2 * op.radius_y;
        let expanded = copy_extend(data, width, height, pad_x, pad_y);
        let input_region = Region::new(
            -(pad_x as i32),
            -(pad_y as i32),
            width + 2 * pad_x,
            height + 2 * pad_y,
        );
        let output_region = Region::new(0, 0, width, height);
        let input = Tile::<U8>::new(input_region, 1, &expanded);
        let mut output_data = vec![0u8; (width * height) as usize];
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn identity_single_element_mask() {
        let op = Open::new(vec![vec![255u8]]).unwrap();
        let input = vec![0, 255, 0, 255, 0, 255, 0, 255, 0];
        assert_eq!(run_full_image(&op, 3, 3, &input), input);
    }

    #[test]
    fn opening_removes_isolated_white_pixel() {
        let op = Open::rect(3).unwrap();
        let input = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(run_full_image(&op, 5, 5, &input), vec![0u8; 25]);
    }

    #[test]
    fn all_dont_care_open_produces_zero() {
        let op = Open::new(vec![vec![128u8, 128u8], vec![128u8, 128u8]]).unwrap();
        assert_eq!(run_full_image(&op, 1, 1, &[255]), vec![0]);
    }

    #[test]
    fn open_is_idempotent_on_binary_image() {
        let op = Open::rect(3).unwrap();
        let input = vec![
            0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 255, 255, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
        ];

        let once = run_full_image(&op, 5, 5, &input);
        let twice = run_full_image(&op, 5, 5, &once);

        assert_eq!(twice, once);
    }

    #[test]
    fn close_after_open_can_differ_from_original() {
        let open = Open::rect(3).unwrap();
        let close = crate::morphology::close::Close::rect(3).unwrap();
        let input = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        let opened = run_full_image(&open, 5, 5, &input);
        let closed = crate::morphology::close::tests::run_full_image(&close, 5, 5, &opened);
        assert_ne!(closed, input);
    }

    #[test]
    fn rejects_invalid_masks_and_zero_rect() {
        assert!(Open::new(vec![]).is_err());
        assert!(Open::new(vec![vec![]]).is_err());
        assert!(Open::new(vec![vec![255u8], vec![255u8, 255u8]]).is_err());
        assert!(Open::new(vec![vec![1u8]]).is_err());
        assert!(Open::rect(0).is_err());
    }

    #[test]
    fn open_declares_double_halo() {
        let op = Open::rect(3).unwrap();
        let output = Region::new(5, 7, 11, 13);
        assert_eq!(op.demand_hint(), DemandHint::SmallTile);
        assert_eq!(op.required_input_region(&output), Region::new(3, 5, 15, 17));
        assert_eq!(
            op.node_spec(11, 13),
            NodeSpec {
                input_tile_w: 15,
                input_tile_h: 17,
                output_tile_w: 11,
                output_tile_h: 13,
                coordinate_driven_source: None,
            }
        );
    }

    #[test]
    fn validate_region_contract_rejects_overflowing_intermediate_scratch() {
        let op = Open::new(vec![vec![255u8]]).unwrap();
        let huge = Region::new(0, 0, u32::MAX, u32::MAX);

        let err = op.validate_region_contract(huge, 2, huge, 2).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width: u32::MAX,
                height: u32::MAX,
                bands: 2,
                ..
            }
        ));
    }

    proptest! {
        #[test]
        fn one_by_one_open_is_identity(
            width in 1u32..=6,
            height in 1u32..=6,
            pixels in proptest::collection::vec(0u8..=255u8, 1..=36),
        ) {
            let len = (width * height) as usize;
            prop_assume!(pixels.len() >= len);
            let input = pixels[..len].to_vec();
            let op = Open::new(vec![vec![255u8]]).unwrap();
            prop_assert_eq!(run_full_image(&op, width, height, &input), input);
        }
    }
}
