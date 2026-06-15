use std::{any::Any, marker::PhantomData};

use crate::{
    domain::op::{DynOperation, NodeSpec, Op, OperationBridge},
    domain::{
        error::BuildError,
        format::{BandFormat, BandFormatId},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Fill mode for pixels outside the embedded source image.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtendMode {
    /// Fill with the zero value of the sample type.
    Black,
    /// Fill with the maximum white value for the sample type.
    White,
    /// Fill with a caller-supplied background vector.
    ///
    /// A single value is expanded to every band. Otherwise the vector length
    /// must match the image band count.
    Background(Vec<f64>),
    /// Replicate the nearest source edge pixel.
    Copy,
    /// Backwards-compatible alias for [`ExtendMode::Copy`].
    Edge,
    /// Tile the source periodically.
    Repeat,
    /// Tile the source periodically with every other tile mirrored.
    Mirror,
}

/// libvips-style compass gravity for anchored embedding and cropping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gravity {
    #[default]
    /// Uses the `Centre` variant of `Gravity`.
    Centre,
    /// Uses the `North` variant of `Gravity`.
    North,
    /// Uses the `East` variant of `Gravity`.
    East,
    /// Uses the `South` variant of `Gravity`.
    South,
    /// Uses the `West` variant of `Gravity`.
    West,
    /// Uses the `NorthEast` variant of `Gravity`.
    NorthEast,
    /// Uses the `SouthEast` variant of `Gravity`.
    SouthEast,
    /// Uses the `SouthWest` variant of `Gravity`.
    SouthWest,
    /// Uses the `NorthWest` variant of `Gravity`.
    NorthWest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbedExtend {
    Black,
    White,
    Background,
    Copy,
    Repeat,
    Mirror,
}

/// Sample conversion support for libvips-style embed constants.
pub trait EmbedSample: Copy + bytemuck::Pod + 'static {
    /// Returns or performs zero value.
    fn zero_value() -> Self;
    /// Returns or performs white value.
    fn white_value() -> Self;
    /// Creates this value from background.
    fn from_background(value: f64) -> Self;
}

impl EmbedSample for u8 {
    #[inline(always)]
    fn zero_value() -> Self {
        0
    }

    #[inline(always)]
    fn white_value() -> Self {
        Self::MAX
    }

    #[inline(always)]
    fn from_background(value: f64) -> Self {
        value
            .round()
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl EmbedSample for u16 {
    #[inline(always)]
    fn zero_value() -> Self {
        0
    }

    #[inline(always)]
    fn white_value() -> Self {
        Self::MAX
    }

    #[inline(always)]
    fn from_background(value: f64) -> Self {
        value
            .round()
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl EmbedSample for i16 {
    #[inline(always)]
    fn zero_value() -> Self {
        0
    }

    #[inline(always)]
    fn white_value() -> Self {
        Self::MAX
    }

    #[inline(always)]
    fn from_background(value: f64) -> Self {
        value
            .round()
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl EmbedSample for u32 {
    #[inline(always)]
    fn zero_value() -> Self {
        0
    }

    #[inline(always)]
    fn white_value() -> Self {
        Self::MAX
    }

    #[inline(always)]
    fn from_background(value: f64) -> Self {
        value
            .round()
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl EmbedSample for i32 {
    #[inline(always)]
    fn zero_value() -> Self {
        0
    }

    #[inline(always)]
    fn white_value() -> Self {
        Self::MAX
    }

    #[inline(always)]
    fn from_background(value: f64) -> Self {
        value
            .round()
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl EmbedSample for f32 {
    #[inline(always)]
    fn zero_value() -> Self {
        0.0
    }

    #[inline(always)]
    fn white_value() -> Self {
        1.0
    }

    #[inline(always)]
    fn from_background(value: f64) -> Self {
        value as Self
    }
}

impl EmbedSample for f64 {
    #[inline(always)]
    fn zero_value() -> Self {
        0.0
    }

    #[inline(always)]
    fn white_value() -> Self {
        1.0
    }

    #[inline(always)]
    fn from_background(value: f64) -> Self {
        value
    }
}

/// Embed a source image into a larger canvas, matching libvips `embed` extend modes.
pub struct Embed<F: BandFormat>
where
    F::Sample: EmbedSample,
{
    dst_width: u32,
    dst_height: u32,
    x_off: i32,
    y_off: i32,
    src_width: u32,
    src_height: u32,
    extend: EmbedExtend,
    fill_pixel: Vec<F::Sample>,
    bands: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> Embed<F>
where
    F::Sample: EmbedSample,
{
    #[allow(clippy::too_many_arguments)]
    /// Returns or performs try with gravity.
    pub fn try_with_gravity(
        dst_width: u32,
        dst_height: u32,
        gravity: Gravity,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
        bands: u32,
    ) -> Result<Self, BuildError> {
        let (x_off, y_off) = gravity_offsets(gravity, src_width, src_height, dst_width, dst_height);
        Self::try_new(
            dst_width, dst_height, x_off, y_off, src_width, src_height, extend, bands,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Returns or performs try new.
    pub fn try_new(
        dst_width: u32,
        dst_height: u32,
        x_off: i32,
        y_off: i32,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
        bands: u32,
    ) -> Result<Self, BuildError> {
        validate_dimensions(dst_width, dst_height, src_width, src_height, bands)?;

        let requires_overlap = matches!(
            extend,
            ExtendMode::Black
                | ExtendMode::White
                | ExtendMode::Background(_)
                | ExtendMode::Copy
                | ExtendMode::Edge
        );
        if requires_overlap
            && !source_overlaps_canvas(x_off, y_off, src_width, src_height, dst_width, dst_height)
        {
            return Err(BuildError::InvalidEmbedParameters {
                message: "source rectangle must overlap the output canvas for black, white, background, and copy extend modes",
            });
        }

        let (extend, fill_pixel) = match extend {
            ExtendMode::Black => (
                EmbedExtend::Black,
                vec![F::Sample::zero_value(); bands as usize],
            ),
            ExtendMode::White => (
                EmbedExtend::White,
                vec![F::Sample::white_value(); bands as usize],
            ),
            ExtendMode::Background(values) => {
                let fill_pixel = convert_background::<F>(&values, bands)?;
                (EmbedExtend::Background, fill_pixel)
            }
            ExtendMode::Copy | ExtendMode::Edge => (
                EmbedExtend::Copy,
                vec![F::Sample::zero_value(); bands as usize],
            ),
            ExtendMode::Repeat => (
                EmbedExtend::Repeat,
                vec![F::Sample::zero_value(); bands as usize],
            ),
            ExtendMode::Mirror => (
                EmbedExtend::Mirror,
                vec![F::Sample::zero_value(); bands as usize],
            ),
        };

        Ok(Self {
            dst_width,
            dst_height,
            x_off,
            y_off,
            src_width,
            src_height,
            extend,
            fill_pixel,
            bands,
            _format: PhantomData,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Creates a new `Embed`.
    pub fn new(
        dst_width: u32,
        dst_height: u32,
        x_off: u32,
        y_off: u32,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
        bands: u32,
    ) -> Result<Self, BuildError> {
        Self::try_new(
            dst_width,
            dst_height,
            x_off as i32,
            y_off as i32,
            src_width,
            src_height,
            extend,
            bands,
        )
    }

    #[inline]
    const fn repeated_pixel_index(&self, canvas_x: i32, canvas_y: i32) -> usize {
        let src_x = (canvas_x - self.x_off).rem_euclid(self.src_width as i32) as usize;
        let src_y = (canvas_y - self.y_off).rem_euclid(self.src_height as i32) as usize;
        (src_y * self.src_width as usize + src_x) * self.bands as usize
    }

    #[inline]
    const fn mirrored_pixel_index(&self, canvas_x: i32, canvas_y: i32) -> usize {
        let src_x = mirror_coord(canvas_x - self.x_off, self.src_width);
        let src_y = mirror_coord(canvas_y - self.y_off, self.src_height);
        (src_y * self.src_width as usize + src_x) * self.bands as usize
    }

    #[inline]
    fn overlap_in_output(&self, output: Region) -> Option<(usize, usize, usize, usize)> {
        let left = output.x.max(self.x_off);
        let top = output.y.max(self.y_off);
        let right = (output.x + output.width as i32).min(self.x_off + self.src_width as i32);
        let bottom = (output.y + output.height as i32).min(self.y_off + self.src_height as i32);

        if left >= right || top >= bottom {
            return None;
        }

        Some((
            (left - output.x) as usize,
            (top - output.y) as usize,
            (right - left) as usize,
            (bottom - top) as usize,
        ))
    }

    #[inline]
    fn fill_output(&self, output: &mut [F::Sample]) {
        match self.extend {
            EmbedExtend::Black => output.fill(F::Sample::zero_value()),
            EmbedExtend::White => output.fill(F::Sample::white_value()),
            EmbedExtend::Background => fill_repeated_pixel(output, &self.fill_pixel),
            EmbedExtend::Copy | EmbedExtend::Repeat | EmbedExtend::Mirror => {}
        }
    }
}

#[inline]
fn fill_repeated_pixel<T: Copy>(output: &mut [T], pixel: &[T]) {
    if output.is_empty() {
        return;
    }

    debug_assert!(
        !pixel.is_empty(),
        "fill pixel must contain at least one sample"
    );
    debug_assert!(
        output.len().is_multiple_of(pixel.len()),
        "output must contain a whole number of pixels"
    );

    output[..pixel.len()].copy_from_slice(pixel);
    let mut filled = pixel.len();
    while filled < output.len() {
        let copy_len = filled.min(output.len() - filled);
        let (prefix, tail) = output.split_at_mut(filled);
        tail[..copy_len].copy_from_slice(&prefix[..copy_len]);
        filled += copy_len;
    }
}

impl<F: BandFormat> Op for Embed<F>
where
    F::Sample: EmbedSample,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        if matches!(self.extend, EmbedExtend::Repeat | EmbedExtend::Mirror) {
            // Repeat/mirror can wrap within one output tile; request the full source.
            Region::new(0, 0, self.src_width, self.src_height)
        } else {
            Region::new(
                output.x - self.x_off,
                output.y - self.y_off,
                output.width,
                output.height,
            )
        }
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        if matches!(self.extend, EmbedExtend::Repeat | EmbedExtend::Mirror) {
            NodeSpec {
                input_tile_w: self.src_width,
                input_tile_h: self.src_height,
                output_tile_w: tile_w,
                output_tile_h: tile_h,
                coordinate_driven_source: None,
            }
        } else {
            NodeSpec {
                input_tile_w: tile_w,
                input_tile_h: tile_h,
                output_tile_w: tile_w,
                output_tile_h: tile_h,
                coordinate_driven_source: None,
            }
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let bands = output.bands as usize;
        let output_width = output.region.width as usize;
        let input_width = input.region.width as usize;

        match self.extend {
            EmbedExtend::Black | EmbedExtend::White | EmbedExtend::Background => {
                self.fill_output(output.data);

                if let Some((left, top, width, height)) = self.overlap_in_output(output.region) {
                    let row_samples = width * bands;
                    for row in 0..height {
                        let src_start = ((top + row) * input_width + left) * bands;
                        let dst_start = ((top + row) * output_width + left) * bands;
                        output.data[dst_start..dst_start + row_samples]
                            .copy_from_slice(&input.data[src_start..src_start + row_samples]);
                    }
                }

                return;
            }
            EmbedExtend::Copy => {
                let Some((left, top, width, height)) = self.overlap_in_output(output.region) else {
                    return;
                };

                let row_samples = width * bands;
                for row in 0..height {
                    let src_start = ((top + row) * input_width + left) * bands;
                    let dst_start = ((top + row) * output_width + left) * bands;
                    output.data[dst_start..dst_start + row_samples]
                        .copy_from_slice(&input.data[src_start..src_start + row_samples]);

                    let row_start = (top + row) * output_width * bands;
                    let row_slice = &mut output.data[row_start..row_start + output_width * bands];
                    let (left_side, remainder) = row_slice.split_at_mut(left * bands);
                    let (middle, right_side) = remainder.split_at_mut(row_samples);

                    if !left_side.is_empty() {
                        fill_repeated_pixel(left_side, &middle[..bands]);
                    }
                    if !right_side.is_empty() {
                        fill_repeated_pixel(right_side, &middle[row_samples - bands..row_samples]);
                    }
                }

                let first_row_start = top * output_width * bands;
                let last_row_start = (top + height - 1) * output_width * bands;
                let row_samples = output_width * bands;

                if top > 0 {
                    let (prefix, rest) = output.data.split_at_mut(first_row_start);
                    let first_row = &rest[..row_samples];
                    for row in prefix.chunks_exact_mut(row_samples) {
                        row.copy_from_slice(first_row);
                    }
                }

                let bottom_start = (top + height) * row_samples;
                if bottom_start < output.data.len() {
                    let (prefix, suffix) = output.data.split_at_mut(bottom_start);
                    let last_row = &prefix[last_row_start..last_row_start + row_samples];
                    for row in suffix.chunks_exact_mut(row_samples) {
                        row.copy_from_slice(last_row);
                    }
                }

                return;
            }
            EmbedExtend::Repeat | EmbedExtend::Mirror => {}
        }

        for row in 0..output.region.height as usize {
            for col in 0..output.region.width as usize {
                let canvas_x = output.region.x + col as i32;
                let canvas_y = output.region.y + row as i32;
                let dst_idx = (row * output_width + col) * bands;

                let src_idx = match self.extend {
                    EmbedExtend::Repeat => self.repeated_pixel_index(canvas_x, canvas_y),
                    EmbedExtend::Mirror => self.mirrored_pixel_index(canvas_x, canvas_y),
                    EmbedExtend::Black
                    | EmbedExtend::White
                    | EmbedExtend::Background
                    | EmbedExtend::Copy => {
                        debug_assert!(
                            false,
                            "EmbedOp fast paths must handle non-repeating extend modes"
                        );
                        return;
                    }
                };

                output.data[dst_idx..dst_idx + bands]
                    .copy_from_slice(&input.data[src_idx..src_idx + bands]);
            }
        }
    }
}

pub(crate) struct EmbedBridge<F: BandFormat>
where
    F::Sample: EmbedSample,
{
    inner: OperationBridge<Embed<F>>,
}

impl<F: BandFormat> EmbedBridge<F>
where
    F::Sample: EmbedSample,
{
    #[allow(clippy::too_many_arguments)]
    pub fn try_with_gravity(
        dst_width: u32,
        dst_height: u32,
        gravity: Gravity,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
        bands: u32,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            inner: OperationBridge::new(
                Embed::try_with_gravity(
                    dst_width, dst_height, gravity, src_width, src_height, extend, bands,
                )?,
                bands,
            ),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        dst_width: u32,
        dst_height: u32,
        x_off: i32,
        y_off: i32,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
        bands: u32,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            inner: OperationBridge::new(
                Embed::try_new(
                    dst_width, dst_height, x_off, y_off, src_width, src_height, extend, bands,
                )?,
                bands,
            ),
        })
    }
}

impl<F: BandFormat> DynOperation for EmbedBridge<F>
where
    F::Sample: EmbedSample + Send,
{
    fn input_format(&self) -> BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> BandFormatId {
        self.inner.output_format()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    fn demand_hint(&self) -> DemandHint {
        self.inner.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        self.inner.op.dst_width
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        self.inner.op.dst_height
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn Any + Send> {
        self.inner.dyn_start_with_tile(tile_w, tile_h)
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        self.inner
            .dyn_process_region(state, input, output, input_region, output_region);
    }
}

const fn validate_dimensions(
    dst_width: u32,
    dst_height: u32,
    src_width: u32,
    src_height: u32,
    bands: u32,
) -> Result<(), BuildError> {
    if dst_width == 0 || dst_height == 0 || src_width == 0 || src_height == 0 {
        return Err(BuildError::InvalidEmbedParameters {
            message: "source and output dimensions must be greater than zero",
        });
    }
    if bands == 0 {
        return Err(BuildError::InvalidEmbedParameters {
            message: "band count must be greater than zero",
        });
    }
    Ok(())
}

#[must_use]
/// Returns or performs gravity offsets.
pub fn gravity_offsets(
    gravity: Gravity,
    image_w: u32,
    image_h: u32,
    target_w: u32,
    target_h: u32,
) -> (i32, i32) {
    let horizontal = i64::from(target_w) - i64::from(image_w);
    let vertical = i64::from(target_h) - i64::from(image_h);

    let x = match gravity {
        Gravity::Centre | Gravity::North | Gravity::South => horizontal / 2,
        Gravity::East | Gravity::NorthEast | Gravity::SouthEast => horizontal,
        Gravity::West | Gravity::NorthWest | Gravity::SouthWest => 0,
    };
    let y = match gravity {
        Gravity::Centre | Gravity::East | Gravity::West => vertical / 2,
        Gravity::South | Gravity::SouthEast | Gravity::SouthWest => vertical,
        Gravity::North | Gravity::NorthEast | Gravity::NorthWest => 0,
    };

    (i64_to_i32_saturating(x), i64_to_i32_saturating(y))
}

#[inline]
fn i64_to_i32_saturating(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn source_overlaps_canvas(
    x_off: i32,
    y_off: i32,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> bool {
    let left = i64::from(x_off);
    let top = i64::from(y_off);
    let right = left + i64::from(src_width);
    let bottom = top + i64::from(src_height);
    left < i64::from(dst_width) && top < i64::from(dst_height) && right > 0 && bottom > 0
}

fn convert_background<F: BandFormat>(
    values: &[f64],
    bands: u32,
) -> Result<Vec<F::Sample>, BuildError>
where
    F::Sample: EmbedSample,
{
    match values.len() {
        0 => Err(BuildError::InvalidEmbedParameters {
            message: "background must contain at least one value",
        }),
        1 => Ok(vec![F::Sample::from_background(values[0]); bands as usize]),
        len if len == bands as usize => Ok(values
            .iter()
            .map(|&value| F::Sample::from_background(value))
            .collect()),
        _ => Err(BuildError::InvalidEmbedParameters {
            message: "background must contain either one value or one value per band",
        }),
    }
}

#[inline]
const fn mirror_coord(coord: i32, limit: u32) -> usize {
    let period = (limit * 2) as i32;
    let wrapped = coord.rem_euclid(period);
    if wrapped < limit as i32 {
        wrapped as usize
    } else {
        (period - 1 - wrapped) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, op::DynOperation};
    use proptest::prelude::*;

    const ALL_GRAVITIES: [Gravity; 9] = [
        Gravity::Centre,
        Gravity::North,
        Gravity::East,
        Gravity::South,
        Gravity::West,
        Gravity::NorthEast,
        Gravity::SouthEast,
        Gravity::SouthWest,
        Gravity::NorthWest,
    ];

    fn op(
        dst_width: u32,
        dst_height: u32,
        x_off: i32,
        y_off: i32,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
        bands: u32,
    ) -> Embed<U8> {
        match Embed::<U8>::try_new(
            dst_width, dst_height, x_off, y_off, src_width, src_height, extend, bands,
        ) {
            Ok(op) => op,
            Err(error) => panic!("embed construction failed: {error}"),
        }
    }

    fn run_embed(
        op: &Embed<U8>,
        src_pixels: &[u8],
        input_region: Region,
        output_region: Region,
        bands: u32,
    ) -> Vec<u8> {
        let mut out = vec![0u8; output_region.pixel_count() * bands as usize];
        let input = Tile::<U8>::new(input_region, bands, src_pixels);
        let mut output = TileMut::<U8>::new(output_region, bands, &mut out);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out
    }

    #[test]
    fn required_input_region_subtracts_signed_offset() {
        let op = op(4, 4, 1, 2, 3, 2, ExtendMode::Black, 1);
        assert_eq!(
            op.required_input_region(&Region::new(1, 2, 2, 1)),
            Region::new(0, 0, 2, 1)
        );
    }

    #[test]
    fn repeat_and_mirror_request_full_source_tile() {
        let repeat = op(8, 8, -3, 2, 3, 2, ExtendMode::Repeat, 1);
        let mirror = op(8, 8, -3, 2, 3, 2, ExtendMode::Mirror, 1);
        let output = Region::new(4, 5, 2, 1);
        assert_eq!(
            repeat.required_input_region(&output),
            Region::new(0, 0, 3, 2)
        );
        assert_eq!(
            mirror.required_input_region(&output),
            Region::new(0, 0, 3, 2)
        );
    }

    #[test]
    fn bridge_reports_canvas_dimensions() {
        let bridge = match EmbedBridge::<U8>::try_new(10, 20, 1, 2, 4, 5, ExtendMode::Black, 3) {
            Ok(bridge) => bridge,
            Err(error) => panic!("bridge construction failed: {error}"),
        };
        assert_eq!(bridge.output_width(4), 10);
        assert_eq!(bridge.output_height(5), 20);
        assert_eq!(bridge.bands(), 3);
    }

    #[test]
    fn gravity_offsets_match_libvips_compass_positions_for_padding() {
        let expected = [
            (Gravity::Centre, (1, 1)),
            (Gravity::North, (1, 0)),
            (Gravity::East, (2, 1)),
            (Gravity::South, (1, 2)),
            (Gravity::West, (0, 1)),
            (Gravity::NorthEast, (2, 0)),
            (Gravity::SouthEast, (2, 2)),
            (Gravity::SouthWest, (0, 2)),
            (Gravity::NorthWest, (0, 0)),
        ];

        for (gravity, offsets) in expected {
            assert_eq!(gravity_offsets(gravity, 2, 1, 4, 3), offsets);
        }
    }

    #[test]
    fn gravity_offsets_match_libvips_compass_positions_for_cropping() {
        let expected = [
            (Gravity::Centre, (-1, -1)),
            (Gravity::North, (-1, 0)),
            (Gravity::East, (-2, -1)),
            (Gravity::South, (-1, -2)),
            (Gravity::West, (0, -1)),
            (Gravity::NorthEast, (-2, 0)),
            (Gravity::SouthEast, (-2, -2)),
            (Gravity::SouthWest, (0, -2)),
            (Gravity::NorthWest, (0, 0)),
        ];

        for (gravity, offsets) in expected {
            assert_eq!(gravity_offsets(gravity, 5, 4, 3, 2), offsets);
        }
    }

    #[test]
    fn embed_with_gravity_places_source_at_all_compass_positions() {
        let expected_positions = [
            (1_i32, 1_i32),
            (1, 0),
            (2, 1),
            (1, 2),
            (0, 1),
            (2, 0),
            (2, 2),
            (0, 2),
            (0, 0),
        ];

        for (gravity, (x, y)) in ALL_GRAVITIES.into_iter().zip(expected_positions) {
            let op =
                Embed::<U8>::try_with_gravity(3, 3, gravity, 1, 1, ExtendMode::Black, 1).unwrap();
            let mut input = vec![0_u8; 9];
            input[(y as usize) * 3 + x as usize] = 7;
            let result = run_embed(
                &op,
                &input,
                Region::new(-x, -y, 3, 3),
                Region::new(0, 0, 3, 3),
                1,
            );
            let mut expected = vec![0_u8; 9];
            expected[(y as usize) * 3 + x as usize] = 7;
            assert_eq!(
                result, expected,
                "gravity {gravity:?} placed source incorrectly"
            );
        }
    }

    #[test]
    fn embed_with_gravity_allows_black_extend_cropping_when_target_is_smaller() {
        let op =
            Embed::<U8>::try_with_gravity(3, 2, Gravity::SouthEast, 5, 4, ExtendMode::Black, 1)
                .unwrap();
        let output_region = Region::new(0, 0, 3, 2);
        let input_region = op.required_input_region(&output_region);

        assert_eq!(input_region, Region::new(2, 2, 3, 2));

        let result = run_embed(
            &op,
            &[12, 13, 14, 17, 18, 19],
            input_region,
            output_region,
            1,
        );
        assert_eq!(result, vec![12, 13, 14, 17, 18, 19]);
    }

    #[test]
    fn white_fill_uses_max_sample() {
        let op = op(3, 1, 1, 0, 1, 1, ExtendMode::White, 1);
        let input_region = Region::new(-1, 0, 3, 1);
        let output_region = Region::new(0, 0, 3, 1);
        let result = run_embed(&op, &[77, 77, 77], input_region, output_region, 1);
        assert_eq!(result, vec![u8::MAX, 77, u8::MAX]);
    }

    #[test]
    fn background_fill_expands_single_value_to_all_bands() {
        let op = op(2, 1, 1, 0, 1, 1, ExtendMode::Background(vec![12.0]), 3);
        let input_region = Region::new(-1, 0, 2, 1);
        let output_region = Region::new(0, 0, 2, 1);
        let input = [90, 91, 92, 10, 20, 30];
        let result = run_embed(&op, &input, input_region, output_region, 3);
        assert_eq!(result, vec![12, 12, 12, 10, 20, 30]);
    }

    #[test]
    fn white_fill_preserves_interleaved_multiband_layout() {
        let op = op(3, 1, 1, 0, 1, 1, ExtendMode::White, 3);
        let input_region = Region::new(-1, 0, 3, 1);
        let output_region = Region::new(0, 0, 3, 1);
        let input = [1, 2, 3, 9, 8, 7, 4, 5, 6];
        let result = run_embed(&op, &input, input_region, output_region, 3);
        assert_eq!(
            result,
            vec![
                u8::MAX,
                u8::MAX,
                u8::MAX,
                9,
                8,
                7,
                u8::MAX,
                u8::MAX,
                u8::MAX
            ]
        );
    }

    #[test]
    fn copy_fill_replicates_nearest_edge_pixel() {
        let op = op(4, 1, 1, 0, 2, 1, ExtendMode::Copy, 1);
        let input_region = Region::new(-1, 0, 4, 1);
        let output_region = Region::new(0, 0, 3, 1);
        let result = run_embed(&op, &[10, 10, 20, 20], input_region, output_region, 1);
        assert_eq!(result, vec![10, 10, 20]);
    }

    #[test]
    fn repeat_wraps_negative_offset_modulo_source_size() {
        let op = op(5, 1, -1, 0, 3, 1, ExtendMode::Repeat, 1);
        let result = run_embed(
            &op,
            &[10, 20, 30],
            Region::new(0, 0, 3, 1),
            Region::new(0, 0, 5, 1),
            1,
        );
        assert_eq!(result, vec![20, 30, 10, 20, 30]);
    }

    #[test]
    fn repeat_preserves_interleaved_multiband_pixels() {
        let op = op(5, 1, -1, 0, 3, 1, ExtendMode::Repeat, 3);
        let result = run_embed(
            &op,
            &[10, 11, 12, 20, 21, 22, 30, 31, 32],
            Region::new(0, 0, 3, 1),
            Region::new(0, 0, 5, 1),
            3,
        );
        assert_eq!(
            result,
            vec![20, 21, 22, 30, 31, 32, 10, 11, 12, 20, 21, 22, 30, 31, 32]
        );
    }

    #[test]
    fn mirror_reflects_at_source_edges() {
        let op = op(8, 1, 0, 0, 3, 1, ExtendMode::Mirror, 1);
        let result = run_embed(
            &op,
            &[10, 20, 30],
            Region::new(0, 0, 3, 1),
            Region::new(0, 0, 8, 1),
            1,
        );
        assert_eq!(result, vec![10, 20, 30, 30, 20, 10, 10, 20]);
    }

    #[test]
    fn mirror_exact_boundary_reflects_last_source_pixel() {
        let op = op(4, 1, 0, 0, 3, 1, ExtendMode::Mirror, 1);
        let result = run_embed(
            &op,
            &[10, 20, 30],
            Region::new(0, 0, 3, 1),
            Region::new(3, 0, 1, 1),
            1,
        );
        assert_eq!(result, vec![30]);
    }

    #[test]
    fn mirror_preserves_interleaved_multiband_pixels() {
        let op = op(5, 1, 0, 0, 2, 1, ExtendMode::Mirror, 3);
        let result = run_embed(
            &op,
            &[1, 2, 3, 4, 5, 6],
            Region::new(0, 0, 2, 1),
            Region::new(0, 0, 5, 1),
            3,
        );
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 4, 5, 6, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn white_fill_without_overlap_paints_entire_output() {
        let op = op(8, 1, 6, 0, 1, 1, ExtendMode::White, 1);
        let result = run_embed(
            &op,
            &[77],
            Region::new(0, 0, 1, 1),
            Region::new(0, 0, 4, 1),
            1,
        );
        assert_eq!(result, vec![u8::MAX; 4]);
    }

    #[test]
    fn embed_without_overlap_returns_typed_error() {
        let result = Embed::<U8>::try_new(120, 120, 120, 0, 100, 100, ExtendMode::Black, 1);
        assert!(matches!(
            result,
            Err(BuildError::InvalidEmbedParameters { .. })
        ));
    }

    #[test]
    fn embed_exact_fit_succeeds() {
        let result = Embed::<U8>::try_new(100, 100, 50, 50, 50, 50, ExtendMode::Black, 1);
        assert!(result.is_ok(), "expected exact-fit embed to succeed");
    }

    #[test]
    fn embed_negative_offset_with_overlap_succeeds() {
        let result = Embed::<U8>::try_new(100, 100, -1, 0, 50, 50, ExtendMode::Black, 1);
        assert!(result.is_ok(), "expected partial-overlap embed to succeed");
    }

    #[test]
    fn helper_validators_reject_invalid_dimensions_and_background_shapes() {
        assert!(matches!(
            validate_dimensions(0, 1, 1, 1, 1),
            Err(BuildError::InvalidEmbedParameters { .. })
        ));
        assert!(matches!(
            validate_dimensions(1, 1, 1, 1, 0),
            Err(BuildError::InvalidEmbedParameters { .. })
        ));
        assert!(source_overlaps_canvas(0, 0, 2, 2, 2, 2));
        assert!(source_overlaps_canvas(-1, 0, 2, 2, 2, 2));
        assert!(!source_overlaps_canvas(-2, 0, 2, 2, 2, 2));
        assert_eq!(
            convert_background::<U8>(&[5.0, 260.0], 2).unwrap(),
            vec![5, 255]
        );
        assert!(matches!(
            convert_background::<U8>(&[], 1),
            Err(BuildError::InvalidEmbedParameters { .. })
        ));
        assert!(matches!(
            convert_background::<U8>(&[1.0, 2.0], 3),
            Err(BuildError::InvalidEmbedParameters { .. })
        ));
    }

    #[test]
    fn mirror_coord_and_fill_repeated_pixel_cover_boundary_cases() {
        assert_eq!(mirror_coord(-1, 3), 0);
        assert_eq!(mirror_coord(3, 3), 2);
        assert_eq!(mirror_coord(4, 3), 1);

        let mut output = [0_u8; 6];
        fill_repeated_pixel(&mut output, &[7, 8]);
        assert_eq!(output, [7, 8, 7, 8, 7, 8]);

        let mut empty: [u8; 0] = [];
        fill_repeated_pixel(&mut empty, &[1]);
    }

    #[test]
    fn edge_alias_matches_copy_and_copy_extend_replicates_rows_vertically() {
        let copy = op(3, 4, 1, 1, 1, 2, ExtendMode::Copy, 1);
        let edge = op(3, 4, 1, 1, 1, 2, ExtendMode::Edge, 1);
        let input_region = Region::new(-1, -1, 3, 4);
        let output_region = Region::new(0, 0, 3, 4);
        let input = [9_u8, 9, 9, 10, 10, 10, 20, 20, 20, 21, 21, 21];
        let expected = vec![10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20];
        assert_eq!(
            run_embed(&copy, &input, input_region, output_region, 1),
            expected
        );
        assert_eq!(
            run_embed(&edge, &input, input_region, output_region, 1),
            expected
        );
    }

    #[test]
    fn copy_extend_requires_overlap_and_node_specs_match_extend_mode() {
        let black = op(4, 4, 1, 1, 2, 2, ExtendMode::Black, 1);
        let repeat = op(4, 4, -1, 0, 2, 2, ExtendMode::Repeat, 1);

        assert!(matches!(
            Embed::<U8>::try_new(4, 4, 10, 10, 1, 1, ExtendMode::Copy, 1),
            Err(BuildError::InvalidEmbedParameters { .. })
        ));
        assert_eq!(black.node_spec(3, 5).input_tile_w, 3);
        assert_eq!(repeat.node_spec(3, 5).input_tile_w, 2);
        assert_eq!(repeat.node_spec(3, 5).input_tile_h, 2);
    }

    #[test]
    fn unsigned_constructor_accepts_offsets_and_repeat_uses_canvas_coordinates() {
        let op = Embed::<U8>::new(4, 3, 1, 1, 2, 1, ExtendMode::Repeat, 1).unwrap();
        let result = run_embed(
            &op,
            &[1, 2],
            Region::new(0, 0, 2, 1),
            Region::new(2, 1, 2, 1),
            1,
        );
        assert_eq!(result, vec![2, 1]);
    }

    proptest! {
        #[test]
        fn zero_offset_same_size_is_identity(
            pixels in proptest::collection::vec(0u8..=255, 1..=64),
        ) {
            let width = pixels.len() as u32;
            let op = op(width, 1, 0, 0, width, 1, ExtendMode::Black, 1);
            let region = Region::new(0, 0, width, 1);
            let result = run_embed(&op, &pixels, region, region, 1);
            prop_assert_eq!(result, pixels);
        }

        #[test]
        fn black_left_border_is_zero(x_off in 1i32..=8, fill in 0u8..=255) {
            let op = op(x_off as u32 + 1, 1, x_off, 0, 1, 1, ExtendMode::Black, 1);
            let input_region = Region::new(-x_off, 0, x_off as u32, 1);
            let output_region = Region::new(0, 0, x_off as u32, 1);
            let src = vec![fill; x_off as usize];
            let result = run_embed(&op, &src, input_region, output_region, 1);
            prop_assert!(result.iter().all(|&value| value == 0));
        }

        #[test]
        fn repeat_matches_euclidean_modulo(
            x_off in -8i32..=8,
            canvas_x in 0i32..=16,
            width in 1u32..=8,
        ) {
            let src = (0..width).map(|value| value as u8).collect::<Vec<_>>();
            let op = op(17, 1, x_off, 0, width, 1, ExtendMode::Repeat, 1);
            let result = run_embed(
                &op,
                &src,
                Region::new(0, 0, width, 1),
                Region::new(canvas_x, 0, 1, 1),
                1,
            );
            let expected = (canvas_x - x_off).rem_euclid(width as i32) as u8;
            prop_assert_eq!(result, vec![expected]);
        }

        #[test]
        fn mirror_matches_reflected_period(
            x_off in -8i32..=8,
            canvas_x in 0i32..=16,
            width in 1u32..=8,
        ) {
            let src = (0..width).map(|value| value as u8).collect::<Vec<_>>();
            let op = op(17, 1, x_off, 0, width, 1, ExtendMode::Mirror, 1);
            let result = run_embed(
                &op,
                &src,
                Region::new(0, 0, width, 1),
                Region::new(canvas_x, 0, 1, 1),
                1,
            );
            let expected = mirror_coord(canvas_x - x_off, width) as u8;
            prop_assert_eq!(result, vec![expected]);
        }
    }
}
