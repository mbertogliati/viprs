//! Drawing operations that paint or blend geometric primitives into images.
/// Provides the `draw_circle` module for this domain area.
pub mod draw_circle;
/// Provides the `draw_flood` module for this domain area.
pub mod draw_flood;
/// Provides the `draw_image` module for this domain area.
pub mod draw_image;
/// Provides the `draw_line` module for this domain area.
pub mod draw_line;
/// Provides the `draw_mask` module for this domain area.
pub mod draw_mask;
/// Provides the `draw_rect` module for this domain area.
pub mod draw_rect;
/// Provides the `draw_smudge` module for this domain area.
pub mod draw_smudge;

use std::collections::VecDeque;

use crate::domain::{
    error::{DrawError, ViprsError},
    image::Region,
};

pub use draw_circle::DrawCircleOp;
pub use draw_flood::DrawFloodOp;
pub use draw_flood::DrawFloodOp as FloodFillOp;
pub use draw_image::{DrawImageMode, DrawImageOp, draw_image};
pub use draw_line::DrawLineOp;
pub use draw_mask::{DrawMaskOp, draw_mask};
pub use draw_rect::DrawRectOp;
pub use draw_smudge::{DrawSmudgeOp, draw_smudge};

/// Draw behavior for shapes that can be filled or outlined.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DrawMode {
    /// Uses the `Fill` variant of `DrawMode`.
    Fill,
    /// Uses the `Stroke` variant of `DrawMode`.
    Stroke,
}

/// Draw a single pixel into an interleaved image buffer.
pub fn draw_point<T: Copy>(
    buf: &mut [T],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    color: &[T],
) -> Result<(), ViprsError> {
    let bands = validate_draw_buffer(buf, width, height, color)?;
    draw_point_in_region(
        buf,
        Region::new(0, 0, width, height),
        bands as u32,
        x,
        y,
        color,
    );
    Ok(())
}

/// Draw a rectangle into an interleaved image buffer.
pub fn draw_rect<T: Copy>(
    buf: &mut [T],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    rect_width: u32,
    rect_height: u32,
    color: &[T],
    mode: DrawMode,
) -> Result<(), ViprsError> {
    let bands = validate_draw_buffer(buf, width, height, color)?;
    draw_rect_in_region(
        buf,
        Region::new(0, 0, width, height),
        bands as u32,
        x,
        y,
        rect_width,
        rect_height,
        color,
        mode,
    );
    Ok(())
}

/// Draw a 1-pixel-wide line into an interleaved image buffer using Bresenham.
pub fn draw_line<T: Copy>(
    buf: &mut [T],
    width: u32,
    height: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: &[T],
) -> Result<(), ViprsError> {
    let bands = validate_draw_buffer(buf, width, height, color)?;
    draw_line_in_region(
        buf,
        Region::new(0, 0, width, height),
        bands as u32,
        x0,
        y0,
        x1,
        y1,
        color,
    );
    Ok(())
}

/// Draw a circle into an interleaved image buffer using the midpoint algorithm.
pub fn draw_circle<T: Copy>(
    buf: &mut [T],
    width: u32,
    height: u32,
    cx: i32,
    cy: i32,
    radius: u32,
    color: &[T],
    mode: DrawMode,
) -> Result<(), ViprsError> {
    let bands = validate_draw_buffer(buf, width, height, color)?;
    draw_circle_in_region(
        buf,
        Region::new(0, 0, width, height),
        bands as u32,
        cx,
        cy,
        radius,
        color,
        mode,
    );
    Ok(())
}

/// Flood-fill an interleaved image buffer starting from a seed point.
pub fn draw_flood<T: Copy + PartialEq>(
    buf: &mut [T],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    color: &[T],
) -> Result<(), ViprsError> {
    let bands = validate_draw_buffer(buf, width, height, color)?;
    draw_flood_in_region(
        buf,
        Region::new(0, 0, width, height),
        bands as u32,
        x,
        y,
        color,
    );
    Ok(())
}

pub(crate) fn validate_ink<T>(ink: &[T]) -> Result<(), ViprsError> {
    if ink.is_empty() {
        return Err(DrawError::EmptyColor.into());
    }

    Ok(())
}

pub(crate) struct OverlayClip {
    left: u32,
    top: u32,
    width: u32,
    height: u32,
    sub_left: u32,
    sub_top: u32,
}

pub(crate) fn clip_overlay(
    region: Region,
    overlay_x: i32,
    overlay_y: i32,
    overlay_width: u32,
    overlay_height: u32,
) -> Option<OverlayClip> {
    let region_left = i64::from(region.x);
    let region_top = i64::from(region.y);
    let region_right = region_left + i64::from(region.width);
    let region_bottom = region_top + i64::from(region.height);
    let overlay_left = i64::from(overlay_x);
    let overlay_top = i64::from(overlay_y);
    let overlay_right = overlay_left + i64::from(overlay_width);
    let overlay_bottom = overlay_top + i64::from(overlay_height);

    let left = region_left.max(overlay_left);
    let top = region_top.max(overlay_top);
    let right = region_right.min(overlay_right);
    let bottom = region_bottom.min(overlay_bottom);
    if left >= right || top >= bottom {
        return None;
    }

    let local_left = u32::try_from(left - region_left).ok()?;
    let local_top = u32::try_from(top - region_top).ok()?;
    let width = u32::try_from(right - left).ok()?;
    let height = u32::try_from(bottom - top).ok()?;
    let sub_left = u32::try_from(left - overlay_left).ok()?;
    let sub_top = u32::try_from(top - overlay_top).ok()?;

    Some(OverlayClip {
        left: local_left,
        top: local_top,
        width,
        height,
        sub_left,
        sub_top,
    })
}

pub(crate) fn draw_line_in_region<T: Copy>(
    data: &mut [T],
    region: Region,
    bands: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: &[T],
) {
    let Some(bands) = validate_region_ink(data, region, bands, color) else {
        return;
    };

    let mut x0 = i64::from(x0);
    let mut y0 = i64::from(y0);
    let x1 = i64::from(x1);
    let y1 = i64::from(y1);

    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        set_pixel_clipped(data, region, bands, x0, y0, color);
        if x0 == x1 && y0 == y1 {
            break;
        }

        let err2 = err * 2;
        if err2 >= dy {
            err += dy;
            x0 += sx;
        }
        if err2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

pub(crate) fn draw_rect_in_region<T: Copy>(
    data: &mut [T],
    region: Region,
    bands: u32,
    x: i32,
    y: i32,
    rect_width: u32,
    rect_height: u32,
    color: &[T],
    mode: DrawMode,
) {
    let Some(bands) = validate_region_ink(data, region, bands, color) else {
        return;
    };

    let x = i64::from(x);
    let y = i64::from(y);
    let rect_width = i64::from(rect_width);
    let rect_height = i64::from(rect_height);
    if rect_width <= 0 || rect_height <= 0 {
        return;
    }

    match mode {
        DrawMode::Fill => fill_rect(data, region, bands, x, y, rect_width, rect_height, color),
        DrawMode::Stroke if rect_width > 2 && rect_height > 2 => {
            fill_rect(data, region, bands, x, y, rect_width, 1, color);
            fill_rect(
                data,
                region,
                bands,
                x + rect_width - 1,
                y,
                1,
                rect_height,
                color,
            );
            fill_rect(
                data,
                region,
                bands,
                x,
                y + rect_height - 1,
                rect_width,
                1,
                color,
            );
            fill_rect(data, region, bands, x, y, 1, rect_height, color);
        }
        DrawMode::Stroke => {
            fill_rect(data, region, bands, x, y, rect_width, rect_height, color);
        }
    }
}

pub(crate) fn draw_circle_in_region<T: Copy>(
    data: &mut [T],
    region: Region,
    bands: u32,
    cx: i32,
    cy: i32,
    radius: u32,
    color: &[T],
    mode: DrawMode,
) {
    let Some(bands) = validate_region_ink(data, region, bands, color) else {
        return;
    };

    let cx = i64::from(cx);
    let cy = i64::from(cy);
    let mut x = 0_i64;
    let mut y = i64::from(radius);
    let mut d = 3_i64 - 2_i64 * i64::from(radius);

    while x < y {
        draw_circle_octants(data, region, bands, cx, cy, x, y, color, mode);
        if d < 0 {
            d += 4 * x + 6;
        } else {
            d += 4 * (x - y) + 10;
            y -= 1;
        }
        x += 1;
    }

    if x == y {
        draw_circle_octants(data, region, bands, cx, cy, x, y, color, mode);
    }
}

pub(crate) fn draw_flood_in_region<T: Copy + PartialEq>(
    data: &mut [T],
    region: Region,
    bands: u32,
    x: i32,
    y: i32,
    color: &[T],
) {
    let Some(bands) = validate_region_ink(data, region, bands, color) else {
        return;
    };

    let Some(seed_offset) = point_offset(region, bands, i64::from(x), i64::from(y)) else {
        return;
    };

    // Flood fill needs runtime-sized frontier storage plus a seed-pixel copy because the
    // tile band count is dynamic. This mirrors libvips draw_flood.c, which also allocates
    // heap state for the traversal.
    let seed = data[seed_offset..seed_offset + bands].to_vec();
    if seed.as_slice() == color {
        return;
    }

    // Heap allocation is intentional here: libvips also allocates dynamic traversal state
    // for flood fill, and the frontier size depends on the connected region.
    let mut queue = VecDeque::with_capacity(region.pixel_count());
    queue.push_back((i64::from(x), i64::from(y)));

    while let Some((cx, cy)) = queue.pop_front() {
        let Some(offset) = point_offset(region, bands, cx, cy) else {
            continue;
        };

        if data[offset..offset + bands] != seed[..] {
            continue;
        }

        data[offset..offset + bands].copy_from_slice(color);
        queue.push_back((cx + 1, cy));
        queue.push_back((cx - 1, cy));
        queue.push_back((cx, cy + 1));
        queue.push_back((cx, cy - 1));
    }
}

pub(crate) fn draw_point_in_region<T: Copy>(
    data: &mut [T],
    region: Region,
    bands: u32,
    x: i32,
    y: i32,
    color: &[T],
) {
    let Some(bands) = validate_region_ink(data, region, bands, color) else {
        return;
    };

    set_pixel_clipped(data, region, bands, i64::from(x), i64::from(y), color);
}

fn validate_draw_buffer<T>(
    buf: &[T],
    width: u32,
    height: u32,
    color: &[T],
) -> Result<usize, ViprsError> {
    validate_ink(color)?;

    let bands = color.len();
    let expected = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|pixels| pixels.checked_mul(bands))
        .ok_or(DrawError::BufferDimensionsOverflow {
            width,
            height,
            bands,
        })?;

    if buf.len() != expected {
        return Err(DrawError::BufferLengthMismatch {
            len: buf.len(),
            expected,
            width,
            height,
            bands,
        }
        .into());
    }

    Ok(bands)
}

fn validate_region_ink<T>(data: &[T], region: Region, bands: u32, color: &[T]) -> Option<usize> {
    let bands = usize::try_from(bands).ok()?;
    let expected = region.pixel_count().checked_mul(bands)?;
    debug_assert_eq!(
        data.len(),
        expected,
        "tile data length must equal region area * bands"
    );
    debug_assert_eq!(
        color.len(),
        bands,
        "draw ink band count must match tile band count"
    );
    if data.len() != expected || color.len() != bands || color.is_empty() {
        return None;
    }

    Some(bands)
}

fn draw_circle_octants<T: Copy>(
    data: &mut [T],
    region: Region,
    bands: usize,
    cx: i64,
    cy: i64,
    x: i64,
    y: i64,
    color: &[T],
    mode: DrawMode,
) {
    match mode {
        DrawMode::Fill => {
            draw_scanline(data, region, bands, cy + y, cx - x, cx + x, color);
            draw_scanline(data, region, bands, cy - y, cx - x, cx + x, color);
            draw_scanline(data, region, bands, cy + x, cx - y, cx + y, color);
            draw_scanline(data, region, bands, cy - x, cx - y, cx + y, color);
        }
        DrawMode::Stroke => {
            set_pixel_clipped(data, region, bands, cx - x, cy + y, color);
            set_pixel_clipped(data, region, bands, cx + x, cy + y, color);
            set_pixel_clipped(data, region, bands, cx - x, cy - y, color);
            set_pixel_clipped(data, region, bands, cx + x, cy - y, color);
            set_pixel_clipped(data, region, bands, cx - y, cy + x, color);
            set_pixel_clipped(data, region, bands, cx + y, cy + x, color);
            set_pixel_clipped(data, region, bands, cx - y, cy - x, color);
            set_pixel_clipped(data, region, bands, cx + y, cy - x, color);
        }
    }
}

fn fill_rect<T: Copy>(
    data: &mut [T],
    region: Region,
    bands: usize,
    x: i64,
    y: i64,
    rect_width: i64,
    rect_height: i64,
    color: &[T],
) {
    let left = x.max(i64::from(region.x));
    let top = y.max(i64::from(region.y));
    let right = (x + rect_width).min(i64::from(region.x) + i64::from(region.width));
    let bottom = (y + rect_height).min(i64::from(region.y) + i64::from(region.height));

    if left >= right || top >= bottom {
        return;
    }

    for row in top..bottom {
        draw_scanline(data, region, bands, row, left, right - 1, color);
    }
}

fn draw_scanline<T: Copy>(
    data: &mut [T],
    region: Region,
    bands: usize,
    y: i64,
    x1: i64,
    x2: i64,
    color: &[T],
) {
    let top = i64::from(region.y);
    let bottom = top + i64::from(region.height);
    if y < top || y >= bottom {
        return;
    }

    let left = x1.max(i64::from(region.x));
    let right = x2.min(i64::from(region.x) + i64::from(region.width) - 1);
    if left > right {
        return;
    }

    for x in left..=right {
        set_pixel_clipped(data, region, bands, x, y, color);
    }
}

fn set_pixel_clipped<T: Copy>(
    data: &mut [T],
    region: Region,
    bands: usize,
    x: i64,
    y: i64,
    color: &[T],
) {
    let Some(offset) = point_offset(region, bands, x, y) else {
        return;
    };

    data[offset..offset + bands].copy_from_slice(color);
}

fn point_offset(region: Region, bands: usize, x: i64, y: i64) -> Option<usize> {
    let left = i64::from(region.x);
    let top = i64::from(region.y);
    if x < left
        || y < top
        || x >= left + i64::from(region.width)
        || y >= top + i64::from(region.height)
    {
        return None;
    }

    let local_x = usize::try_from(x - left).ok()?;
    let local_y = usize::try_from(y - top).ok()?;
    Some((local_y * region.width as usize + local_x) * bands)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn draw_rect_fill_sets_expected_pixels() {
        let mut buf = vec![0_u8; 5 * 4];

        draw_rect(&mut buf, 5, 4, 1, 1, 3, 2, &[9], DrawMode::Fill).unwrap();

        assert_eq!(
            buf,
            vec![
                0, 0, 0, 0, 0, //
                0, 9, 9, 9, 0, //
                0, 9, 9, 9, 0, //
                0, 0, 0, 0, 0,
            ]
        );
    }

    #[test]
    fn draw_rect_stroke_leaves_interior_untouched() {
        let mut buf = vec![0_u8; 5 * 5];

        draw_rect(&mut buf, 5, 5, 1, 1, 3, 3, &[7], DrawMode::Stroke).unwrap();

        assert_eq!(buf[2 * 5 + 2], 0);
        assert_eq!(buf[5 + 1], 7);
        assert_eq!(buf[5 + 2], 7);
        assert_eq!(buf[5 + 3], 7);
        assert_eq!(buf[2 * 5 + 1], 7);
        assert_eq!(buf[2 * 5 + 3], 7);
        assert_eq!(buf[3 * 5 + 1], 7);
        assert_eq!(buf[3 * 5 + 2], 7);
        assert_eq!(buf[3 * 5 + 3], 7);
    }

    #[test]
    fn draw_point_sets_exactly_one_pixel() {
        let mut buf = vec![0_u8; 3 * 3];

        draw_point(&mut buf, 3, 3, 1, 2, &[5]).unwrap();

        let non_zero: Vec<_> = buf
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (*value != 0).then_some((index, *value)))
            .collect();
        assert_eq!(non_zero, vec![(7, 5)]);
    }

    #[test]
    fn draw_line_rasterizes_diagonal() {
        let mut buf = vec![0_u8; 5 * 5];

        draw_line(&mut buf, 5, 5, 0, 0, 4, 4, &[3]).unwrap();

        for index in 0..5 {
            assert_eq!(buf[index * 5 + index], 3);
        }
    }

    #[test]
    fn draw_circle_stroke_matches_reference_points() {
        let mut buf = vec![0_u8; 5 * 5];

        draw_circle(&mut buf, 5, 5, 2, 2, 2, &[1], DrawMode::Stroke).unwrap();

        let expected = [
            (2, 4),
            (2, 0),
            (0, 2),
            (4, 2),
            (1, 4),
            (3, 4),
            (1, 0),
            (3, 0),
            (0, 1),
            (4, 1),
            (0, 3),
            (4, 3),
        ];

        for (x, y) in expected {
            assert_eq!(buf[y * 5 + x], 1, "missing point at ({x}, {y})");
        }
    }

    #[test]
    fn draw_rect_rgba_overwrites_pixel_without_alpha_compositing() {
        let mut buf = vec![
            10_u8, 20, 30, 40, 10, 20, 30, 40, //
            10, 20, 30, 40, 10, 20, 30, 40,
        ];

        draw_rect(
            &mut buf,
            2,
            2,
            1,
            0,
            1,
            1,
            &[100, 110, 120, 64],
            DrawMode::Fill,
        )
        .unwrap();

        assert_eq!(&buf[4..8], &[100, 110, 120, 64]);
        assert_eq!(&buf[0..4], &[10, 20, 30, 40]);
    }

    #[test]
    fn draw_line_rgba_overwrites_each_drawn_sample() {
        let mut buf = vec![5_u8; 3 * 3 * 4];

        draw_line(&mut buf, 3, 3, 0, 0, 2, 2, &[9, 8, 7, 6]).unwrap();

        for index in 0..3 {
            let offset = (index * 3 + index) * 4;
            assert_eq!(&buf[offset..offset + 4], &[9, 8, 7, 6]);
        }
        assert_eq!(&buf[4..8], &[5, 5, 5, 5]);
    }

    #[test]
    fn draw_circle_rgba_stroke_preserves_overwrite_semantics() {
        let mut buf = vec![1_u8; 5 * 5 * 4];

        draw_circle(&mut buf, 5, 5, 2, 2, 1, &[9, 10, 11, 12], DrawMode::Stroke).unwrap();

        for (x, y) in [(2, 1), (1, 2), (3, 2), (2, 3)] {
            let offset = (y * 5 + x) * 4;
            assert_eq!(&buf[offset..offset + 4], &[9, 10, 11, 12]);
        }

        assert_eq!(&buf[0..4], &[1, 1, 1, 1]);
    }

    #[test]
    fn draw_point_out_of_bounds_is_noop() {
        let mut buf = vec![0_u8; 4];

        draw_point(&mut buf, 2, 2, -1, 0, &[7]).unwrap();

        assert_eq!(buf, vec![0, 0, 0, 0]);
    }

    #[test]
    fn draw_flood_fills_connected_component() {
        let mut buf = vec![
            0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2,
        ];

        draw_flood(&mut buf, 5, 5, 1, 1, &[9]).unwrap();

        assert_eq!(buf[6], 9);
        assert_eq!(buf[7], 9);
        assert_eq!(buf[11], 9);
        assert_eq!(buf[12], 9);
        assert_eq!(buf[18], 2);
        assert_eq!(buf[19], 2);
    }

    proptest! {
        #[test]
        fn draw_point_only_changes_target_pixel(
            width in 1_u32..8,
            height in 1_u32..8,
            x in 0_i32..7,
            y in 0_i32..7,
            value in any::<u8>()
        ) {
            prop_assume!(x < width as i32);
            prop_assume!(y < height as i32);

            let mut buf = vec![0_u8; width as usize * height as usize];
            draw_point(&mut buf, width, height, x, y, &[value]).unwrap();

            for yy in 0..height as usize {
                for xx in 0..width as usize {
                    let expected = if xx == x as usize && yy == y as usize {
                        value
                    } else {
                        0
                    };
                    prop_assert_eq!(buf[yy * width as usize + xx], expected);
                }
            }
        }

        #[test]
        fn draw_horizontal_line_marks_exact_span(
            (width, height, y, x1, x2, value) in (1_u32..12, 1_u32..12)
                .prop_flat_map(|(width, height)| {
                    (
                        Just(width),
                        Just(height),
                        0_u32..height,
                        0_u32..width,
                        0_u32..width,
                        1_u8..=u8::MAX,
                    )
                })
        ) {
            let mut buf = vec![0_u8; width as usize * height as usize];
            draw_line(
                &mut buf,
                width,
                height,
                x1 as i32,
                y as i32,
                x2 as i32,
                y as i32,
                &[value],
            )
            .unwrap();

            let start = x1.min(x2) as usize;
            let end = x1.max(x2) as usize;
            for yy in 0..height as usize {
                for xx in 0..width as usize {
                    let expected = if yy == y as usize && (start..=end).contains(&xx) {
                        value
                    } else {
                        0
                    };
                    prop_assert_eq!(buf[yy * width as usize + xx], expected);
                }
            }
        }

        #[test]
        fn draw_rect_fill_paints_expected_area(
            (width, height, rect_width, rect_height, x, y, value) in
                (1_u32..10, 1_u32..10)
                    .prop_flat_map(|(width, height)| {
                        (Just(width), Just(height), 1_u32..=width, 1_u32..=height)
                    })
                    .prop_flat_map(|(width, height, rect_width, rect_height)| {
                        (
                            Just(width),
                            Just(height),
                            Just(rect_width),
                            Just(rect_height),
                            0_u32..(width - rect_width + 1),
                            0_u32..(height - rect_height + 1),
                            1_u8..=u8::MAX,
                        )
                    })
        ) {
            let mut buf = vec![0_u8; width as usize * height as usize];
            draw_rect(
                &mut buf,
                width,
                height,
                x as i32,
                y as i32,
                rect_width,
                rect_height,
                &[value],
                DrawMode::Fill,
            )
            .unwrap();

            let painted = buf.iter().filter(|&&pixel| pixel == value).count();
            prop_assert_eq!(painted, rect_width as usize * rect_height as usize);

            for yy in 0..height as usize {
                for xx in 0..width as usize {
                    let inside_x = (x as usize..(x + rect_width) as usize).contains(&xx);
                    let inside_y = (y as usize..(y + rect_height) as usize).contains(&yy);
                    let expected = if inside_x && inside_y { value } else { 0 };
                    prop_assert_eq!(buf[yy * width as usize + xx], expected);
                }
            }
        }

        #[test]
        fn draw_circle_stroke_stays_within_one_pixel_of_radius(
            radius in 1_u32..10,
            value in 1_u8..=u8::MAX
        ) {
            let size = radius * 2 + 3;
            let cx = (radius + 1) as i32;
            let cy = (radius + 1) as i32;
            let mut buf = vec![0_u8; size as usize * size as usize];

            draw_circle(
                &mut buf,
                size,
                size,
                cx,
                cy,
                radius,
                &[value],
                DrawMode::Stroke,
            )
            .unwrap();

            let lower = i64::from(radius.saturating_sub(1));
            let upper = i64::from(radius + 1);
            let mut painted = 0_usize;

            for yy in 0..size as usize {
                for xx in 0..size as usize {
                    if buf[yy * size as usize + xx] != value {
                        continue;
                    }

                    painted += 1;
                    let dx = xx as i64 - i64::from(cx);
                    let dy = yy as i64 - i64::from(cy);
                    let distance_sq = dx * dx + dy * dy;
                    prop_assert!(distance_sq >= lower * lower);
                    prop_assert!(distance_sq <= upper * upper);
                }
            }

            prop_assert!(painted > 0);
        }
    }
}
