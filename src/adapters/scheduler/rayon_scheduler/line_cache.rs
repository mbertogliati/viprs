use super::{DynImageSource, LineCacheAccess, Region, ViprsError};

#[derive(Clone, Debug)]
/// The `SequentialLineCache` type provides concrete adapter functionality in the `scheduler` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::scheduler::rayon_scheduler::SequentialLineCache>();
/// ```
pub struct SequentialLineCache {
    image_width: u32,
    image_height: u32,
    pixel_bytes: usize,
    lines_ahead: usize,
    refill_lines: usize,
    access: LineCacheAccess,
    start_y: i32,
    line_count: usize,
    next_source_y: i32,
    max_cached_lines: usize,
    bytes: Vec<u8>,
}

impl SequentialLineCache {
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::scheduler::rayon_scheduler::new;
    /// ```
    pub(crate) fn new(
        image_width: u32,
        image_height: u32,
        pixel_bytes: usize,
        lines_ahead: usize,
        refill_lines: usize,
        access: LineCacheAccess,
    ) -> Self {
        let clamped_lines = lines_ahead.max(1);
        let row_bytes = image_width as usize * pixel_bytes;
        Self {
            image_width,
            image_height,
            pixel_bytes,
            lines_ahead: clamped_lines,
            refill_lines: refill_lines.max(1),
            access,
            start_y: 0,
            line_count: 0,
            next_source_y: 0,
            max_cached_lines: 0,
            bytes: vec![0; row_bytes.saturating_mul(clamped_lines)],
        }
    }

    const fn row_bytes(&self) -> usize {
        self.image_width as usize * self.pixel_bytes
    }

    const fn cached_end_y(&self) -> i32 {
        self.start_y + self.line_count as i32
    }

    fn region_bottom(region: Region) -> Option<i32> {
        region.y.checked_add(i32::try_from(region.height).ok()?)
    }

    fn is_supported_rect(&self, region: Region) -> bool {
        if region.is_empty()
            || region.x < 0
            || region.y < 0
            || region.width > self.image_width
            || region.height as usize > self.lines_ahead
        {
            return false;
        }

        let Some(bottom) = Self::region_bottom(region) else {
            return false;
        };
        let Some(right) = region
            .x
            .checked_add(i32::try_from(region.width).ok().unwrap_or(i32::MAX))
        else {
            return false;
        };

        right <= self.image_width as i32 && bottom <= self.image_height as i32
    }

    fn drop_front_lines(&mut self, lines: usize) {
        if lines == 0 || self.line_count == 0 {
            return;
        }

        let dropped = lines.min(self.line_count);
        let row_bytes = self.row_bytes();
        let remaining = self.line_count - dropped;
        let src_start = dropped * row_bytes;
        let src_end = src_start + remaining * row_bytes;
        self.bytes.copy_within(src_start..src_end, 0);
        self.line_count = remaining;
        self.start_y += dropped as i32;
    }

    fn append_lines(
        &mut self,
        source: &dyn DynImageSource,
        chunk_top: i32,
        chunk_height: u32,
    ) -> Result<(), ViprsError> {
        let row_bytes = self.row_bytes();
        let start = self.line_count * row_bytes;
        let len = chunk_height as usize * row_bytes;
        let region = Region::new(0, chunk_top, self.image_width, chunk_height);
        source.read_region(region, &mut self.bytes[start..start + len])?;
        self.line_count += chunk_height as usize;
        self.next_source_y = chunk_top + chunk_height as i32;
        self.max_cached_lines = self.max_cached_lines.max(self.line_count);
        Ok(())
    }

    fn copy_cached_region(&self, region: Region, output: &mut [u8]) {
        let row_bytes = self.row_bytes();
        let copy_row_bytes = region.width as usize * self.pixel_bytes;
        let x_offset = region.x as usize * self.pixel_bytes;
        let y_offset = (region.y - self.start_y) as usize;
        for row in 0..region.height as usize {
            let src_start = (y_offset + row) * row_bytes + x_offset;
            let dst_start = row * copy_row_bytes;
            output[dst_start..dst_start + copy_row_bytes]
                .copy_from_slice(&self.bytes[src_start..src_start + copy_row_bytes]);
        }
    }

    const fn reset_window(&mut self, start_y: i32) {
        self.start_y = start_y;
        self.line_count = 0;
        self.next_source_y = start_y;
    }

    pub(super) fn read_region(
        &mut self,
        source: &dyn DynImageSource,
        region: Region,
        output: &mut [u8],
    ) -> Result<bool, ViprsError> {
        if !self.is_supported_rect(region) {
            return Ok(false);
        }

        if self.access == LineCacheAccess::Random && region.y < self.start_y {
            self.reset_window(region.y);
        }

        if region.y < self.start_y {
            return Err(ViprsError::Scheduler(format!(
                "sequential line cache request moved behind retained window: region y={} cache start={}",
                region.y, self.start_y
            )));
        }

        let bottom = Self::region_bottom(region).ok_or_else(|| {
            ViprsError::Scheduler("sequential line cache region bottom overflowed".into())
        })?;
        while self.cached_end_y() < bottom {
            let remaining = (self.image_height as i32 - self.next_source_y).max(0) as u32;
            if remaining == 0 {
                break;
            }
            let needed = (bottom - self.cached_end_y()).max(0) as u32;
            let chunk_height = needed.min(self.refill_lines as u32).min(remaining).max(1);
            while self.line_count + chunk_height as usize > self.lines_ahead {
                if self.access == LineCacheAccess::Random {
                    if self.start_y >= region.y {
                        break;
                    }
                } else if self.start_y >= region.y {
                    break;
                }
                self.drop_front_lines(1);
            }
            self.append_lines(source, self.next_source_y, chunk_height)?;
        }

        if self.cached_end_y() < bottom {
            return Ok(false);
        }

        self.copy_cached_region(region, output);
        Ok(true)
    }

    #[cfg(test)]
    pub(super) fn max_cached_lines(&self) -> usize {
        self.max_cached_lines
    }
}
