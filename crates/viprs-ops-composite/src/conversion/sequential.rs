/// Public pipeline marker for libvips-style `linecache`.
///
/// This does not add a pixel-processing node. Instead it requests thin-strip
/// execution with a bounded full-width line cache while preserving random-access
/// scheduling for the rest of the pipeline.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::sequential::LineCacheOp;
///
/// let op = LineCacheOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineCacheOp {
    lines_ahead: usize,
}

impl LineCacheOp {
    #[must_use]
    /// Creates a new `LineCacheOp`.
    pub const fn new(lines_ahead: usize) -> Self {
        Self { lines_ahead }
    }

    #[must_use]
    /// Returns or performs lines ahead.
    pub const fn lines_ahead(self) -> usize {
        self.lines_ahead
    }
}

impl Default for LineCacheOp {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Public pipeline marker for libvips-style `sequential`.
///
/// Like libvips, this keeps execution top-to-bottom and layers a bounded
/// scanline cache over upstream source reads.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::sequential::SequentialOp;
///
/// let op = SequentialOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SequentialOp {
    line_cache: LineCacheOp,
}

impl SequentialOp {
    #[must_use]
    /// Creates a new `SequentialOp`.
    pub const fn new(lines_ahead: usize) -> Self {
        Self {
            line_cache: LineCacheOp::new(lines_ahead),
        }
    }

    #[must_use]
    /// Returns or performs lines ahead.
    pub const fn lines_ahead(self) -> usize {
        self.line_cache.lines_ahead()
    }
}
