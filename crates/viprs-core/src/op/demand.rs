use crate::image::Region;

/// Preferred output tile geometry for an operation, matching libvips demand styles.
///
/// This gives planners a cheap hint about cache-friendly tile shapes before any per-op buffers are
/// allocated.
///
/// # Examples
/// ```rust
/// # use viprs_core::op::DemandHint;
/// assert_eq!(DemandHint::OneLine.tile_height(100, 50), 1);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DemandHint {
    /// Uses the `Any` variant of `DemandHint`.
    Any = 0,
    /// Uses the `ThinStrip` variant of `DemandHint`.
    ThinStrip = 1,
    /// Uses the `FatStrip` variant of `DemandHint`.
    FatStrip = 2,
    /// Uses the `SmallTile` variant of `DemandHint`.
    SmallTile = 3,
    /// Uses the `FullImage` variant of `DemandHint`.
    FullImage = 4,
    /// Uses the `OneLine` variant of `DemandHint`.
    OneLine = 5,
}

impl DemandHint {
    /// Return the planner's preferred tile width for an image width.
    ///
    /// This keeps tile-shape policy with the demand hint instead of scattering it across
    /// schedulers.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::op::DemandHint;
    /// assert_eq!(DemandHint::SmallTile.tile_width(512), 128);
    /// ```
    #[must_use]
    pub const fn tile_width(self, image_width: u32) -> u32 {
        match self {
            Self::SmallTile => 128,
            Self::FullImage | Self::OneLine | Self::Any | Self::ThinStrip | Self::FatStrip => {
                image_width
            }
        }
    }

    /// Return the planner's preferred tile height for image dimensions.
    ///
    /// This complements [`DemandHint::tile_width`] when choosing scheduler tile shapes.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::op::DemandHint;
    /// assert_eq!(DemandHint::FullImage.tile_height(100, 20), 20);
    /// ```
    #[must_use]
    pub const fn tile_height(self, image_width: u32, image_height: u32) -> u32 {
        match self {
            Self::SmallTile => 128,
            Self::FatStrip | Self::Any => {
                if image_height <= 1024 {
                    256
                } else if image_height >= 4096 {
                    // Larger tiles for very large images reduce per-tile dispatch overhead
                    // and improve memcpy efficiency (fewer, larger transfers).
                    64
                } else {
                    16
                }
            }
            Self::ThinStrip => {
                if image_width > 10_000 {
                    1
                } else {
                    16
                }
            }
            Self::FullImage => image_height,
            Self::OneLine => 1,
        }
    }
}

/// Declares the tile dimensions a node needs from its upstream buffer and
/// writes to its downstream buffer.
///
/// For pixel-local operations (`required_input_region` is identity) all four
/// fields equal the pipeline tile size. Override `node_spec` in `DynOperation`
/// or `DynViewOp` only for operations that change tile geometry, e.g.:
///
/// - Convolution with radius `r`: input is `(tile_w + 2r) × (tile_h + 2r)`.
/// - Rotate90: output tile has dimensions `(tile_h, tile_w)`.
///
/// This is a compile-time declaration used by `compile()` to pre-allocate
/// per-node buffers. It is deliberately separate from `required_input_region`
/// (runtime coordinate mapping) and from `output_width`/`output_height`
/// (full image dimension propagation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinateDrivenSourceSpec {
    /// Source-direct slot whose read bounds are computed after another slot is realized.
    pub source_slot: usize,
    /// Slot that must be materialized first to derive `source_slot` demand.
    pub dependency_slot: usize,
}

/// Static buffer-shape declaration for one compiled node.
///
/// The pipeline compiler uses this to size per-node tiles ahead of execution, separate from the
/// runtime coordinate mapping returned by `required_input_region`.
///
/// # Examples
/// ```rust
/// # use viprs_core::op::NodeSpec;
/// let spec = NodeSpec::identity(64, 32);
/// assert_eq!(spec.output_tile_w, 64);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeSpec {
    /// Width of the input tile this node reads from its upstream buffer.
    pub input_tile_w: u32,
    /// Height of the input tile this node reads from its upstream buffer.
    pub input_tile_h: u32,
    /// Width of the output tile this node writes to its downstream buffer.
    pub output_tile_w: u32,
    /// Height of the output tile this node writes to its downstream buffer.
    pub output_tile_h: u32,
    /// Optional declaration that one source-direct slot depends on another realized slot.
    pub coordinate_driven_source: Option<CoordinateDrivenSourceSpec>,
}

/// How a single-input operation wants the scheduler to fill its source buffer.
///
/// The default is `Rect`: source bytes are read as one packed rectangle and the
/// operation receives that same region. `PointGrid` is a sparse-demand shape for
/// libvips-style point sampling: the scheduler reads one 1x1 source region per
/// output pixel and packs those pixels into `input_region`.
///
/// This does not replace `required_input_region`; it is deliberately narrower and
/// expresses sparse-demand shape rather than the fallback packed rectangle.
/// Dynamic pipelines preserve `PointGrid` across single-input pixel-local nodes;
/// any merge or non-pixel-local step falls back to `Rect` over
/// `bounding_source_region()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceReadPlan {
    /// Uses the `Rect` variant of `SourceReadPlan`.
    Rect {
        /// Stores the `region` value for this item.
        region: Region,
    },
    /// Uses the `PointGrid` variant of `SourceReadPlan`.
    PointGrid {
        /// Stores the `input_region` value for this item.
        input_region: Region,
        /// Stores the `source_origin_x` value for this item.
        source_origin_x: i32,
        /// Stores the `source_origin_y` value for this item.
        source_origin_y: i32,
        /// Stores the `x_step` value for this item.
        x_step: u32,
        /// Stores the `y_step` value for this item.
        y_step: u32,
    },
}

impl SourceReadPlan {
    /// Build a packed-rectangle read plan.
    ///
    /// This is the default source read shape for most operations.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{image::Region, op::SourceReadPlan};
    /// let plan = SourceReadPlan::rect(Region::new(0, 0, 2, 2));
    /// assert_eq!(plan.produced_region().width, 2);
    /// ```
    #[must_use]
    pub const fn rect(region: Region) -> Self {
        Self::Rect { region }
    }

    /// Return the packed region materialized by this read plan.
    ///
    /// This lets planners reason about downstream tile shapes even for sparse source reads.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{image::Region, op::SourceReadPlan};
    /// let plan = SourceReadPlan::rect(Region::new(1, 2, 3, 4));
    /// assert_eq!(plan.produced_region(), Region::new(1, 2, 3, 4));
    /// ```
    #[must_use]
    pub const fn produced_region(self) -> Region {
        match self {
            Self::Rect { region }
            | Self::PointGrid {
                input_region: region,
                ..
            } => region,
        }
    }

    /// Return the smallest packed source rectangle that covers this plan.
    ///
    /// Sparse point-grid reads use this as a conservative fallback when upstream propagation
    /// cannot preserve sparse demand.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{image::Region, op::SourceReadPlan};
    /// let plan = SourceReadPlan::rect(Region::new(0, 0, 2, 2));
    /// assert_eq!(plan.bounding_source_region(), Region::new(0, 0, 2, 2));
    /// ```
    #[must_use]
    pub const fn bounding_source_region(self) -> Region {
        match self {
            Self::Rect { region } => region,
            Self::PointGrid {
                input_region,
                source_origin_x,
                source_origin_y,
                x_step,
                y_step,
            } => {
                if input_region.is_empty() {
                    return Region::new(source_origin_x, source_origin_y, 0, 0);
                }

                Region::new(
                    source_origin_x,
                    source_origin_y,
                    (input_region.width - 1) * x_step + 1,
                    (input_region.height - 1) * y_step + 1,
                )
            }
        }
    }
}

impl NodeSpec {
    /// Default spec: all four dimensions equal `tile_w × tile_h`.
    /// Correct for every pixel-local operation.
    #[must_use]
    pub const fn identity(tile_w: u32, tile_h: u32) -> Self {
        Self {
            input_tile_w: tile_w,
            input_tile_h: tile_h,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    /// Attach a coordinate-driven source dependency to this node specification.
    ///
    /// This is used by multi-input ops that must materialize one slot before computing another
    /// slot's source bounds.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::op::NodeSpec;
    /// let spec = NodeSpec::identity(8, 8).with_coordinate_driven_source(1, 0);
    /// assert!(spec.coordinate_driven_source.is_some());
    /// ```
    #[must_use]
    pub const fn with_coordinate_driven_source(
        mut self,
        source_slot: usize,
        dependency_slot: usize,
    ) -> Self {
        self.coordinate_driven_source = Some(CoordinateDrivenSourceSpec {
            source_slot,
            dependency_slot,
        });
        self
    }
}
