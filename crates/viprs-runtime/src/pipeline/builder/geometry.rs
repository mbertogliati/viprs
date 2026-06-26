use super::state::validate_extract_area_bounds;
use super::{
    Angle, Angle45, BandFormatId, BuildError, Commit, Committed, DynOperation, DynViewOp,
    EmbedBridge, ExtendMode, ExtractArea, F32, F64, Flip, Gravity, GridBridge, I16, I32,
    ImagePipeline, InterpolationKernel, MsbOp, OperationBridge, ReplicateBridge, RotBridge,
    SimilarityBridge, SubsampleBridge, U8, U16, U32, ViewBridge, Wrap, ZoomBridge,
};

#[inline]
const fn rot45_to_right_angle(angle: Angle45) -> Option<Angle> {
    match angle {
        Angle45::D0 => Some(Angle::D0),
        Angle45::D90 => Some(Angle::D90),
        Angle45::D180 => Some(Angle::D180),
        Angle45::D270 => Some(Angle::D270),
        _ => None,
    }
}

#[inline]
const fn rot45_to_degrees(angle: Angle45) -> f64 {
    match angle {
        Angle45::D0 => 0.0,
        Angle45::D45 => 45.0,
        Angle45::D90 => 90.0,
        Angle45::D135 => 135.0,
        Angle45::D180 => 180.0,
        Angle45::D225 => 225.0,
        Angle45::D270 => 270.0,
        Angle45::D315 => 315.0,
    }
}

impl<Op: Commit> ImagePipeline<Op> {
    /// Crop the image to the rectangle `(x, y, width, height)` in source coordinates.
    ///
    /// Updates the pipeline output dimensions to `width × height`. Dispatches over
    /// the current format. This is a zero-copy operation: the coordinate shift is
    /// encoded in `required_input_region`; no `process_region` is called.
    pub fn extract_area(
        mut self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        self.flush_pending()?;
        let (image_width, image_height) = self.current_dimensions();
        validate_extract_area_bounds(x, y, width, height, image_width, image_height)?;
        let bands = self.bands;
        let op: Box<dyn DynViewOp> = match self.current_format {
            BandFormatId::U8 => Box::new(ViewBridge::new(
                ExtractArea::<U8>::new(x, y, width, height),
                bands,
            )),
            BandFormatId::U16 => Box::new(ViewBridge::new(
                ExtractArea::<U16>::new(x, y, width, height),
                bands,
            )),
            BandFormatId::I16 => Box::new(ViewBridge::new(
                ExtractArea::<I16>::new(x, y, width, height),
                bands,
            )),
            BandFormatId::U32 => Box::new(ViewBridge::new(
                ExtractArea::<U32>::new(x, y, width, height),
                bands,
            )),
            BandFormatId::I32 => Box::new(ViewBridge::new(
                ExtractArea::<I32>::new(x, y, width, height),
                bands,
            )),
            BandFormatId::F32 => Box::new(ViewBridge::new(
                ExtractArea::<F32>::new(x, y, width, height),
                bands,
            )),
            BandFormatId::F64 => Box::new(ViewBridge::new(
                ExtractArea::<F64>::new(x, y, width, height),
                bands,
            )),
        };

        let idx = self.arena.add_view_node(op);
        if let Some(prev) = self.last_node {
            self.arena.connect(prev, idx)?;
        }
        self.last_node = Some(idx);
        // Dimension propagation is handled automatically by compile() via
        // output_width/output_height. No manual set_dimensions call needed.
        Ok(self.into_state(Committed))
    }

    /// Embed the image in a larger canvas of `dst_width × dst_height`, placing its
    /// top-left corner at `(x_off, y_off)`. Border pixels are filled according to
    /// `extend`. Output dimensions become `dst_width × dst_height`.
    ///
    /// `src_width` and `src_height` must match the current pipeline output dimensions.
    /// Dispatches over the current format.
    #[allow(clippy::too_many_arguments)]
    /// `embed` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::pipeline::builder::embed;
    /// ```
    pub fn embed(
        self,
        dst_width: u32,
        dst_height: u32,
        x_off: u32,
        y_off: u32,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        let x_off = i32::try_from(x_off).map_err(|_| BuildError::InvalidEmbedParameters {
            message: "unsigned embed offsets must fit within i32",
        })?;
        let y_off = i32::try_from(y_off).map_err(|_| BuildError::InvalidEmbedParameters {
            message: "unsigned embed offsets must fit within i32",
        })?;
        self.embed_signed(
            dst_width, dst_height, x_off, y_off, src_width, src_height, extend,
        )
    }

    /// Embed the image with signed libvips offsets. This accepts negative `x_off`
    /// and `y_off`, matching `vips_embed`.
    #[allow(clippy::too_many_arguments)]
    /// `embed_signed` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::pipeline::builder::embed_signed;
    /// ```
    pub fn embed_signed(
        self,
        dst_width: u32,
        dst_height: u32,
        x_off: i32,
        y_off: i32,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        let bands = self.bands;
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(EmbedBridge::<U8>::try_new(
                dst_width, dst_height, x_off, y_off, src_width, src_height, extend, bands,
            )?),
            BandFormatId::U16 => Box::new(EmbedBridge::<U16>::try_new(
                dst_width, dst_height, x_off, y_off, src_width, src_height, extend, bands,
            )?),
            BandFormatId::I16 => Box::new(EmbedBridge::<I16>::try_new(
                dst_width, dst_height, x_off, y_off, src_width, src_height, extend, bands,
            )?),
            BandFormatId::U32 => Box::new(EmbedBridge::<U32>::try_new(
                dst_width, dst_height, x_off, y_off, src_width, src_height, extend, bands,
            )?),
            BandFormatId::I32 => Box::new(EmbedBridge::<I32>::try_new(
                dst_width, dst_height, x_off, y_off, src_width, src_height, extend, bands,
            )?),
            BandFormatId::F32 => Box::new(EmbedBridge::<F32>::try_new(
                dst_width, dst_height, x_off, y_off, src_width, src_height, extend, bands,
            )?),
            BandFormatId::F64 => Box::new(EmbedBridge::<F64>::try_new(
                dst_width, dst_height, x_off, y_off, src_width, src_height, extend, bands,
            )?),
        };
        self.then(op)
    }

    /// Embed the image within a canvas using libvips-style compass gravity.
    ///
    /// `src_width` and `src_height` must match the current pipeline output dimensions.
    pub fn embed_with_gravity(
        self,
        dst_width: u32,
        dst_height: u32,
        gravity: Gravity,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        let bands = self.bands;
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(EmbedBridge::<U8>::try_with_gravity(
                dst_width, dst_height, gravity, src_width, src_height, extend, bands,
            )?),
            BandFormatId::U16 => Box::new(EmbedBridge::<U16>::try_with_gravity(
                dst_width, dst_height, gravity, src_width, src_height, extend, bands,
            )?),
            BandFormatId::I16 => Box::new(EmbedBridge::<I16>::try_with_gravity(
                dst_width, dst_height, gravity, src_width, src_height, extend, bands,
            )?),
            BandFormatId::U32 => Box::new(EmbedBridge::<U32>::try_with_gravity(
                dst_width, dst_height, gravity, src_width, src_height, extend, bands,
            )?),
            BandFormatId::I32 => Box::new(EmbedBridge::<I32>::try_with_gravity(
                dst_width, dst_height, gravity, src_width, src_height, extend, bands,
            )?),
            BandFormatId::F32 => Box::new(EmbedBridge::<F32>::try_with_gravity(
                dst_width, dst_height, gravity, src_width, src_height, extend, bands,
            )?),
            BandFormatId::F64 => Box::new(EmbedBridge::<F64>::try_with_gravity(
                dst_width, dst_height, gravity, src_width, src_height, extend, bands,
            )?),
        };
        self.then(op)
    }

    /// Flip the image horizontally (mirror left-right).
    ///
    /// Output dimensions are unchanged. Dispatches over the current format.
    pub fn flip_horizontal(self) -> Result<ImagePipeline<Committed>, BuildError> {
        let bands = self.bands;
        let (image_width, _) = self.current_dimensions();
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(OperationBridge::new(
                Flip::<U8>::horizontal(image_width),
                bands,
            )),
            BandFormatId::U16 => Box::new(OperationBridge::new(
                Flip::<U16>::horizontal(image_width),
                bands,
            )),
            BandFormatId::I16 => Box::new(OperationBridge::new(
                Flip::<I16>::horizontal(image_width),
                bands,
            )),
            BandFormatId::U32 => Box::new(OperationBridge::new(
                Flip::<U32>::horizontal(image_width),
                bands,
            )),
            BandFormatId::I32 => Box::new(OperationBridge::new(
                Flip::<I32>::horizontal(image_width),
                bands,
            )),
            BandFormatId::F32 => Box::new(OperationBridge::new(
                Flip::<F32>::horizontal(image_width),
                bands,
            )),
            BandFormatId::F64 => Box::new(OperationBridge::new(
                Flip::<F64>::horizontal(image_width),
                bands,
            )),
        };
        self.then(op)
    }

    /// Flip the image vertically (mirror top-bottom).
    ///
    /// Output dimensions are unchanged. Dispatches over the current format.
    pub fn flip_vertical(self) -> Result<ImagePipeline<Committed>, BuildError> {
        let bands = self.bands;
        let (_, image_height) = self.current_dimensions();
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(OperationBridge::new(
                Flip::<U8>::vertical(image_height),
                bands,
            )),
            BandFormatId::U16 => Box::new(OperationBridge::new(
                Flip::<U16>::vertical(image_height),
                bands,
            )),
            BandFormatId::I16 => Box::new(OperationBridge::new(
                Flip::<I16>::vertical(image_height),
                bands,
            )),
            BandFormatId::U32 => Box::new(OperationBridge::new(
                Flip::<U32>::vertical(image_height),
                bands,
            )),
            BandFormatId::I32 => Box::new(OperationBridge::new(
                Flip::<I32>::vertical(image_height),
                bands,
            )),
            BandFormatId::F32 => Box::new(OperationBridge::new(
                Flip::<F32>::vertical(image_height),
                bands,
            )),
            BandFormatId::F64 => Box::new(OperationBridge::new(
                Flip::<F64>::vertical(image_height),
                bands,
            )),
        };
        self.then(op)
    }

    /// Rotate the image by a multiple of 90 degrees clockwise.
    pub fn rot(self, angle: Angle) -> Result<ImagePipeline<Committed>, BuildError> {
        let (image_width, image_height) = self.current_dimensions();
        let bands = self.bands;
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(RotBridge::<U8>::new(
                image_width,
                image_height,
                angle,
                bands,
            )),
            BandFormatId::U16 => Box::new(RotBridge::<U16>::new(
                image_width,
                image_height,
                angle,
                bands,
            )),
            BandFormatId::I16 => Box::new(RotBridge::<I16>::new(
                image_width,
                image_height,
                angle,
                bands,
            )),
            BandFormatId::U32 => Box::new(RotBridge::<U32>::new(
                image_width,
                image_height,
                angle,
                bands,
            )),
            BandFormatId::I32 => Box::new(RotBridge::<I32>::new(
                image_width,
                image_height,
                angle,
                bands,
            )),
            BandFormatId::F32 => Box::new(RotBridge::<F32>::new(
                image_width,
                image_height,
                angle,
                bands,
            )),
            BandFormatId::F64 => Box::new(RotBridge::<F64>::new(
                image_width,
                image_height,
                angle,
                bands,
            )),
        };
        self.then(op)
    }

    /// Rotate the image by a multiple of 45 degrees.
    ///
    /// Right-angle rotations keep the exact integer mapping from [`ImagePipeline::rot`].
    /// Diagonal rotations expand the canvas automatically to cover the rotated image.
    pub fn rot45(self, angle: Angle45) -> Result<ImagePipeline<Committed>, BuildError> {
        if let Some(right_angle) = rot45_to_right_angle(angle) {
            return self.rot(right_angle);
        }

        let bands = self.bands;
        let (image_width, image_height) = self.current_dimensions();
        let angle_degrees = rot45_to_degrees(angle);

        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(SimilarityBridge::<U8>::new(
                1.0,
                angle_degrees,
                InterpolationKernel::Nearest,
                image_width,
                image_height,
                bands,
            )),
            BandFormatId::U16 => Box::new(SimilarityBridge::<U16>::new(
                1.0,
                angle_degrees,
                InterpolationKernel::Nearest,
                image_width,
                image_height,
                bands,
            )),
            BandFormatId::I16 => Box::new(SimilarityBridge::<I16>::new(
                1.0,
                angle_degrees,
                InterpolationKernel::Nearest,
                image_width,
                image_height,
                bands,
            )),
            BandFormatId::U32 => Box::new(SimilarityBridge::<U32>::new(
                1.0,
                angle_degrees,
                InterpolationKernel::Nearest,
                image_width,
                image_height,
                bands,
            )),
            BandFormatId::I32 => Box::new(SimilarityBridge::<I32>::new(
                1.0,
                angle_degrees,
                InterpolationKernel::Nearest,
                image_width,
                image_height,
                bands,
            )),
            BandFormatId::F32 => Box::new(SimilarityBridge::<F32>::new(
                1.0,
                angle_degrees,
                InterpolationKernel::Nearest,
                image_width,
                image_height,
                bands,
            )),
            BandFormatId::F64 => Box::new(SimilarityBridge::<F64>::new(
                1.0,
                angle_degrees,
                InterpolationKernel::Nearest,
                image_width,
                image_height,
                bands,
            )),
        };
        self.then(op)
    }

    /// Rotate the image 90° clockwise.
    ///
    /// Output dimensions are transposed: a W×H input becomes an H×W output.
    /// `compile()` propagates the new dimensions automatically via `output_width`/`output_height`
    /// declared by `Rotate90Bridge` — no manual `set_dimensions` call is needed.
    pub fn rotate90(self) -> Result<ImagePipeline<Committed>, BuildError> {
        self.rot(Angle::D90)
    }

    /// Rotate the image 180°.
    pub fn rotate180(self) -> Result<ImagePipeline<Committed>, BuildError> {
        self.rot(Angle::D180)
    }

    /// Rotate the image 270° clockwise (90° counter-clockwise).
    pub fn rotate270(self) -> Result<ImagePipeline<Committed>, BuildError> {
        self.rot(Angle::D270)
    }

    /// Tile the current image `across × down` times.
    pub fn replicate(self, across: u32, down: u32) -> Result<ImagePipeline<Committed>, BuildError> {
        if across == 0 || down == 0 {
            return Err(BuildError::SourceHint {
                context: "replicate",
                message: "across and down must be >= 1".to_string(),
            });
        }

        let bands = self.bands;
        let image_width = self.arena.width;
        let image_height = self.arena.height;
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(ReplicateBridge::<U8>::new(
                image_width,
                image_height,
                across,
                down,
                bands,
            )),
            BandFormatId::U16 => Box::new(ReplicateBridge::<U16>::new(
                image_width,
                image_height,
                across,
                down,
                bands,
            )),
            BandFormatId::I16 => Box::new(ReplicateBridge::<I16>::new(
                image_width,
                image_height,
                across,
                down,
                bands,
            )),
            BandFormatId::U32 => Box::new(ReplicateBridge::<U32>::new(
                image_width,
                image_height,
                across,
                down,
                bands,
            )),
            BandFormatId::I32 => Box::new(ReplicateBridge::<I32>::new(
                image_width,
                image_height,
                across,
                down,
                bands,
            )),
            BandFormatId::F32 => Box::new(ReplicateBridge::<F32>::new(
                image_width,
                image_height,
                across,
                down,
                bands,
            )),
            BandFormatId::F64 => Box::new(ReplicateBridge::<F64>::new(
                image_width,
                image_height,
                across,
                down,
                bands,
            )),
        };
        self.then(op)
    }

    /// Extract the most-significant byte from each integer band.
    pub fn msb(self) -> Result<ImagePipeline<Committed>, BuildError> {
        // TODO(fusion): integrate msb into Concretize chain.
        let bands = self.bands;
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => {
                Box::new(OperationBridge::new_pixel_local(MsbOp::<U8>::new(), bands))
            }
            BandFormatId::U16 => {
                Box::new(OperationBridge::new_pixel_local(MsbOp::<U16>::new(), bands))
            }
            BandFormatId::I16 => {
                Box::new(OperationBridge::new_pixel_local(MsbOp::<I16>::new(), bands))
            }
            BandFormatId::U32 => {
                Box::new(OperationBridge::new_pixel_local(MsbOp::<U32>::new(), bands))
            }
            BandFormatId::I32 => {
                Box::new(OperationBridge::new_pixel_local(MsbOp::<I32>::new(), bands))
            }
            format => {
                return Err(BuildError::UnsupportedFormat { op: "msb", format });
            }
        };
        self.then(op)
    }

    /// Rearrange a vertical strip of `tile_height`-tall frames into a grid.
    pub fn grid(
        self,
        tile_height: u32,
        across: u32,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        let bands = self.bands;
        let image_width = self.arena.width;
        let image_height = self.arena.height;
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(GridBridge::<U8>::new(
                image_width,
                image_height,
                tile_height,
                across,
                bands,
            )),
            BandFormatId::U16 => Box::new(GridBridge::<U16>::new(
                image_width,
                image_height,
                tile_height,
                across,
                bands,
            )),
            BandFormatId::I16 => Box::new(GridBridge::<I16>::new(
                image_width,
                image_height,
                tile_height,
                across,
                bands,
            )),
            BandFormatId::U32 => Box::new(GridBridge::<U32>::new(
                image_width,
                image_height,
                tile_height,
                across,
                bands,
            )),
            BandFormatId::I32 => Box::new(GridBridge::<I32>::new(
                image_width,
                image_height,
                tile_height,
                across,
                bands,
            )),
            BandFormatId::F32 => Box::new(GridBridge::<F32>::new(
                image_width,
                image_height,
                tile_height,
                across,
                bands,
            )),
            BandFormatId::F64 => Box::new(GridBridge::<F64>::new(
                image_width,
                image_height,
                tile_height,
                across,
                bands,
            )),
        };
        self.then(op)
    }

    /// Decimate by integer factors, taking every `xfac`-th / `yfac`-th pixel.
    pub fn subsample(self, xfac: u32, yfac: u32) -> Result<ImagePipeline<Committed>, BuildError> {
        let bands = self.bands;
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(SubsampleBridge::<U8>::new(xfac, yfac, bands)?),
            BandFormatId::U16 => Box::new(SubsampleBridge::<U16>::new(xfac, yfac, bands)?),
            BandFormatId::I16 => Box::new(SubsampleBridge::<I16>::new(xfac, yfac, bands)?),
            BandFormatId::U32 => Box::new(SubsampleBridge::<U32>::new(xfac, yfac, bands)?),
            BandFormatId::I32 => Box::new(SubsampleBridge::<I32>::new(xfac, yfac, bands)?),
            BandFormatId::F32 => Box::new(SubsampleBridge::<F32>::new(xfac, yfac, bands)?),
            BandFormatId::F64 => Box::new(SubsampleBridge::<F64>::new(xfac, yfac, bands)?),
        };
        self.then(op)
    }

    /// Integer nearest-neighbour upscale.
    pub fn zoom(self, xfac: u32, yfac: u32) -> Result<ImagePipeline<Committed>, BuildError> {
        if xfac == 0 || yfac == 0 {
            return Err(BuildError::SourceHint {
                context: "zoom",
                message: "xfac and yfac must be >= 1".to_string(),
            });
        }

        let bands = self.bands;
        if bands == 0 {
            return Err(BuildError::SourceHint {
                context: "zoom",
                message: "band count must be >= 1".to_string(),
            });
        }
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(ZoomBridge::<U8>::new(xfac, yfac, bands)),
            BandFormatId::U16 => Box::new(ZoomBridge::<U16>::new(xfac, yfac, bands)),
            BandFormatId::I16 => Box::new(ZoomBridge::<I16>::new(xfac, yfac, bands)),
            BandFormatId::U32 => Box::new(ZoomBridge::<U32>::new(xfac, yfac, bands)),
            BandFormatId::I32 => Box::new(ZoomBridge::<I32>::new(xfac, yfac, bands)),
            BandFormatId::F32 => Box::new(ZoomBridge::<F32>::new(xfac, yfac, bands)),
            BandFormatId::F64 => Box::new(ZoomBridge::<F64>::new(xfac, yfac, bands)),
        };
        self.then(op)
    }

    /// Wrap image origin so input pixel `(0, 0)` appears at `(x, y)`.
    pub fn wrap(self, x: i32, y: i32) -> Result<ImagePipeline<Committed>, BuildError> {
        let bands = self.bands;
        let image_width = self.arena.width;
        let image_height = self.arena.height;

        if image_width == 0 || image_height == 0 {
            return Err(BuildError::SourceHint {
                context: "wrap",
                message: "image width and height must be >= 1".to_string(),
            });
        }

        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(OperationBridge::new(
                Wrap::<U8>::new(image_width, image_height, x, y),
                bands,
            )),
            BandFormatId::U16 => Box::new(OperationBridge::new(
                Wrap::<U16>::new(image_width, image_height, x, y),
                bands,
            )),
            BandFormatId::I16 => Box::new(OperationBridge::new(
                Wrap::<I16>::new(image_width, image_height, x, y),
                bands,
            )),
            BandFormatId::U32 => Box::new(OperationBridge::new(
                Wrap::<U32>::new(image_width, image_height, x, y),
                bands,
            )),
            BandFormatId::I32 => Box::new(OperationBridge::new(
                Wrap::<I32>::new(image_width, image_height, x, y),
                bands,
            )),
            BandFormatId::F32 => Box::new(OperationBridge::new(
                Wrap::<F32>::new(image_width, image_height, x, y),
                bands,
            )),
            BandFormatId::F64 => Box::new(OperationBridge::new(
                Wrap::<F64>::new(image_width, image_height, x, y),
                bands,
            )),
        };
        self.then(op)
    }
}
