use viprs_core::error::BuildError;

use super::super::{
    Angle, Angle45, ExtendMode, Gravity,
    pipeline::{CommitState, Committed, ImagePipeline},
};

impl<State> ImagePipeline<State>
where
    State: CommitState,
{
    /// Crop the image to a source-coordinate rectangle.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the rectangle is outside the current image.
    pub fn extract_area(
        self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.extract_area(x, y, width, height)?,
        ))
    }

    /// Embed the image in a larger canvas.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when dimensions or offsets are invalid.
    #[allow(clippy::too_many_arguments)]
    // REASON: Mirrors the existing libvips-style embed parameter contract.
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
        Ok(ImagePipeline::from_builder(self.commit()?.builder.embed(
            dst_width, dst_height, x_off, y_off, src_width, src_height, extend,
        )?))
    }

    /// Embed the image with signed offsets.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when dimensions or offsets are invalid.
    #[allow(clippy::too_many_arguments)]
    // REASON: Mirrors the existing libvips-style embed parameter contract.
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
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.embed_signed(
                dst_width, dst_height, x_off, y_off, src_width, src_height, extend,
            )?,
        ))
    }

    /// Embed the image in a larger canvas using compass gravity.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when dimensions are invalid.
    pub fn embed_with_gravity(
        self,
        dst_width: u32,
        dst_height: u32,
        gravity: Gravity,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.embed_with_gravity(
                dst_width, dst_height, gravity, src_width, src_height, extend,
            )?,
        ))
    }

    /// Flip the image horizontally.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current pipeline cannot accept the operation.
    pub fn flip_horizontal(self) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.flip_horizontal()?,
        ))
    }

    /// Flip the image vertically.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current pipeline cannot accept the operation.
    pub fn flip_vertical(self) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.flip_vertical()?,
        ))
    }

    /// Rotate the image by a right angle.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current pipeline cannot accept the operation.
    pub fn rot(self, angle: Angle) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.rot(angle)?,
        ))
    }

    /// Rotate the image by a multiple of 45 degrees.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current pipeline cannot accept the operation.
    pub fn rot45(self, angle: Angle45) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.rot45(angle)?,
        ))
    }

    /// Alias for [`ImagePipeline::rot`].
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current pipeline cannot accept the operation.
    pub fn rotate(self, angle: Angle) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.rotate(angle)?,
        ))
    }

    /// Rotate the image 90 degrees clockwise.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current pipeline cannot accept the operation.
    pub fn rotate90(self) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.rotate90()?,
        ))
    }

    /// Rotate the image 180 degrees.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current pipeline cannot accept the operation.
    pub fn rotate180(self) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.rotate180()?,
        ))
    }

    /// Rotate the image 270 degrees clockwise.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current pipeline cannot accept the operation.
    pub fn rotate270(self) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.rotate270()?,
        ))
    }

    /// Tile the current image.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when factors are invalid.
    pub fn replicate(self, across: u32, down: u32) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.replicate(across, down)?,
        ))
    }

    /// Rearrange a vertical strip into a grid.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current pipeline cannot accept the operation.
    pub fn grid(
        self,
        tile_height: u32,
        across: u32,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.grid(tile_height, across)?,
        ))
    }

    /// Decimate by integer factors.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when factors are invalid.
    pub fn subsample(self, xfac: u32, yfac: u32) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.subsample(xfac, yfac)?,
        ))
    }

    /// Upscale with nearest-neighbour integer factors.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when factors are invalid.
    pub fn zoom(self, xfac: u32, yfac: u32) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.zoom(xfac, yfac)?,
        ))
    }

    /// Wrap the image origin.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current image is empty.
    pub fn wrap(self, x: i32, y: i32) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.wrap(x, y)?,
        ))
    }
}
