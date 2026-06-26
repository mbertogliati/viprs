use super::{
  BandFormatId, BuildError, Cast, Colorspace, ColorspaceId, ColourspaceDispatcher, Conv2d,
  DynOperation, F32, F64, Commit, GaussBlurH, GaussBlurV, GaussOutputFormat, I16, I32, Committed,
  Interpretation, Lab, LabSSharpen, LabSToLab, LabToLabS, OperationBridge, ImagePipeline, U8,
  U16, U32,
};

impl<Op: Commit> ImagePipeline<Op> {
    /// Insert a `Cast` operation converting the current format to `target`.
    ///
    /// Only combinations with a `CastSample` impl are supported. Unsupported pairs
    /// return `BuildError::UnsupportedFormat`.
    pub fn cast(self, target: BandFormatId) -> Result<ImagePipeline<Committed>, BuildError> {
        // TODO(fusion): integrate cast into Concretize chain.
        let bands = self.bands;
        let source_fmt = self.current_format;
        let op: Box<dyn DynOperation> =
            match (source_fmt, target) {
                (BandFormatId::U8, BandFormatId::U8) => Box::new(OperationBridge::new_pixel_local(
                    Cast::<U8, U8>::new(bands),
                    bands,
                )),
                (BandFormatId::U8, BandFormatId::F32) => Box::new(
                    OperationBridge::new_pixel_local(Cast::<U8, F32>::new(bands), bands),
                ),
                (BandFormatId::U8, BandFormatId::U16) => Box::new(
                    OperationBridge::new_pixel_local(Cast::<U8, U16>::new(bands), bands),
                ),
                (BandFormatId::U16, BandFormatId::F32) => Box::new(
                    OperationBridge::new_pixel_local(Cast::<U16, F32>::new(bands), bands),
                ),
                (BandFormatId::F32, BandFormatId::F32) => Box::new(
                    OperationBridge::new_pixel_local(Cast::<F32, F32>::new(bands), bands),
                ),
                (BandFormatId::F32, BandFormatId::U8) => Box::new(
                    OperationBridge::new_pixel_local(Cast::<F32, U8>::new(bands), bands),
                ),
                (BandFormatId::F32, BandFormatId::F64) => Box::new(
                    OperationBridge::new_pixel_local(Cast::<F32, F64>::new(bands), bands),
                ),
                (BandFormatId::F64, BandFormatId::F32) => Box::new(
                    OperationBridge::new_pixel_local(Cast::<F64, F32>::new(bands), bands),
                ),
                (_, _) => {
                    return Err(BuildError::UnsupportedFormat {
                        op: "cast",
                        format: source_fmt,
                    });
                }
            };
        self.then(op)
    }

    /// Apply a 2D convolution with `kernel` to the image.
    ///
    /// The output format is always F32 regardless of the input format. The caller
    /// must cast back to a different format downstream if needed.
    ///
    /// Only F32 input is supported in this MVP. For other input formats, cast to F32
    /// first with `cast(BandFormatId::F32)`. Returns `BuildError::UnsupportedFormat`
    /// for unsupported input formats.
    pub fn conv2d(self, kernel: Vec<Vec<f64>>) -> Result<ImagePipeline<Committed>, BuildError> {
        let bands = self.bands;
        let source_fmt = self.current_format;
        let op: Box<dyn DynOperation> = match source_fmt {
            BandFormatId::U8 => {
                let conv =
                    Conv2d::<U8>::new(kernel).map_err(|_| BuildError::UnsupportedFormat {
                        op: "conv2d",
                        format: source_fmt,
                    })?;
                Box::new(OperationBridge::new(conv, bands))
            }
            BandFormatId::U16 => {
                let conv =
                    Conv2d::<U16>::new(kernel).map_err(|_| BuildError::UnsupportedFormat {
                        op: "conv2d",
                        format: source_fmt,
                    })?;
                Box::new(OperationBridge::new(conv, bands))
            }
            BandFormatId::I16 => {
                let conv =
                    Conv2d::<I16>::new(kernel).map_err(|_| BuildError::UnsupportedFormat {
                        op: "conv2d",
                        format: source_fmt,
                    })?;
                Box::new(OperationBridge::new(conv, bands))
            }
            BandFormatId::U32 => {
                let conv =
                    Conv2d::<U32>::new(kernel).map_err(|_| BuildError::UnsupportedFormat {
                        op: "conv2d",
                        format: source_fmt,
                    })?;
                Box::new(OperationBridge::new(conv, bands))
            }
            BandFormatId::I32 => {
                let conv =
                    Conv2d::<I32>::new(kernel).map_err(|_| BuildError::UnsupportedFormat {
                        op: "conv2d",
                        format: source_fmt,
                    })?;
                Box::new(OperationBridge::new(conv, bands))
            }
            BandFormatId::F32 => {
                let conv =
                    Conv2d::<F32>::new(kernel).map_err(|_| BuildError::UnsupportedFormat {
                        op: "conv2d",
                        format: source_fmt,
                    })?;
                Box::new(OperationBridge::new(conv, bands))
            }
            BandFormatId::F64 => {
                let conv =
                    Conv2d::<F64>::new(kernel).map_err(|_| BuildError::UnsupportedFormat {
                        op: "conv2d",
                        format: source_fmt,
                    })?;
                Box::new(OperationBridge::new(conv, bands))
            }
        };
        self.then(op)
    }

    /// Apply a separable Gaussian blur with the library-selected intermediate format.
    ///
    /// `U8` stays on the fixed-point path for both passes. All other formats use an
    /// `F32` intermediate selected by [`GaussOutputFormat`].
    pub fn gauss_blur(self, sigma: f32) -> Result<ImagePipeline<Committed>, BuildError> {
        let bands = self.bands;
        match self.current_format {
            BandFormatId::U8 => self
                .then(Box::new(OperationBridge::new(
                    GaussBlurH::<U8>::new(sigma),
                    bands,
                )))?
                .then(Box::new(OperationBridge::new(
                    GaussBlurV::<U8>::new(sigma),
                    bands,
                ))),
            BandFormatId::U16 => self
                .then(Box::new(OperationBridge::new(
                    GaussBlurH::<U16>::new(sigma),
                    bands,
                )))?
                .then(Box::new(OperationBridge::new(
                    GaussBlurV::<GaussOutputFormat<U16>>::new(sigma),
                    bands,
                ))),
            BandFormatId::I16 => self
                .then(Box::new(OperationBridge::new(
                    GaussBlurH::<I16>::new(sigma),
                    bands,
                )))?
                .then(Box::new(OperationBridge::new(
                    GaussBlurV::<GaussOutputFormat<I16>>::new(sigma),
                    bands,
                ))),
            BandFormatId::U32 => self
                .then(Box::new(OperationBridge::new(
                    GaussBlurH::<U32>::new(sigma),
                    bands,
                )))?
                .then(Box::new(OperationBridge::new(
                    GaussBlurV::<GaussOutputFormat<U32>>::new(sigma),
                    bands,
                ))),
            BandFormatId::I32 => self
                .then(Box::new(OperationBridge::new(
                    GaussBlurH::<I32>::new(sigma),
                    bands,
                )))?
                .then(Box::new(OperationBridge::new(
                    GaussBlurV::<GaussOutputFormat<I32>>::new(sigma),
                    bands,
                ))),
            BandFormatId::F32 => self
                .then(Box::new(OperationBridge::new(
                    GaussBlurH::<F32>::new(sigma),
                    bands,
                )))?
                .then(Box::new(OperationBridge::new(
                    GaussBlurV::<GaussOutputFormat<F32>>::new(sigma),
                    bands,
                ))),
            BandFormatId::F64 => self
                .then(Box::new(OperationBridge::new(
                    GaussBlurH::<F64>::new(sigma),
                    bands,
                )))?
                .then(Box::new(OperationBridge::new(
                    GaussBlurV::<GaussOutputFormat<F64>>::new(sigma),
                    bands,
                ))),
        }
    }

    /// Apply libvips-style LabS-aware sharpen semantics.
    ///
    /// The builder converts the current image to `Lab`, quantizes to `LabS`, sharpens only
    /// the `L` channel, converts back to `Lab`, then restores the original colorspace.
    pub fn sharpen(
        self,
        sigma: f32,
        x1: f32,
        y2: f32,
        y3: f32,
        m1: f32,
        m2: f32,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        let builder = self.flush_into_identity()?;
        let original_colorspace = builder
            .current_colorspace
            .ok_or(BuildError::UnknownColorspace)?;

        let mut builder = if original_colorspace == ColorspaceId::Lab {
            builder
        } else {
            builder.colourspace::<Lab>()?
        };

        let bands = builder.bands;
        builder = builder.then(Box::new(OperationBridge::new_pixel_local(LabToLabS, bands)))?;
        builder = builder.then(Box::new(OperationBridge::new(
            LabSSharpen::new(sigma, x1, y2, y3, m1, m2),
            bands,
        )))?;
        builder = builder.then(Box::new(OperationBridge::new_pixel_local(LabSToLab, bands)))?;

        if original_colorspace == ColorspaceId::Lab {
            Ok(builder)
        } else {
            builder.colourspace_to(original_colorspace)
        }
    }

    /// Convert the current image to colorspace `To`.
    ///
    /// Uses the centralized `ColourspaceDispatcher` BFS graph to find the shortest
    /// conversion chain. Returns `BuildError::UnknownColorspace` if the current
    /// colorspace is `None`, and `BuildError::UnsupportedColourConversion` for
    /// pairs without a registered path.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipeline = PipelineBuilder::from_source(source)
    ///     .with_colorspace(ColorspaceId::SRgb)
    ///     .colourspace::<Lab>()?
    ///     .build()?;
    /// ```
    pub fn colourspace<To: Colorspace>(self) -> Result<ImagePipeline<Committed>, BuildError> {
        self.colourspace_to(To::ID)
    }

    fn colourspace_to(self, to: ColorspaceId) -> Result<ImagePipeline<Committed>, BuildError> {
        let builder = self.flush_into_identity()?;
        let from = builder
            .current_colorspace
            .ok_or(BuildError::UnknownColorspace)?;
        if from == to {
            return Ok(builder);
        }

        let ops = ColourspaceDispatcher::new()
            .build_path(from, to, builder.bands, builder.current_format)?
            .ok_or(BuildError::UnsupportedColourConversion { from, to })?;
        let mut builder = builder;
        for op in ops {
            builder = builder.then(op)?;
        }

        Ok(builder)
    }
}

#[inline]
pub(in crate::pipeline::builder) const fn interpretation_to_colorspace(
    interpretation: Interpretation,
) -> Option<ColorspaceId> {
    match interpretation {
        Interpretation::BW | Interpretation::Grey16 => Some(ColorspaceId::Greyscale),
        Interpretation::Xyz => Some(ColorspaceId::Xyz),
        Interpretation::Lab => Some(ColorspaceId::Lab),
        Interpretation::Cmyk => Some(ColorspaceId::Cmyk),
        Interpretation::Cmc => Some(ColorspaceId::Ucs),
        Interpretation::Lch => Some(ColorspaceId::Lch),
        Interpretation::Srgb | Interpretation::Rgb => Some(ColorspaceId::SRgb),
        Interpretation::Yxy => Some(ColorspaceId::Yxy),
        Interpretation::Rgb16 => Some(ColorspaceId::Rgb16),
        Interpretation::Scrgb => Some(ColorspaceId::ScRgb),
        Interpretation::Hsv => Some(ColorspaceId::Hsv),
        _ => None,
    }
}
