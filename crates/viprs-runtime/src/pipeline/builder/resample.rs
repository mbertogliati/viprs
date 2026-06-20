use super::state::{validate_reduce_factors, validate_reduce_kernel};
use super::{
    AffineBridge, BandFormatId, BuildError, ColorspaceId, CopyOp, DemandHint, DynOperation, F32,
    F64, FlattenBridge, Flush, I16, I32, Identity, InterpolationKernel, Interpretation, NonZeroU8,
    OperationBridge, PipelineBuilder, Premultiply, ReduceBridge, ReduceHBridge, ReduceVBridge,
    Resize, ResizeNode, ShrinkBridge, ShrinkHBridge, ShrinkVBridge, SimilarityBridge, Thumbnail,
    ThumbnailNode, U8, U16, U32, Unpremultiply, flatten_has_alpha,
};

impl<Op: Flush> PipelineBuilder<Op> {
    /// Reduce the image width by `factor` using `kernel` (horizontal downscale).
    ///
    /// `factor` must be >= 1.0. Output width is `round(input_width / factor)`.
    /// `compile()` propagates the new dimensions automatically via `output_width`
    /// declared by `ReduceHBridge` — no manual `set_dimensions` call is needed.
    pub fn reduce_h(
        self,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        self.reduce_h_with_hint(factor, kernel, DemandHint::ThinStrip)
    }

    #[inline]
    fn forward_affine_to_backward(
        matrix: [f64; 4],
        tx: f64,
        ty: f64,
        output_w: u32,
        output_h: u32,
    ) -> Result<([f64; 4], f64, f64), BuildError> {
        if !matrix.iter().all(|value| value.is_finite()) || !tx.is_finite() || !ty.is_finite() {
            return Err(BuildError::InvalidAffineMatrix {
                matrix,
                reason: "matrix coefficients and translation must be finite",
            });
        }

        let determinant = matrix[0].mul_add(matrix[3], -matrix[1] * matrix[2]);
        if determinant.abs() <= f64::EPSILON {
            return Err(BuildError::DegenerateAffineTransform {
                matrix,
                output_width: output_w,
                output_height: output_h,
                reason: "matrix determinant is singular",
            });
        }

        let inv_det = 1.0 / determinant;
        let backward_matrix = [
            matrix[3] * inv_det,
            -matrix[1] * inv_det,
            -matrix[2] * inv_det,
            matrix[0] * inv_det,
        ];
        let backward_tx = -backward_matrix[1].mul_add(ty, backward_matrix[0] * tx);
        let backward_ty = -backward_matrix[3].mul_add(ty, backward_matrix[2] * tx);

        Ok((backward_matrix, backward_tx, backward_ty))
    }

    fn reduce_h_with_hint(
        self,
        factor: f64,
        kernel: InterpolationKernel,
        demand_hint: DemandHint,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        validate_reduce_kernel("reduce_h", kernel)?;
        let bands = self.bands;
        let (input_w, _) = self.current_dimensions();
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(ReduceHBridge::<U8>::new(
                factor,
                kernel,
                bands,
                input_w,
                demand_hint,
            )?),
            BandFormatId::U16 => Box::new(ReduceHBridge::<U16>::new(
                factor,
                kernel,
                bands,
                input_w,
                demand_hint,
            )?),
            BandFormatId::I16 => Box::new(ReduceHBridge::<I16>::new(
                factor,
                kernel,
                bands,
                input_w,
                demand_hint,
            )?),
            BandFormatId::U32 => Box::new(ReduceHBridge::<U32>::new(
                factor,
                kernel,
                bands,
                input_w,
                demand_hint,
            )?),
            BandFormatId::I32 => Box::new(ReduceHBridge::<I32>::new(
                factor,
                kernel,
                bands,
                input_w,
                demand_hint,
            )?),
            BandFormatId::F32 => Box::new(ReduceHBridge::<F32>::new(
                factor,
                kernel,
                bands,
                input_w,
                demand_hint,
            )?),
            BandFormatId::F64 => Box::new(ReduceHBridge::<F64>::new(
                factor,
                kernel,
                bands,
                input_w,
                demand_hint,
            )?),
        };
        self.then(op)
    }

    /// `shrink_h` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::shrink_h;
    /// ```
    pub fn shrink_h(self, factor: u32) -> Result<PipelineBuilder<Identity>, BuildError> {
        self.shrink_h_with_ceil(factor, false)
    }

    /// `shrink_h_with_ceil` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::shrink_h_with_ceil;
    /// ```
    pub fn shrink_h_with_ceil(
        self,
        factor: u32,
        ceil: bool,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        let bands = self.bands;
        if factor == 0 {
            return Err(BuildError::SourceHint {
                context: "shrink_h",
                message: "factor must be >= 1".to_string(),
            });
        }
        if bands == 0 {
            return Err(BuildError::SourceHint {
                context: "shrink_h",
                message: "band count must be >= 1".to_string(),
            });
        }
        let source_width = self.current_dimensions().0;
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(ShrinkHBridge::<U8>::new_with_ceil_and_source_width(
                factor,
                ceil,
                bands,
                Some(source_width),
            )?),
            BandFormatId::U16 => Box::new(ShrinkHBridge::<U16>::new_with_ceil_and_source_width(
                factor,
                ceil,
                bands,
                Some(source_width),
            )?),
            BandFormatId::I16 => Box::new(ShrinkHBridge::<I16>::new_with_ceil_and_source_width(
                factor,
                ceil,
                bands,
                Some(source_width),
            )?),
            BandFormatId::U32 => Box::new(ShrinkHBridge::<U32>::new_with_ceil_and_source_width(
                factor,
                ceil,
                bands,
                Some(source_width),
            )?),
            BandFormatId::I32 => Box::new(ShrinkHBridge::<I32>::new_with_ceil_and_source_width(
                factor,
                ceil,
                bands,
                Some(source_width),
            )?),
            BandFormatId::F32 => Box::new(ShrinkHBridge::<F32>::new_with_ceil_and_source_width(
                factor,
                ceil,
                bands,
                Some(source_width),
            )?),
            BandFormatId::F64 => Box::new(ShrinkHBridge::<F64>::new_with_ceil_and_source_width(
                factor,
                ceil,
                bands,
                Some(source_width),
            )?),
        };
        self.then(op)
    }

    /// `shrink` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::shrink;
    /// ```
    pub fn shrink(
        self,
        h_factor: u32,
        v_factor: u32,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        let bands = self.bands;
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(ShrinkBridge::<U8>::new(
                h_factor as usize,
                v_factor as usize,
                bands,
            )?),
            BandFormatId::U16 => Box::new(ShrinkBridge::<U16>::new(
                h_factor as usize,
                v_factor as usize,
                bands,
            )?),
            BandFormatId::I16 => Box::new(ShrinkBridge::<I16>::new(
                h_factor as usize,
                v_factor as usize,
                bands,
            )?),
            BandFormatId::U32 => Box::new(ShrinkBridge::<U32>::new(
                h_factor as usize,
                v_factor as usize,
                bands,
            )?),
            BandFormatId::I32 => Box::new(ShrinkBridge::<I32>::new(
                h_factor as usize,
                v_factor as usize,
                bands,
            )?),
            BandFormatId::F32 => Box::new(ShrinkBridge::<F32>::new(
                h_factor as usize,
                v_factor as usize,
                bands,
            )?),
            BandFormatId::F64 => Box::new(ShrinkBridge::<F64>::new(
                h_factor as usize,
                v_factor as usize,
                bands,
            )?),
        };
        self.then(op)
    }

    /// Reduce the image height by `factor` using `kernel` (vertical downscale).
    ///
    /// `factor` must be >= 1.0. Output height is `round(input_height / factor)`.
    /// `compile()` propagates the new dimensions automatically via `output_height`
    /// declared by `ReduceVBridge` — no manual `set_dimensions` call is needed.
    pub fn reduce_v(
        self,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        self.reduce_v_with_hint(factor, kernel, DemandHint::ThinStrip)
    }

    fn reduce_v_with_hint(
        self,
        factor: f64,
        kernel: InterpolationKernel,
        demand_hint: DemandHint,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        validate_reduce_kernel("reduce_v", kernel)?;
        let bands = self.bands;
        let (_, input_h) = self.current_dimensions();
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(ReduceVBridge::<U8>::new(
                factor,
                kernel,
                bands,
                input_h,
                demand_hint,
            )?),
            BandFormatId::U16 => Box::new(ReduceVBridge::<U16>::new(
                factor,
                kernel,
                bands,
                input_h,
                demand_hint,
            )?),
            BandFormatId::I16 => Box::new(ReduceVBridge::<I16>::new(
                factor,
                kernel,
                bands,
                input_h,
                demand_hint,
            )?),
            BandFormatId::U32 => Box::new(ReduceVBridge::<U32>::new(
                factor,
                kernel,
                bands,
                input_h,
                demand_hint,
            )?),
            BandFormatId::I32 => Box::new(ReduceVBridge::<I32>::new(
                factor,
                kernel,
                bands,
                input_h,
                demand_hint,
            )?),
            BandFormatId::F32 => Box::new(ReduceVBridge::<F32>::new(
                factor,
                kernel,
                bands,
                input_h,
                demand_hint,
            )?),
            BandFormatId::F64 => Box::new(ReduceVBridge::<F64>::new(
                factor,
                kernel,
                bands,
                input_h,
                demand_hint,
            )?),
        };
        self.then(op)
    }

    /// `reduce` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::reduce;
    /// ```
    pub fn reduce(
        self,
        h_factor: f64,
        v_factor: f64,
        kernel: InterpolationKernel,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        validate_reduce_factors(h_factor, v_factor)?;
        validate_reduce_kernel("reduce", kernel)?;
        let bands = self.bands;
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => {
                Box::new(ReduceBridge::<U8>::new(h_factor, v_factor, kernel, bands)?)
            }
            BandFormatId::U16 => {
                Box::new(ReduceBridge::<U16>::new(h_factor, v_factor, kernel, bands)?)
            }
            BandFormatId::I16 => {
                Box::new(ReduceBridge::<I16>::new(h_factor, v_factor, kernel, bands)?)
            }
            BandFormatId::U32 => {
                Box::new(ReduceBridge::<U32>::new(h_factor, v_factor, kernel, bands)?)
            }
            BandFormatId::I32 => {
                Box::new(ReduceBridge::<I32>::new(h_factor, v_factor, kernel, bands)?)
            }
            BandFormatId::F32 => {
                Box::new(ReduceBridge::<F32>::new(h_factor, v_factor, kernel, bands)?)
            }
            BandFormatId::F64 => {
                Box::new(ReduceBridge::<F64>::new(h_factor, v_factor, kernel, bands)?)
            }
        };
        self.then(op)
    }

    /// `shrink_v` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::shrink_v;
    /// ```
    pub fn shrink_v(self, factor: u32) -> Result<PipelineBuilder<Identity>, BuildError> {
        self.shrink_v_with_ceil(factor, false)
    }

    /// `shrink_v_with_ceil` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::shrink_v_with_ceil;
    /// ```
    pub fn shrink_v_with_ceil(
        self,
        factor: u32,
        ceil: bool,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        let bands = self.bands;
        if factor == 0 {
            return Err(BuildError::SourceHint {
                context: "shrink_v",
                message: "factor must be >= 1".to_string(),
            });
        }
        if bands == 0 {
            return Err(BuildError::SourceHint {
                context: "shrink_v",
                message: "band count must be >= 1".to_string(),
            });
        }
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(ShrinkVBridge::<U8>::new_with_ceil(factor, ceil, bands)?),
            BandFormatId::U16 => {
                Box::new(ShrinkVBridge::<U16>::new_with_ceil(factor, ceil, bands)?)
            }
            BandFormatId::I16 => {
                Box::new(ShrinkVBridge::<I16>::new_with_ceil(factor, ceil, bands)?)
            }
            BandFormatId::U32 => {
                Box::new(ShrinkVBridge::<U32>::new_with_ceil(factor, ceil, bands)?)
            }
            BandFormatId::I32 => {
                Box::new(ShrinkVBridge::<I32>::new_with_ceil(factor, ceil, bands)?)
            }
            BandFormatId::F32 => {
                Box::new(ShrinkVBridge::<F32>::new_with_ceil(factor, ceil, bands)?)
            }
            BandFormatId::F64 => {
                Box::new(ShrinkVBridge::<F64>::new_with_ceil(factor, ceil, bands)?)
            }
        };
        self.then(op)
    }

    /// Apply an affine transform to the image.
    ///
    /// `matrix` is `[a, b, c, d]` (row-major 2×2) in forward form:
    /// `x_out = a*x_in + b*y_in + tx`, `y_out = c*x_in + d*y_in + ty`.
    /// The builder converts this to the internal backward-mapped affine kernel,
    /// so positive translations move the image right/down and expose zero-filled
    /// background on the uncovered edges.
    /// `output_w` and `output_h` fix the output image dimensions; the pipeline
    /// compiler propagates these via `AffineBridge::output_width`/`output_height`.
    pub fn affine(
        self,
        matrix: [f64; 4],
        tx: f64,
        ty: f64,
        output_w: u32,
        output_h: u32,
        kernel: InterpolationKernel,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        let (backward_matrix, backward_tx, backward_ty) =
            Self::forward_affine_to_backward(matrix, tx, ty, output_w, output_h)?;
        self.affine_backward_with_extend_and_hint(
            backward_matrix,
            backward_tx,
            backward_ty,
            output_w,
            output_h,
            kernel,
            DemandHint::SmallTile,
            crate::domain::ops::resample::affine::ExtendMode::Background(vec![0.0]),
        )
    }

    fn affine_backward_with_extend_and_hint(
        self,
        matrix: [f64; 4],
        tx: f64,
        ty: f64,
        output_w: u32,
        output_h: u32,
        kernel: InterpolationKernel,
        demand_hint: DemandHint,
        extend: crate::domain::ops::resample::affine::ExtendMode,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        let bands = self.bands;
        let (input_w, input_h) = self.current_dimensions();
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(AffineBridge::<U8>::new_with_extend(
                matrix,
                tx,
                ty,
                kernel,
                input_w,
                input_h,
                output_w,
                output_h,
                bands,
                demand_hint,
                extend,
            )?),
            BandFormatId::U16 => Box::new(AffineBridge::<U16>::new_with_extend(
                matrix,
                tx,
                ty,
                kernel,
                input_w,
                input_h,
                output_w,
                output_h,
                bands,
                demand_hint,
                extend,
            )?),
            BandFormatId::I16 => Box::new(AffineBridge::<I16>::new_with_extend(
                matrix,
                tx,
                ty,
                kernel,
                input_w,
                input_h,
                output_w,
                output_h,
                bands,
                demand_hint,
                extend,
            )?),
            BandFormatId::U32 => Box::new(AffineBridge::<U32>::new_with_extend(
                matrix,
                tx,
                ty,
                kernel,
                input_w,
                input_h,
                output_w,
                output_h,
                bands,
                demand_hint,
                extend,
            )?),
            BandFormatId::I32 => Box::new(AffineBridge::<I32>::new_with_extend(
                matrix,
                tx,
                ty,
                kernel,
                input_w,
                input_h,
                output_w,
                output_h,
                bands,
                demand_hint,
                extend,
            )?),
            BandFormatId::F32 => Box::new(AffineBridge::<F32>::new_with_extend(
                matrix,
                tx,
                ty,
                kernel,
                input_w,
                input_h,
                output_w,
                output_h,
                bands,
                demand_hint,
                extend,
            )?),
            BandFormatId::F64 => Box::new(AffineBridge::<F64>::new_with_extend(
                matrix,
                tx,
                ty,
                kernel,
                input_w,
                input_h,
                output_w,
                output_h,
                bands,
                demand_hint,
                extend,
            )?),
        };
        self.then(op)
    }

    /// `similarity` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::similarity;
    /// ```
    pub fn similarity(
        self,
        scale: f64,
        angle: f64,
        kernel: InterpolationKernel,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        let bands = self.bands;
        let (input_w, input_h) = self.current_dimensions();
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(SimilarityBridge::<U8>::new(
                scale, angle, kernel, input_w, input_h, bands,
            )),
            BandFormatId::U16 => Box::new(SimilarityBridge::<U16>::new(
                scale, angle, kernel, input_w, input_h, bands,
            )),
            BandFormatId::I16 => Box::new(SimilarityBridge::<I16>::new(
                scale, angle, kernel, input_w, input_h, bands,
            )),
            BandFormatId::U32 => Box::new(SimilarityBridge::<U32>::new(
                scale, angle, kernel, input_w, input_h, bands,
            )),
            BandFormatId::I32 => Box::new(SimilarityBridge::<I32>::new(
                scale, angle, kernel, input_w, input_h, bands,
            )),
            BandFormatId::F32 => Box::new(SimilarityBridge::<F32>::new(
                scale, angle, kernel, input_w, input_h, bands,
            )),
            BandFormatId::F64 => Box::new(SimilarityBridge::<F64>::new(
                scale, angle, kernel, input_w, input_h, bands,
            )),
        };
        self.then(op)
    }

    /// `resize` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::resize;
    /// ```
    pub fn resize(self, resize: Resize) -> Result<PipelineBuilder<Identity>, BuildError> {
        if self.bands == 0 {
            return Err(BuildError::SourceHint {
                context: "resize",
                message: "band count must be >= 1".to_string(),
            });
        }

        let plan = resize.into_pipeline_nodes(self.arena.width, self.arena.height);

        if plan.nodes.is_empty() {
            let bands = self.bands;
            let op: Box<dyn DynOperation> = match self.current_format {
                BandFormatId::U8 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<U8>::default(),
                    bands,
                )),
                BandFormatId::U16 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<U16>::default(),
                    bands,
                )),
                BandFormatId::I16 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<I16>::default(),
                    bands,
                )),
                BandFormatId::U32 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<U32>::default(),
                    bands,
                )),
                BandFormatId::I32 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<I32>::default(),
                    bands,
                )),
                BandFormatId::F32 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<F32>::default(),
                    bands,
                )),
                BandFormatId::F64 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<F64>::default(),
                    bands,
                )),
            };
            return self.then(op);
        }

        let mut builder = self.flush_into_identity()?;
        for node in plan.nodes {
            builder = match node {
                ResizeNode::ShrinkH { factor } => builder.shrink_h(factor)?,
                ResizeNode::ShrinkV { factor } => builder.shrink_v(factor)?,
                ResizeNode::ReduceH { factor, kernel } => builder.reduce_h(factor, kernel)?,
                ResizeNode::ReduceV { factor, kernel } => builder.reduce_v(factor, kernel)?,
                ResizeNode::Zoom { xfac, yfac } => builder.zoom(xfac, yfac)?,
                ResizeNode::Affine {
                    matrix,
                    tx,
                    ty,
                    output_width,
                    output_height,
                    kernel,
                } => builder.affine_backward_with_extend_and_hint(
                    matrix,
                    tx,
                    ty,
                    output_width,
                    output_height,
                    kernel,
                    DemandHint::SmallTile,
                    crate::domain::ops::resample::affine::ExtendMode::Copy,
                )?,
            };
        }

        Ok(builder)
    }

    /// `premultiply` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::premultiply;
    /// ```
    pub fn premultiply(self) -> Result<PipelineBuilder<Identity>, BuildError> {
        let bands = self.bands;
        let max_alpha = self.current_interpretation.map_or_else(
            || {
                self.current_colorspace
                    .map_or(255.0, ColorspaceId::max_alpha)
            },
            Interpretation::max_alpha,
        );
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(OperationBridge::new_pixel_local(
                Premultiply::<U8>::new_with_max_alpha(bands, max_alpha),
                bands,
            )),
            BandFormatId::U16 => Box::new(OperationBridge::new_pixel_local(
                Premultiply::<U16>::new_with_max_alpha(bands, max_alpha),
                bands,
            )),
            BandFormatId::I16 => Box::new(OperationBridge::new_pixel_local(
                Premultiply::<I16>::new_with_max_alpha(bands, max_alpha),
                bands,
            )),
            BandFormatId::U32 => Box::new(OperationBridge::new_pixel_local(
                Premultiply::<U32>::new_with_max_alpha(bands, max_alpha),
                bands,
            )),
            BandFormatId::I32 => Box::new(OperationBridge::new_pixel_local(
                Premultiply::<I32>::new_with_max_alpha(bands, max_alpha),
                bands,
            )),
            BandFormatId::F32 => Box::new(OperationBridge::new_pixel_local(
                Premultiply::<F32>::new_with_max_alpha(bands, max_alpha),
                bands,
            )),
            BandFormatId::F64 => Box::new(OperationBridge::new_pixel_local(
                Premultiply::<F64>::new_with_max_alpha(bands, max_alpha),
                bands,
            )),
        };
        self.then(op)
    }

    /// `unpremultiply` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::unpremultiply;
    /// ```
    pub fn unpremultiply(self) -> Result<PipelineBuilder<Identity>, BuildError> {
        let bands = self.bands;
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => Box::new(OperationBridge::new_pixel_local(
                Unpremultiply::<U8>::new(bands),
                bands,
            )),
            BandFormatId::U16 => Box::new(OperationBridge::new_pixel_local(
                Unpremultiply::<U16>::new_with_max_alpha(bands, 65535.0),
                bands,
            )),
            BandFormatId::F32 => Box::new(OperationBridge::new_pixel_local(
                Unpremultiply::<F32>::new(bands),
                bands,
            )),
            format => {
                return Err(BuildError::UnsupportedFormat {
                    op: "unpremultiply",
                    format,
                });
            }
        };
        self.then(op)
    }

    /// `flatten` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::flatten;
    /// ```
    pub fn flatten(self, background: [f32; 4]) -> Result<PipelineBuilder<Identity>, BuildError> {
        let input_bands = self.bands;
        if !flatten_has_alpha(input_bands) {
            let op: Box<dyn DynOperation> = match self.current_format {
                BandFormatId::U8 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<U8>::default(),
                    input_bands,
                )),
                BandFormatId::U16 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<U16>::default(),
                    input_bands,
                )),
                BandFormatId::I16 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<I16>::default(),
                    input_bands,
                )),
                BandFormatId::U32 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<U32>::default(),
                    input_bands,
                )),
                BandFormatId::I32 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<I32>::default(),
                    input_bands,
                )),
                BandFormatId::F32 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<F32>::default(),
                    input_bands,
                )),
                BandFormatId::F64 => Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<F64>::default(),
                    input_bands,
                )),
            };
            return self.then(op);
        }
        let op: Box<dyn DynOperation> = match self.current_format {
            BandFormatId::U8 => {
                let bg = background[..(input_bands - 1) as usize]
                    .iter()
                    .map(|value| (value * 255.0).round().clamp(0.0, 255.0) as u8)
                    .collect();
                Box::new(FlattenBridge::<U8>::new(input_bands, bg))
            }
            BandFormatId::F32 => {
                let bg = background[..(input_bands - 1) as usize].to_vec();
                Box::new(FlattenBridge::<F32>::new(input_bands, bg))
            }
            format => {
                return Err(BuildError::UnsupportedFormat {
                    op: "flatten",
                    format,
                });
            }
        };
        self.then(op)
    }

    /// `thumbnail` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::pipeline::builder::thumbnail;
    /// ```
    #[allow(clippy::needless_pass_by_value)]
    // REASON: public API stability for the builder-style `thumbnail` entry point.
    pub fn thumbnail(
        mut self,
        thumbnail: Thumbnail,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        thumbnail.validate_input(self.bands)?;
        let (mut input_width, mut input_height) = self.current_dimensions();
        let mut plan = thumbnail.into_pipeline_nodes(input_width, input_height, self.bands);
        if self.last_node.is_none()
            && let (Some(source), Some(factor)) = (
                self.arena.source.as_mut(),
                plan.shrink_factor
                    .and_then(|factor| NonZeroU8::new(factor as u8)),
            )
        {
            let new_dimensions = match source.set_thumbnail_shrink_on_load(factor) {
                Ok(true) => Some((source.width(), source.height())),
                Ok(false) => None,
                Err(err) => {
                    return Err(BuildError::SourceHint {
                        context: "thumbnail",
                        message: err.to_string(),
                    });
                }
            };
            if let Some((width, height)) = new_dimensions {
                self.arena.set_dimensions(width, height);
            }
        }
        if self.last_node.is_none() && plan.shrink_factor.is_some() {
            (input_width, input_height) = self.current_dimensions();
            plan = thumbnail.into_pipeline_nodes_without_shrink_hint(
                input_width,
                input_height,
                self.bands,
            );
        }

        let thumbnail_is_large = input_width > 512 || input_height > 512;
        // Force ThinStrip for large-input thumbnails regardless of whether the source
        // applied shrink-on-load. A preloaded (in-memory) source can't shrink on load,
        // but the pipeline is still processing a large image and needs ThinStrip to avoid
        // per-thread source buffers sized to factor×FatStrip-tile-height×source-width
        // (e.g., ShrinkV factor=19 with FatStrip tile_h=256 → 127 MB per thread).
        if self.last_node.is_none() && thumbnail_is_large {
            self.arena
                .set_demand_hint_override(Some(DemandHint::ThinStrip));
        }
        let has_nodes = !plan.nodes.is_empty();

        let mut builder = self.flush_into_identity()?;
        for node in plan.nodes {
            builder = match node {
                ThumbnailNode::Premultiply => builder.premultiply()?,
                ThumbnailNode::ShrinkH { factor } => builder.shrink_h(factor)?,
                ThumbnailNode::ShrinkV { factor } => builder.shrink_v(factor)?,
                ThumbnailNode::ReduceH { factor, kernel } => {
                    builder.reduce_h_with_hint(factor, kernel, DemandHint::ThinStrip)?
                }
                ThumbnailNode::ReduceV { factor, kernel } => {
                    builder.reduce_v_with_hint(factor, kernel, DemandHint::ThinStrip)?
                }
                ThumbnailNode::Affine {
                    matrix,
                    tx,
                    ty,
                    output_width,
                    output_height,
                    kernel,
                } => {
                    let demand_hint = if thumbnail_is_large {
                        DemandHint::ThinStrip
                    } else {
                        DemandHint::SmallTile
                    };
                    builder.affine_backward_with_extend_and_hint(
                        matrix,
                        tx,
                        ty,
                        output_width,
                        output_height,
                        kernel,
                        demand_hint,
                        crate::domain::ops::resample::affine::ExtendMode::Copy,
                    )?
                }
                ThumbnailNode::ExtractArea {
                    x,
                    y,
                    width,
                    height,
                } => builder.extract_area(x, y, width, height)?,
                ThumbnailNode::Unpremultiply => builder.unpremultiply()?,
                ThumbnailNode::Flatten { background } => builder.flatten(background)?,
            };
        }

        if has_nodes {
            Ok(builder)
        } else {
            builder.affine(
                [1.0, 0.0, 0.0, 1.0],
                0.0,
                0.0,
                plan.output_width,
                plan.output_height,
                InterpolationKernel::Nearest,
            )
        }
    }
}
