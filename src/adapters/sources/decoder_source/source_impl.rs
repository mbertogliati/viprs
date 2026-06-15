use super::{
    Arc, BandFormat, DecoderBacking, DecoderSource, DemandHint, Image, ImageDecoder, ImageMetadata,
    ImageSource, LoadOptions, NonZeroU8, Region, ShrinkSample, ThumbnailPreShrinkMode, ViprsError,
    checked_region_end, expected_output_len, materialize_residual_thumbnail_shrink,
    normalize_shrink_factor, normalize_streaming_options, shrunk_dimension,
    software_box_shrink_generic, streaming_backing_shrink_factor, streaming_eager_decode,
    thumbnail_pre_shrink_mode,
};

impl<D: ImageDecoder, F: BandFormat, M> DecoderSource<'_, D, F, M> {
    const fn view_shrink_factor(&self) -> u8 {
        if self.shrink_factor <= self.backing_shrink_factor {
            1
        } else {
            self.shrink_factor / self.backing_shrink_factor
        }
    }

    fn effective_dimension(&self, dimension: u32) -> u32 {
        shrunk_dimension(dimension, self.view_shrink_factor())
    }

    /// Return the [`LoadOptions`] used to produce the pixel buffer.
    pub const fn load_options(&self) -> &LoadOptions {
        &self.opts
    }

    /// Return the normalized shrink factor requested at construction time.
    pub const fn shrink_factor(&self) -> u8 {
        self.shrink_factor
    }

    /// Return the format name reported by the underlying decoder.
    pub fn format_name(&self) -> &'static str {
        self.decoder.format_name()
    }

    /// Return a reference to the decoded image, if available.
    ///
    /// Returns `None` for streaming sources. In eager mode this is the resident
    /// decoded backing image: either the decoder-native shrunken raster or the
    /// full-resolution image used for post-decode shrink fallback.
    pub fn image(&self) -> Option<&Image<F>> {
        match &self.backing {
            DecoderBacking::Eager(image) => Some(image),
            DecoderBacking::Deferred { image, .. } => image
                .get()
                .and_then(|image| image.as_ref().ok().map(Arc::as_ref)),
            DecoderBacking::Streaming { .. } => None,
        }
    }

    /// Returns true when this source decodes each requested tile directly from
    /// compressed input rather than retaining a decoded frame.
    pub const fn is_streaming(&self) -> bool {
        matches!(self.backing, DecoderBacking::Streaming { .. })
    }

    /// Estimated resident decoded bytes kept by the source between tile reads.
    ///
    /// Streaming sources return zero because decoded pixels live only in the
    /// scheduler-owned output buffer passed to `read_region`.
    pub fn resident_decoded_bytes(&self) -> usize {
        match &self.backing {
            DecoderBacking::Eager(image) => image
                .pixels()
                .len()
                .saturating_mul(std::mem::size_of::<F::Sample>()),
            DecoderBacking::Deferred { image, .. } => image.get().map_or(0, |image| {
                image.as_ref().ok().map_or(0, |image| {
                    image
                        .pixels()
                        .len()
                        .saturating_mul(std::mem::size_of::<F::Sample>())
                })
            }),
            DecoderBacking::Streaming { .. } => 0,
        }
    }

    fn materialize_deferred_image(&self) -> Result<&Image<F>, ViprsError>
    where
        F::Sample: ShrinkSample,
    {
        let stable_input = self
            .stable_input
            .as_ref()
            .ok_or_else(|| ViprsError::Codec("DecoderSource: missing stable input".into()))?;
        match &self.backing {
            DecoderBacking::Deferred { image, .. } => {
                let materialized = image.get_or_init(|| {
                    stable_input
                        .decode_with_options::<D, F>(&self.decoder, &self.opts)
                        .map(Arc::new)
                        .map_err(|err| err.to_string())
                });
                match materialized {
                    Ok(image) => Ok(image.as_ref()),
                    Err(err) => Err(ViprsError::Codec(err.clone())),
                }
            }
            _ => Err(ViprsError::Codec(
                "DecoderSource: deferred materialization requested for non-deferred backing".into(),
            )),
        }
    }

    fn eager_decode_with_shrink(&self, factor: NonZeroU8) -> Result<(Image<F>, u8), ViprsError>
    where
        F::Sample: ShrinkSample,
    {
        let effective_factor = normalize_shrink_factor(factor.get());
        let stable_input = self
            .stable_input
            .clone()
            .ok_or_else(|| ViprsError::Codec("DecoderSource: missing stable input".into()))?;
        let mut opts = self.opts.clone();
        opts.shrink_factor = Some(factor);
        let image = stable_input.decode_with_options::<D, F>(&self.decoder, &opts)?;
        let backing_shrink_factor =
            stable_input.backing_shrink_factor(&self.decoder, effective_factor, &image);
        materialize_residual_thumbnail_shrink(image, effective_factor, backing_shrink_factor)
    }
}

// ── ImageSource impl ───────────────────────────────────────────────────────────

impl<D: ImageDecoder + Send + Sync, F: BandFormat, M: Send + Sync> ImageSource
    for DecoderSource<'_, D, F, M>
where
    F::Sample: ShrinkSample,
{
    type Format = F;

    fn width(&self) -> u32 {
        match &self.backing {
            DecoderBacking::Eager(image) => self.effective_dimension(image.width()),
            DecoderBacking::Deferred { width, .. } => self.effective_dimension(*width),
            DecoderBacking::Streaming { width, .. } => *width,
        }
    }

    fn height(&self) -> u32 {
        match &self.backing {
            DecoderBacking::Eager(image) => self.effective_dimension(image.height()),
            DecoderBacking::Deferred { height, .. } => self.effective_dimension(*height),
            DecoderBacking::Streaming { height, .. } => *height,
        }
    }

    fn bands(&self) -> u32 {
        match &self.backing {
            DecoderBacking::Eager(image) => image.bands(),
            DecoderBacking::Deferred { bands, .. } | DecoderBacking::Streaming { bands, .. } => {
                *bands
            }
        }
    }

    /// Decoder-backed sources prefer small tiles: eager mode copies from a
    /// resident raster and streaming mode decodes exactly the requested tile.
    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn metadata(&self) -> ImageMetadata {
        match &self.backing {
            DecoderBacking::Eager(image) => image.metadata().clone(),
            DecoderBacking::Deferred {
                metadata, image, ..
            } => image
                .get()
                .and_then(|image| image.as_ref().ok().map(Arc::as_ref))
                .map_or_else(|| metadata.clone(), |image| image.metadata().clone()),
            DecoderBacking::Streaming { metadata, .. } => metadata.clone(),
        }
    }

    fn set_shrink_on_load(&mut self, factor: NonZeroU8) -> Result<bool, ViprsError> {
        if let DecoderBacking::Streaming {
            input,
            width,
            height,
            bands,
            metadata,
            probe_input,
            ..
        } = &mut self.backing
        {
            let effective_factor = normalize_shrink_factor(factor.get());
            if self.shrink_factor == effective_factor {
                return Ok(self.shrink_factor > 1);
            }

            let mut opts = self.opts.clone();
            opts.shrink_factor = Some(factor);
            let decode_opts = normalize_streaming_options(&opts, effective_factor);
            let info = probe_input(&self.decoder, input, &decode_opts)?;
            let native_applied =
                effective_factor == 1 || info.width != *width || info.height != *height;
            if !native_applied {
                return Ok(false);
            }

            *width = info.width;
            *height = info.height;
            *bands = info.bands;
            *metadata = info.metadata;
            self.opts = opts;
            self.shrink_factor = effective_factor;
            self.backing_shrink_factor = effective_factor;
            return Ok(true);
        }
        if matches!(self.backing, DecoderBacking::Deferred { .. }) {
            let effective_factor = normalize_shrink_factor(factor.get());
            if self.shrink_factor == effective_factor {
                return Ok(effective_factor <= self.backing_shrink_factor);
            }
            if effective_factor < self.backing_shrink_factor {
                return Ok(false);
            }

            let (image, backing_shrink_factor) = self.eager_decode_with_shrink(factor)?;
            self.backing = DecoderBacking::Eager(image);
            self.opts.shrink_factor = Some(factor);
            self.shrink_factor = effective_factor;
            self.backing_shrink_factor = backing_shrink_factor;
            return Ok(effective_factor <= self.backing_shrink_factor);
        }

        let effective_factor = normalize_shrink_factor(factor.get());
        let decode_time_shrink_matches_request = effective_factor <= self.backing_shrink_factor;
        if self.shrink_factor == effective_factor {
            return Ok(decode_time_shrink_matches_request);
        }
        if effective_factor < self.backing_shrink_factor {
            return Ok(false);
        }

        self.opts.shrink_factor = Some(factor);
        self.shrink_factor = effective_factor;
        Ok(decode_time_shrink_matches_request)
    }

    fn set_thumbnail_shrink_on_load(&mut self, factor: NonZeroU8) -> Result<bool, ViprsError> {
        let mode = thumbnail_pre_shrink_mode(self.decoder.format_name());
        if matches!(mode, ThumbnailPreShrinkMode::Unsupported) {
            return Ok(false);
        }

        // Software box shrink for formats that have no native shrink-on-load (e.g. PNG).
        //
        // - Eager backing: shrink the resident buffer in-place and return immediately.
        // - Deferred backing: fall through to the standard Deferred handling below,
        //   which calls eager_decode_with_shrink. PngCodec::decode_path_with_options
        //   applies the shrink inline during decode so the 200MB intermediate is
        //   never materialised.
        // - Streaming: fall through (will return false via the fallback at the end).
        if matches!(mode, ThumbnailPreShrinkMode::SoftwareBoxShrink)
            && let DecoderBacking::Eager(ref img) = self.backing
        {
            let effective_factor = normalize_shrink_factor(factor.get());
            if effective_factor <= 1 {
                return Ok(false);
            }
            if self.shrink_factor == effective_factor {
                return Ok(true);
            }
            let shrunken = software_box_shrink_generic(img, usize::from(effective_factor))?;
            self.backing = DecoderBacking::Eager(shrunken);
            self.shrink_factor = effective_factor;
            self.backing_shrink_factor = effective_factor;
            self.opts.shrink_factor = Some(factor);
            return Ok(true);
        }
        // For Deferred: fall through to the Deferred code below.
        // For Streaming: fall through to the false return at the bottom.

        if matches!(self.backing, DecoderBacking::Streaming { .. }) {
            let effective_factor = normalize_shrink_factor(factor.get());
            if self.shrink_factor == effective_factor {
                return Ok(true);
            }

            // Eagerly decode the full image with the shrink factor applied,
            // converting from streaming to eager backing. This avoids costly
            // per-tile re-decode: after shrink-on-load the decoded raster is
            // small (e.g. 2048→~400px) so holding it in RAM is cheap while
            // eliminating repeated full-file parses for each tile request.
            let mut opts = self.opts.clone();
            opts.shrink_factor = Some(factor);
            let image = streaming_eager_decode::<D, F>(&self.backing, &self.decoder, &opts)?;
            let backing_shrink_factor = streaming_backing_shrink_factor::<D, F>(
                &self.backing,
                &self.decoder,
                effective_factor,
                &image,
            );
            self.backing = DecoderBacking::Eager(image);
            self.opts = opts;
            self.shrink_factor = effective_factor;
            self.backing_shrink_factor = backing_shrink_factor;
            return Ok(true);
        }
        if matches!(self.backing, DecoderBacking::Deferred { .. }) {
            let effective_factor = normalize_shrink_factor(factor.get());
            if self.shrink_factor == effective_factor {
                return Ok(true);
            }
            if effective_factor < self.backing_shrink_factor {
                return Ok(false);
            }

            let (image, backing_shrink_factor) = self.eager_decode_with_shrink(factor)?;
            self.backing = DecoderBacking::Eager(image);
            self.opts.shrink_factor = Some(factor);
            self.shrink_factor = effective_factor;
            self.backing_shrink_factor = backing_shrink_factor;
            return Ok(backing_shrink_factor >= effective_factor);
        }

        let effective_factor = normalize_shrink_factor(factor.get());
        if self.shrink_factor == effective_factor {
            return Ok(true);
        }
        if effective_factor < self.backing_shrink_factor {
            return Ok(false);
        }

        let Some(stable_input) = self.stable_input.as_ref() else {
            return Ok(false);
        };

        let mut opts = self.opts.clone();
        opts.shrink_factor = Some(factor);
        let image = stable_input.decode_with_options::<D, F>(&self.decoder, &opts)?;
        let backing_shrink_factor =
            stable_input.backing_shrink_factor(&self.decoder, effective_factor, &image);
        let (image, backing_shrink_factor) =
            materialize_residual_thumbnail_shrink(image, effective_factor, backing_shrink_factor)?;

        self.backing = DecoderBacking::Eager(image);
        self.opts = opts;
        self.shrink_factor = effective_factor;
        self.backing_shrink_factor = backing_shrink_factor;

        Ok(true)
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        match &self.backing {
            DecoderBacking::Eager(image) => self.read_eager_region(image, region, output),
            DecoderBacking::Deferred { .. } => {
                let image = self.materialize_deferred_image()?;
                self.read_eager_region(image, region, output)
            }
            DecoderBacking::Streaming {
                input,
                decode_region,
                ..
            } => {
                let expected =
                    expected_output_len::<F>(region, self.bands(), "DecoderSource streaming")?;
                if output.len() != expected {
                    return Err(ViprsError::Codec(format!(
                        "DecoderSource streaming: output buffer size mismatch (got {}, expected {expected})",
                        output.len()
                    )));
                }

                let decode_opts = normalize_streaming_options(&self.opts, self.shrink_factor);
                decode_region(&self.decoder, input, &decode_opts, region, output)
            }
        }
    }

    fn borrow_region(&self, region: Region) -> Option<&[u8]> {
        // Zero-copy fast path: return a direct slice of the backing pixel buffer when
        // the region is a full-width contiguous strip with no view-shrink applied.
        // This eliminates the memmove that read_eager_region would otherwise perform.
        let image = match &self.backing {
            DecoderBacking::Eager(image) => image,
            DecoderBacking::Deferred { image, .. } => image.get()?.as_ref().ok()?,
            DecoderBacking::Streaming { .. } => return None,
        };

        // borrow_region is only valid when view_shrink_factor == 1 (no post-decode shrink)
        // and the region spans the full image width (pixels are contiguous in the buffer).
        if self.view_shrink_factor() != 1 {
            return None;
        }
        if region.x != 0 || region.width != image.width() {
            return None;
        }
        // Clamp to image bounds
        let Ok((_, end_y)) = checked_region_end(
            region,
            image.width(),
            image.height(),
            "DecoderSource borrow",
        ) else {
            return None;
        };
        if region.y < 0 || end_y > i64::from(image.height()) {
            return None;
        }

        let bands = image.bands() as usize;
        let row_samples = image.width() as usize * bands;
        let sample_size = std::mem::size_of::<F::Sample>();
        let row_bytes = row_samples * sample_size;
        let start_byte = region.y as usize * row_bytes;
        let end_byte = start_byte + region.height as usize * row_bytes;
        let pixel_bytes = bytemuck::cast_slice::<F::Sample, u8>(image.pixels());
        Some(&pixel_bytes[start_byte..end_byte])
    }
}

impl<D: ImageDecoder, F: BandFormat, M> DecoderSource<'_, D, F, M> {
    fn read_eager_region(
        &self,
        image: &Image<F>,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError>
    where
        F::Sample: ShrinkSample,
    {
        let expected = expected_output_len::<F>(region, image.bands(), "DecoderSource eager")?;
        if output.len() != expected {
            return Err(ViprsError::Codec(format!(
                "DecoderSource eager: output buffer size mismatch (got {}, expected {expected})",
                output.len()
            )));
        }

        let bands = image.bands() as usize;
        let pixels = image.pixels();
        let output_samples: &mut [F::Sample] = bytemuck::try_cast_slice_mut(output)
            .map_err(|_| ViprsError::Codec("DecoderSource: output buffer size mismatch".into()))?;
        let view_width = self.effective_dimension(image.width());
        let view_height = self.effective_dimension(image.height());
        let (end_x, end_y) =
            checked_region_end(region, view_width, view_height, "DecoderSource eager")?;
        let factor = usize::from(self.view_shrink_factor().max(1));

        if view_width == 0 || view_height == 0 || bands == 0 {
            return Ok(());
        }

        // Fast path: no view-shrink, region is within image bounds.
        // Copy entire rows at once instead of pixel-by-pixel to minimise
        // call overhead on `copy_from_slice` and enable the compiler to use
        // SIMD or wide loads for the bulk transfer.
        if factor == 1
            && region.x >= 0
            && region.y >= 0
            && end_x <= i64::from(view_width)
            && end_y <= i64::from(view_height)
        {
            let img_w = image.width() as usize;
            let row_samples = region.width as usize * bands;
            for row in 0..region.height as usize {
                let src_row_start =
                    (region.y as usize + row) * img_w * bands + region.x as usize * bands;
                let dst_row_start = row * row_samples;
                output_samples[dst_row_start..dst_row_start + row_samples]
                    .copy_from_slice(&pixels[src_row_start..src_row_start + row_samples]);
            }
            return Ok(());
        }

        let view_width_i64 = i64::from(view_width);
        let view_height_i64 = i64::from(view_height);
        for row in 0..region.height as usize {
            for col in 0..region.width as usize {
                let dst_pixel_idx = row * region.width as usize + col;
                let dst_sample_start = dst_pixel_idx * bands;

                if factor == 1 {
                    let src_x =
                        (i64::from(region.x) + col as i64).clamp(0, view_width_i64 - 1) as usize;
                    let src_y =
                        (i64::from(region.y) + row as i64).clamp(0, view_height_i64 - 1) as usize;
                    let src_pixel_idx = src_y * image.width() as usize + src_x;
                    let src_sample_start = src_pixel_idx * bands;
                    output_samples[dst_sample_start..dst_sample_start + bands]
                        .copy_from_slice(&pixels[src_sample_start..src_sample_start + bands]);
                    continue;
                }

                let view_x =
                    (i64::from(region.x) + col as i64).clamp(0, view_width_i64 - 1) as usize;
                let view_y =
                    (i64::from(region.y) + row as i64).clamp(0, view_height_i64 - 1) as usize;
                let src_x0 = view_x * factor;
                let src_y0 = view_y * factor;
                let src_x1 = ((view_x + 1) * factor).min(image.width() as usize);
                let src_y1 = ((view_y + 1) * factor).min(image.height() as usize);
                let sample_count = (src_x1 - src_x0) * (src_y1 - src_y0);

                for band in 0..bands {
                    let mut sum = 0.0_f64;
                    for src_y in src_y0..src_y1 {
                        let row_base = src_y * image.width() as usize * bands;
                        for src_x in src_x0..src_x1 {
                            let src_idx = row_base + src_x * bands + band;
                            sum += pixels[src_idx].to_f64();
                        }
                    }
                    output_samples[dst_sample_start + band] =
                        F::Sample::from_f64_clamped(sum / sample_count as f64);
                }
            }
        }
        Ok(())
    }
}
