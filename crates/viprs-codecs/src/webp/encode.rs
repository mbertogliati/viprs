use std::ffi::c_void;

use libwebp_sys::{
    WEBP_MUX_ABI_VERSION, WebPData, WebPEncode, WebPFree, WebPMemoryWrite, WebPMemoryWriter,
    WebPMemoryWriterClear, WebPMemoryWriterInit, WebPMuxAssemble, WebPMuxDelete, WebPMuxError,
    WebPMuxSetChunk, WebPMuxSetImage, WebPNewInternal, WebPPicture, WebPPictureFree,
    WebPPictureImportRGB, WebPPictureImportRGBA, WebPValidateConfig,
};
use webp::{PixelLayout, WebPConfig};

use super::super::web_colour::normalize_web_output_u8;
use super::WebpCodec;
use super::common::{
    WEBP_DEFAULT_LOSSLESS, WEBP_DEFAULT_METHOD, WEBP_DEFAULT_QUALITY, WEBP_ICC_CHUNK_FOURCC,
    WEBP_XMP_CHUNK_FOURCC, require_u8,
};
use viprs_core::codec_options::SaveOptions;
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, U8};
use viprs_core::image::InMemoryImage;
use viprs_ports::codec::ImageEncoder;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// WebP-specific encode controls exposed alongside the generic [`SaveOptions`] bridge.
pub struct WebpEncodeOptions {
    /// Lossy or lossless quality, clamped to 0–100.
    pub quality: u8,
    /// Encoder method / effort, clamped to 0–6.
    pub method: u8,
    /// Toggle WebP lossless mode.
    pub lossless: bool,
}

impl WebpEncodeOptions {
    /// Create explicit WebP encode controls.
    #[must_use]
    pub const fn new(quality: u8, method: u8, lossless: bool) -> Self {
        Self {
            quality,
            method,
            lossless,
        }
    }

    #[must_use]
    fn clamped(self) -> Self {
        Self {
            quality: self.quality.min(100),
            method: self.method.min(6),
            lossless: self.lossless,
        }
    }

    #[must_use]
    fn from_save_options(opts: &SaveOptions) -> Self {
        Self::new(
            opts.quality.unwrap_or(WEBP_DEFAULT_QUALITY),
            opts.method.or(opts.effort).unwrap_or(WEBP_DEFAULT_METHOD),
            opts.lossless.unwrap_or(WEBP_DEFAULT_LOSSLESS),
        )
        .clamped()
    }
}

impl Default for WebpEncodeOptions {
    fn default() -> Self {
        Self::new(
            WEBP_DEFAULT_QUALITY,
            WEBP_DEFAULT_METHOD,
            WEBP_DEFAULT_LOSSLESS,
        )
    }
}

impl From<WebpEncodeOptions> for SaveOptions {
    fn from(opts: WebpEncodeOptions) -> Self {
        let opts = opts.clamped();
        Self {
            quality: Some(opts.quality),
            lossless: Some(opts.lossless),
            method: Some(opts.method),
            ..Self::default()
        }
    }
}
fn webp_attach_metadata(
    bitstream: &[u8],
    icc_profile: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> Result<Vec<u8>, ViprsError> {
    let bitstream = WebPData {
        bytes: bitstream.as_ptr(),
        size: bitstream.len(),
    };
    let mux = {
        // SAFETY: `WebPNewInternal` allocates a fresh mux handle using libwebp's ABI version.
        unsafe { WebPNewInternal(WEBP_MUX_ABI_VERSION as i32) }
    };
    if mux.is_null() {
        return Err(ViprsError::Codec("webp: unable to create mux".into()));
    }

    let result = (|| {
        let status = {
            // SAFETY: `mux` is live, `bitstream` points to the encoded still-image bytes
            // for the duration of the call, and `copy_data = 1` lets libwebp own its copy
            // before we return.
            unsafe { WebPMuxSetImage(mux, std::ptr::from_ref(&bitstream), 1) }
        };
        if status != WebPMuxError::WEBP_MUX_OK {
            return Err(ViprsError::Codec(format!(
                "webp: unable to attach image bitstream ({status:?})"
            )));
        }

        if let Some(icc_profile) = icc_profile {
            let chunk = WebPData {
                bytes: icc_profile.as_ptr(),
                size: icc_profile.len(),
            };
            let status = {
                // SAFETY: `mux` is live, the 4CC is null-terminated, `chunk` points to the ICC
                // bytes for the duration of the call, and `copy_data = 1` transfers the chunk
                // contents into mux-owned storage.
                unsafe {
                    WebPMuxSetChunk(
                        mux,
                        WEBP_ICC_CHUNK_FOURCC.as_ptr().cast(),
                        std::ptr::from_ref(&chunk),
                        1,
                    )
                }
            };
            if status != WebPMuxError::WEBP_MUX_OK {
                return Err(ViprsError::Codec(format!(
                    "webp: unable to attach ICC chunk ({status:?})"
                )));
            }
        }

        if let Some(xmp) = xmp {
            let chunk = WebPData {
                bytes: xmp.as_ptr(),
                size: xmp.len(),
            };
            let status = {
                // SAFETY: `mux` is live, the 4CC is null-terminated, `chunk` points to the XMP
                // bytes for the duration of the call, and `copy_data = 1` transfers the chunk
                // contents into mux-owned storage.
                unsafe {
                    WebPMuxSetChunk(
                        mux,
                        WEBP_XMP_CHUNK_FOURCC.as_ptr().cast(),
                        std::ptr::from_ref(&chunk),
                        1,
                    )
                }
            };
            if status != WebPMuxError::WEBP_MUX_OK {
                return Err(ViprsError::Codec(format!(
                    "webp: unable to attach XMP chunk ({status:?})"
                )));
            }
        }

        let mut assembled = std::mem::MaybeUninit::<WebPData>::uninit();
        let status = {
            // SAFETY: `mux` is valid and `assembled` points to writable storage for libwebp.
            unsafe { WebPMuxAssemble(mux, assembled.as_mut_ptr()) }
        };
        if status != WebPMuxError::WEBP_MUX_OK {
            return Err(ViprsError::Codec(format!(
                "webp: unable to assemble mux output ({status:?})"
            )));
        }

        let assembled = {
            // SAFETY: `WebPMuxAssemble` succeeded and initialized `assembled`.
            unsafe { assembled.assume_init() }
        };
        let bytes = {
            // SAFETY: libwebp owns `assembled.bytes` until it is released with
            // `WebPFree`, and `assembled.size` gives the exact assembled payload length.
            unsafe { std::slice::from_raw_parts(assembled.bytes, assembled.size) }.to_vec()
        };
        // SAFETY: `assembled.bytes` was allocated by libwebp and must be freed with `WebPFree`.
        unsafe { WebPFree(assembled.bytes.cast_mut().cast::<c_void>()) };
        Ok(bytes)
    })();

    // SAFETY: `mux` was created by `WebPNewInternal` and must be deleted once.
    unsafe { WebPMuxDelete(mux) };
    result
}
fn encode_webp_advanced(
    pixels: &[u8],
    layout: PixelLayout,
    width: u32,
    height: u32,
    opts: &SaveOptions,
) -> Result<Vec<u8>, ViprsError> {
    let mut config = WebPConfig::new()
        .map_err(|()| ViprsError::Codec("webp: encoder config init failed".into()))?;
    let webp_opts = WebpEncodeOptions::from_save_options(opts);
    let near_lossless = opts.near_lossless.map(|level| level.min(100));
    let lossless = webp_opts.lossless || near_lossless.is_some();

    if lossless {
        config.lossless = 1;
        config.quality = f32::from(webp_opts.quality);
        config.alpha_compression = 0;
    } else {
        config.quality = f32::from(webp_opts.quality);
    }
    config.method = i32::from(webp_opts.method);
    if let Some(level) = near_lossless {
        config.near_lossless = i32::from(level);
    }
    if let Some(enabled) = opts.exact_alpha {
        config.exact = i32::from(enabled);
    }
    if let Some(enabled) = opts.smart_subsample {
        config.use_sharp_yuv = i32::from(enabled);
    }
    let use_argb = lossless || near_lossless.is_some() || opts.smart_subsample.unwrap_or(false);

    // SAFETY: `config` is initialized via libwebp, the imported pixel slice outlives the encode
    // call, and all libwebp-owned allocations are released via `WebPPictureFree` /
    // `WebPMemoryWriterClear` before returning.
    unsafe {
        if WebPValidateConfig(std::ptr::from_ref(&config)) == 0 {
            return Err(ViprsError::Codec(
                "webp: invalid encoder configuration".into(),
            ));
        }

        let mut picture = WebPPicture::new()
            .map_err(|()| ViprsError::Codec("webp: picture init failed".into()))?;
        picture.use_argb = i32::from(use_argb);
        picture.width = i32::try_from(width)
            .map_err(|_| ViprsError::Codec("webp: width exceeds encoder limits".into()))?;
        picture.height = i32::try_from(height)
            .map_err(|_| ViprsError::Codec("webp: height exceeds encoder limits".into()))?;

        let stride = match layout {
            PixelLayout::Rgb => i32::try_from(width.saturating_mul(3))
                .map_err(|_| ViprsError::Codec("webp: RGB stride exceeds encoder limits".into()))?,
            PixelLayout::Rgba => i32::try_from(width.saturating_mul(4)).map_err(|_| {
                ViprsError::Codec("webp: RGBA stride exceeds encoder limits".into())
            })?,
        };
        let imported = match layout {
            PixelLayout::Rgb => {
                WebPPictureImportRGB(std::ptr::from_mut(&mut picture), pixels.as_ptr(), stride)
            }
            PixelLayout::Rgba => {
                WebPPictureImportRGBA(std::ptr::from_mut(&mut picture), pixels.as_ptr(), stride)
            }
        };
        if imported == 0 {
            WebPPictureFree(std::ptr::from_mut(&mut picture));
            return Err(ViprsError::Codec(
                "webp: failed to import pixels into encoder picture".into(),
            ));
        }

        let mut writer = std::mem::MaybeUninit::<WebPMemoryWriter>::uninit();
        WebPMemoryWriterInit(writer.as_mut_ptr());
        let mut writer = writer.assume_init();
        picture.writer = Some(WebPMemoryWrite);
        picture.custom_ptr = std::ptr::from_mut(&mut writer).cast();

        let encoded = if WebPEncode(
            std::ptr::from_ref(&config),
            std::ptr::from_mut(&mut picture),
        ) != 0
        {
            let bytes = std::slice::from_raw_parts(writer.mem, writer.size).to_vec();
            Ok(bytes)
        } else {
            Err(ViprsError::Codec(format!(
                "webp: encode failed with {:?}",
                picture.error_code
            )))
        };

        WebPPictureFree(std::ptr::from_mut(&mut picture));
        WebPMemoryWriterClear(std::ptr::from_mut(&mut writer));
        encoded
    }
}

impl WebpCodec {
    /// Encode with explicit WebP controls while still flowing through [`ImageEncoder`].
    pub fn encode_with_webp_options<F: BandFormat>(
        &self,
        image: &InMemoryImage<F>,
        opts: &WebpEncodeOptions,
    ) -> Result<Vec<u8>, ViprsError> {
        self.encode_with_options(image, &SaveOptions::from(*opts))
    }
}

// ── ImageEncoder ──────────────────────────────────────────────────────────────

impl ImageEncoder for WebpCodec {
    fn format_name(&self) -> &'static str {
        "webp"
    }

    fn encode<F: BandFormat>(&self, image: &InMemoryImage<F>) -> Result<Vec<u8>, ViprsError> {
        self.encode_with_webp_options(image, &WebpEncodeOptions::default())
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &InMemoryImage<F>,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        require_u8::<F>()?;
        let image = normalize_web_output_u8(
            // SAFETY: `require_u8::<F>()` above guarantees `F::Sample == u8`.
            unsafe { &*std::ptr::from_ref(image).cast::<InMemoryImage<U8>>() },
        )?;
        let image = image.as_ref();

        let pixel_bytes: &[u8] = bytemuck::cast_slice(image.pixels());
        let width = image.width();
        let height = image.height();

        // `Encoder` borrows its pixel slice, so each arm must encode before the
        // slice is dropped.  The grayscale arm constructs a temporary RGB Vec
        // and encodes within the same scope.
        let encoded_bytes: Vec<u8> = match image.bands() {
            1 => {
                // WebP has no native grayscale format. Expand to RGB by
                // replicating the single channel into R, G, and B.
                let rgb: Vec<u8> = pixel_bytes.iter().flat_map(|&g| [g, g, g]).collect();
                encode_webp_advanced(&rgb, PixelLayout::Rgb, width, height, opts)?
            }
            3 => encode_webp_advanced(pixel_bytes, PixelLayout::Rgb, width, height, opts)?,
            4 => encode_webp_advanced(pixel_bytes, PixelLayout::Rgba, width, height, opts)?,
            n => {
                return Err(ViprsError::Codec(format!(
                    "webp: unsupported band count {n}; expected 1, 3, or 4"
                )));
            }
        };

        if opts.strip_metadata == Some(true) {
            Ok(encoded_bytes)
        } else {
            webp_attach_metadata(
                &encoded_bytes,
                image.metadata().icc_profile.as_deref(),
                image.metadata().xmp.as_deref(),
            )
        }
    }
}
