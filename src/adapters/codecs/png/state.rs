use std::fs::File;
use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::sync::{
    LazyLock,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
#[cfg(test)]
use std::time::Duration;

use png::{BitDepth, Filter};

use crate::domain::error::ViprsError;

use super::decode_full::png_file_reader;
use super::region_decode::decode_png_full_raster_with_png_crate;

/// Configurable PNG encoder mirroring libvips' compression/interlace controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// The `PngEncoder` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::codecs::png::PngEncoder>();
/// ```
pub struct PngEncoder {
    /// DEFLATE compression level: 0 disables compression, 1 favours throughput,
    /// and 9 maximises effort.
    pub compression: u8,
    /// Enable Adam7 interlacing.
    pub interlace: bool,
    /// Row filter selection strategy.
    pub filter: Filter,
}

impl Default for PngEncoder {
    fn default() -> Self {
        Self {
            compression: 1,
            interlace: false,
            filter: Filter::NoFilter,
        }
    }
}

/// PNG codec: implements both [`ImageDecoder`] and [`ImageEncoder`].
///
/// Only U8 and U16 band formats are supported. All other formats return
/// [`ViprsError::Codec`].
pub struct PngCodec {
    row_scratch: Mutex<Vec<u8>>,
    sequential_path_session: Mutex<Option<PngSequentialPathSession>>,
    interlaced_region_cache: Mutex<Option<PngInterlacedCacheEntry>>,
}

impl Default for PngCodec {
    fn default() -> Self {
        Self {
            row_scratch: Mutex::new(Vec::new()),
            sequential_path_session: Mutex::new(None),
            interlaced_region_cache: Mutex::new(None),
        }
    }
}

impl PngCodec {
    #[cfg(test)]
    pub(super) fn probe_id(&self) -> usize {
        self as *const Self as usize
    }

    pub(super) fn take_row_scratch(&self, row_len: usize) -> Result<Vec<u8>, ViprsError> {
        let mut shared = self
            .row_scratch
            .lock()
            .map_err(|_| ViprsError::Codec("png: row scratch mutex poisoned".into()))?;
        let mut scratch = std::mem::take(&mut *shared);
        drop(shared);
        if scratch.len() != row_len {
            scratch.resize(row_len, 0);
        }
        Ok(scratch)
    }

    pub(super) fn store_row_scratch(&self, scratch: Vec<u8>) -> Result<(), ViprsError> {
        let mut shared = self
            .row_scratch
            .lock()
            .map_err(|_| ViprsError::Codec("png: row scratch mutex poisoned".into()))?;
        if shared.capacity() < scratch.capacity() {
            *shared = scratch;
        }
        Ok(())
    }

    pub(super) fn interlaced_raster_from_bytes(
        &self,
        src: &[u8],
    ) -> Result<Arc<PngInterlacedRaster>, ViprsError> {
        let key = PngInterlacedCacheKey::Bytes {
            addr: src.as_ptr() as usize,
            len: src.len(),
        };
        self.interlaced_raster_from_key(key, || {
            decode_png_full_raster_with_png_crate(Cursor::new(src))
        })
    }

    pub(super) fn interlaced_raster_from_path(
        &self,
        path: &Path,
    ) -> Result<Arc<PngInterlacedRaster>, ViprsError> {
        self.interlaced_raster_from_key(PngInterlacedCacheKey::Path(path.to_path_buf()), || {
            decode_png_full_raster_with_png_crate(png_file_reader(path)?)
        })
    }

    fn interlaced_raster_from_key(
        &self,
        key: PngInterlacedCacheKey,
        decode: impl FnOnce() -> Result<PngInterlacedRaster, ViprsError>,
    ) -> Result<Arc<PngInterlacedRaster>, ViprsError> {
        if let Some(raster) = self.cached_interlaced_raster(&key)? {
            return Ok(raster);
        }

        #[cfg(test)]
        PNG_ROW_DECODE_PROBE.record_full_raster_decode_for(self.probe_id());
        let raster = Arc::new(decode()?);
        let mut cache = self
            .interlaced_region_cache
            .lock()
            .map_err(|_| ViprsError::Codec("png: interlaced region cache mutex poisoned".into()))?;
        *cache = Some(PngInterlacedCacheEntry {
            key,
            raster: Arc::clone(&raster),
        });
        Ok(raster)
    }

    fn cached_interlaced_raster(
        &self,
        key: &PngInterlacedCacheKey,
    ) -> Result<Option<Arc<PngInterlacedRaster>>, ViprsError> {
        let cache = self
            .interlaced_region_cache
            .lock()
            .map_err(|_| ViprsError::Codec("png: interlaced region cache mutex poisoned".into()))?;
        Ok(cache.as_ref().and_then(|entry| {
            if entry.key == *key {
                Some(Arc::clone(&entry.raster))
            } else {
                None
            }
        }))
    }

    pub(super) fn take_sequential_path_session(
        &self,
    ) -> Result<Option<PngSequentialPathSession>, ViprsError> {
        let mut shared = self
            .sequential_path_session
            .lock()
            .map_err(|_| ViprsError::Codec("png: sequential path session mutex poisoned".into()))?;
        #[cfg(test)]
        let _session_mutex_probe =
            PNG_ROW_DECODE_PROBE.hold_sequential_session_mutex(self.probe_id());
        let session = std::mem::take(&mut *shared);
        #[cfg(test)]
        drop(_session_mutex_probe);
        drop(shared);
        Ok(session)
    }

    pub(super) fn store_sequential_path_session(
        &self,
        session: Option<PngSequentialPathSession>,
    ) -> Result<(), ViprsError> {
        let mut shared = self
            .sequential_path_session
            .lock()
            .map_err(|_| ViprsError::Codec("png: sequential path session mutex poisoned".into()))?;
        #[cfg(test)]
        let _session_mutex_probe =
            PNG_ROW_DECODE_PROBE.hold_sequential_session_mutex(self.probe_id());
        *shared = session;
        #[cfg(test)]
        drop(_session_mutex_probe);
        drop(shared);
        Ok(())
    }
}

#[cfg(test)]
#[derive(Default)]
pub(super) struct PngRowDecodeProbe {
    enabled: AtomicBool,
    target_codec: AtomicUsize,
    active: AtomicUsize,
    max_active: AtomicUsize,
    full_raster_decodes: AtomicUsize,
    total_rows: AtomicUsize,
    sequential_session_mutex_holds: AtomicUsize,
    rows_while_sequential_session_mutex_held: AtomicUsize,
    row_delay_ms: AtomicU64,
}

#[cfg(test)]
pub(super) static PNG_ROW_DECODE_PROBE: LazyLock<PngRowDecodeProbe> =
    LazyLock::new(PngRowDecodeProbe::default);

#[cfg(test)]
impl PngRowDecodeProbe {
    pub(super) fn enable(&self, target_codec: usize, row_delay: Duration) {
        self.active.store(0, Ordering::SeqCst);
        self.max_active.store(0, Ordering::SeqCst);
        self.full_raster_decodes.store(0, Ordering::SeqCst);
        self.total_rows.store(0, Ordering::SeqCst);
        self.sequential_session_mutex_holds
            .store(0, Ordering::SeqCst);
        self.rows_while_sequential_session_mutex_held
            .store(0, Ordering::SeqCst);
        self.target_codec.store(target_codec, Ordering::SeqCst);
        self.row_delay_ms
            .store(row_delay.as_millis() as u64, Ordering::SeqCst);
        self.enabled.store(true, Ordering::SeqCst);
    }

    pub(super) fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
        self.active.store(0, Ordering::SeqCst);
        self.max_active.store(0, Ordering::SeqCst);
        self.full_raster_decodes.store(0, Ordering::SeqCst);
        self.total_rows.store(0, Ordering::SeqCst);
        self.sequential_session_mutex_holds
            .store(0, Ordering::SeqCst);
        self.rows_while_sequential_session_mutex_held
            .store(0, Ordering::SeqCst);
        self.target_codec.store(0, Ordering::SeqCst);
        self.row_delay_ms.store(0, Ordering::SeqCst);
    }

    pub(super) fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }

    pub(super) fn total_rows(&self) -> usize {
        self.total_rows.load(Ordering::SeqCst)
    }

    pub(super) fn full_raster_decodes(&self) -> usize {
        self.full_raster_decodes.load(Ordering::SeqCst)
    }

    pub(super) fn rows_while_sequential_session_mutex_held(&self) -> usize {
        self.rows_while_sequential_session_mutex_held
            .load(Ordering::SeqCst)
    }

    fn matches_target(&self, codec_id: usize) -> bool {
        self.enabled.load(Ordering::SeqCst) && self.target_codec.load(Ordering::SeqCst) == codec_id
    }

    pub(super) fn record_full_raster_decode_for(&self, codec_id: usize) {
        if self.matches_target(codec_id) {
            self.full_raster_decodes.fetch_add(1, Ordering::SeqCst);
        }
    }

    pub(super) fn enter_for(&self, codec_id: usize) -> Option<PngRowDecodeProbeGuard<'_>> {
        if !self.matches_target(codec_id) {
            return None;
        }
        if self.sequential_session_mutex_holds.load(Ordering::SeqCst) > 0 {
            self.rows_while_sequential_session_mutex_held
                .fetch_add(1, Ordering::SeqCst);
        }
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.total_rows.fetch_add(1, Ordering::SeqCst);
        let mut observed = self.max_active.load(Ordering::SeqCst);
        while active > observed {
            match self.max_active.compare_exchange(
                observed,
                active,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(current) => observed = current,
            }
        }
        let row_delay_ms = self.row_delay_ms.load(Ordering::SeqCst);
        if row_delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(row_delay_ms));
        }
        Some(PngRowDecodeProbeGuard { probe: self })
    }

    pub(super) fn hold_sequential_session_mutex(
        &self,
        codec_id: usize,
    ) -> Option<PngSequentialSessionMutexProbeGuard<'_>> {
        if !self.matches_target(codec_id) {
            return None;
        }
        self.sequential_session_mutex_holds
            .fetch_add(1, Ordering::SeqCst);
        Some(PngSequentialSessionMutexProbeGuard { probe: self })
    }
}

#[cfg(test)]
pub(super) struct PngRowDecodeProbeGuard<'a> {
    probe: &'a PngRowDecodeProbe,
}

#[cfg(test)]
impl Drop for PngRowDecodeProbeGuard<'_> {
    fn drop(&mut self) {
        self.probe.active.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
pub(super) struct PngSequentialSessionMutexProbeGuard<'a> {
    probe: &'a PngRowDecodeProbe,
}

#[cfg(test)]
impl Drop for PngSequentialSessionMutexProbeGuard<'_> {
    fn drop(&mut self) {
        self.probe
            .sequential_session_mutex_holds
            .fetch_sub(1, Ordering::SeqCst);
    }
}

pub(super) struct PngSequentialPathSession {
    pub(super) path: PathBuf,
    pub(super) reader: png::Reader<BufReader<File>>,
    pub(super) row_scratch: Vec<u8>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) bands: u32,
    pub(super) bit_depth: BitDepth,
    pub(super) next_source_y: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PngInterlacedCacheKey {
    Path(PathBuf),
    Bytes { addr: usize, len: usize },
}

pub(super) struct PngInterlacedCacheEntry {
    key: PngInterlacedCacheKey,
    raster: Arc<PngInterlacedRaster>,
}

pub(super) struct PngInterlacedRaster {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) bands: u32,
    pub(super) bit_depth: BitDepth,
    pub(super) pixels: Vec<u8>,
}

pub(super) const PNG_XMP_KEYWORD: &str = "XML:com.adobe.xmp";
pub(super) const PNG_FILE_READER_CAPACITY: usize = 256 * 1024;
