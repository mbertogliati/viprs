#![allow(clippy::significant_drop_tightening)]
// REASON: limit snapshots intentionally outlive intermediate validation helpers.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::domain::error::ViprsError;

/// Decode-time safety limits for untrusted image input.
///
/// Servers should configure these per request or globally so oversized images are rejected before
/// decode buffers are allocated.
///
/// # Examples
/// ```rust
/// # use viprs::domain::limits::DecodeLimits;
/// let limits = DecodeLimits::default_safe();
/// assert!(limits.validate_u8_image(64, 64, 3).is_ok());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Maximum allowed width in pixels. Default: 32768.
    pub max_width: u32,
    /// Maximum allowed height in pixels. Default: 32768.
    pub max_height: u32,
    /// Maximum total pixel count (width × height). Default: `256_000_000`.
    pub max_pixels: u64,
    /// Maximum decoded buffer size in bytes. Default: 4GB.
    pub max_decode_bytes: u64,
}

impl DecodeLimits {
    /// Safe defaults suitable for most server deployments.
    #[must_use]
    pub const fn default_safe() -> Self {
        Self {
            max_width: 32_768,
            max_height: 32_768,
            max_pixels: 256_000_000,
            max_decode_bytes: 4_294_967_296,
        }
    }

    /// No limits — equivalent to not checking. Use only for trusted input.
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            max_width: u32::MAX,
            max_height: u32::MAX,
            max_pixels: u64::MAX,
            max_decode_bytes: u64::MAX,
        }
    }

    /// Validate dimensions against these limits.
    ///
    /// Returns `Ok(())` when the decoded image fits within all limits, or a
    /// [`ViprsError::ImageTooLarge`] describing the rejected allocation.
    pub fn validate(
        &self,
        width: u32,
        height: u32,
        bands: u32,
        bytes_per_sample: u32,
    ) -> Result<(), ViprsError> {
        let total_bytes = decoded_bytes(width, height, bands, bytes_per_sample);

        if width > self.max_width || height > self.max_height {
            return Err(ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes: total_bytes,
                limit_bytes: decoded_bytes(
                    width.min(self.max_width),
                    height.min(self.max_height),
                    bands,
                    bytes_per_sample,
                ),
                details: "image dimensions exceed decode limits",
            });
        }

        let pixels = u64::from(width) * u64::from(height);
        if pixels > self.max_pixels {
            return Err(ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes: total_bytes,
                limit_bytes: u128::from(self.max_pixels)
                    * u128::from(bands)
                    * u128::from(bytes_per_sample),
                details: "pixel count exceeds decode limits",
            });
        }

        if total_bytes > u128::from(self.max_decode_bytes) {
            return Err(ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes: total_bytes,
                limit_bytes: u128::from(self.max_decode_bytes),
                details: "decoded byte size exceeds decode limits",
            });
        }

        Ok(())
    }

    /// Convenience: validate for a U8 image (1 byte per sample).
    pub fn validate_u8_image(&self, width: u32, height: u32, bands: u32) -> Result<(), ViprsError> {
        self.validate(width, height, bands, 1)
    }
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self::default_safe()
    }
}

/// Request-scoped runtime budgets for pipeline execution.
///
/// Clone a single instance into each request handler to enforce one shared concurrency gate while
/// also bounding output pixels and bytes per request.
///
/// # Examples
/// ```rust
/// # use viprs::domain::limits::ResourceLimits;
/// let limits = ResourceLimits::new(1_000, 4_096, 2);
/// assert_eq!(limits.max_concurrent(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct ResourceLimits {
    max_pixels: u64,
    max_memory_bytes: u64,
    max_concurrent: usize,
    execution_limiter: Arc<ExecutionSemaphore>,
}

impl PartialEq for ResourceLimits {
    fn eq(&self, other: &Self) -> bool {
        self.max_pixels == other.max_pixels
            && self.max_memory_bytes == other.max_memory_bytes
            && self.max_concurrent == other.max_concurrent
    }
}

impl Eq for ResourceLimits {}

impl ResourceLimits {
    /// Create a new shared resource-limit set.
    #[must_use]
    pub fn new(max_pixels: u64, max_memory_bytes: u64, max_concurrent: usize) -> Self {
        let max_pixels = max_pixels.max(1);
        let max_memory_bytes = max_memory_bytes.max(1);
        let max_concurrent = max_concurrent.max(1);
        Self {
            max_pixels,
            max_memory_bytes,
            max_concurrent,
            execution_limiter: Arc::new(ExecutionSemaphore::new(max_concurrent)),
        }
    }

    /// Safe defaults for server use.
    #[must_use]
    pub fn default_safe() -> Self {
        Self::new(
            DecodeLimits::default_safe().max_pixels,
            DecodeLimits::default_safe().max_decode_bytes,
            std::thread::available_parallelism().map_or(4, std::num::NonZero::get),
        )
    }

    /// No practical bounds.
    #[must_use]
    pub fn unlimited() -> Self {
        Self::new(u64::MAX, u64::MAX, usize::MAX)
    }

    /// Maximum decoded or output pixel count per request.
    #[must_use]
    pub const fn max_pixels(&self) -> u64 {
        self.max_pixels
    }

    /// Maximum decoded or output bytes per request.
    #[must_use]
    pub const fn max_memory_bytes(&self) -> u64 {
        self.max_memory_bytes
    }

    /// Maximum in-flight pipeline executions that may run at once.
    #[must_use]
    pub const fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Convert to decode-time header checks.
    #[must_use]
    pub const fn decode_limits(&self) -> DecodeLimits {
        DecodeLimits {
            max_width: u32::MAX,
            max_height: u32::MAX,
            max_pixels: self.max_pixels,
            max_decode_bytes: self.max_memory_bytes,
        }
    }

    /// Validate a post-build output image against the configured request budget.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::ImageTooLarge`] when the output would exceed either
    /// the pixel or byte limit.
    pub fn validate_output(
        &self,
        width: u32,
        height: u32,
        bands: u32,
        bytes_per_sample: u32,
    ) -> Result<(), ViprsError> {
        let total_bytes = decoded_bytes(width, height, bands, bytes_per_sample);
        let pixels = u64::from(width) * u64::from(height);

        if pixels > self.max_pixels {
            return Err(ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes: total_bytes,
                limit_bytes: u128::from(self.max_pixels)
                    * u128::from(bands)
                    * u128::from(bytes_per_sample),
                details: "output pixel count exceeds resource limits",
            });
        }

        if total_bytes > u128::from(self.max_memory_bytes) {
            return Err(ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes: total_bytes,
                limit_bytes: u128::from(self.max_memory_bytes),
                details: "output byte size exceeds resource limits",
            });
        }

        Ok(())
    }

    #[must_use]
    pub(crate) fn execution_limiter(&self) -> Arc<ExecutionSemaphore> {
        Arc::clone(&self.execution_limiter)
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self::default_safe()
    }
}

#[derive(Debug)]
pub(crate) struct ExecutionSemaphore {
    state: Mutex<ExecutionSemaphoreState>,
    ready: Condvar,
}

#[derive(Clone, Copy, Debug)]
struct ExecutionSemaphoreState {
    max_concurrent: usize,
    in_flight: usize,
}

impl ExecutionSemaphore {
    pub(crate) fn new(max_concurrent: usize) -> Self {
        Self {
            state: Mutex::new(ExecutionSemaphoreState {
                max_concurrent: max_concurrent.max(1),
                in_flight: 0,
            }),
            ready: Condvar::new(),
        }
    }

    #[must_use]
    pub(crate) fn acquire(&self) -> ExecutionPermit<'_> {
        let mut state = self.lock_state();
        while state.in_flight >= state.max_concurrent {
            state = self.wait_ready(state);
        }
        state.in_flight += 1;
        ExecutionPermit { semaphore: self }
    }

    fn release(&self) {
        let mut state = self.lock_state();
        if state.in_flight > 0 {
            state.in_flight -= 1;
        }
        self.ready.notify_one();
    }

    fn lock_state(&self) -> MutexGuard<'_, ExecutionSemaphoreState> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn wait_ready<'a>(
        &self,
        state: MutexGuard<'a, ExecutionSemaphoreState>,
    ) -> MutexGuard<'a, ExecutionSemaphoreState> {
        match self.ready.wait(state) {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[cfg(test)]
    fn in_flight(&self) -> usize {
        self.lock_state().in_flight
    }
}

pub(crate) struct ExecutionPermit<'a> {
    semaphore: &'a ExecutionSemaphore,
}

impl Drop for ExecutionPermit<'_> {
    fn drop(&mut self) {
        self.semaphore.release();
    }
}

#[inline]
fn decoded_bytes(width: u32, height: u32, bands: u32, bytes_per_sample: u32) -> u128 {
    u128::from(width) * u128::from(height) * u128::from(bands) * u128::from(bytes_per_sample)
}

#[cfg(test)]
mod tests {
    use super::{DecodeLimits, ResourceLimits};
    use crate::domain::error::ViprsError;

    #[test]
    fn default_safe_uses_server_defaults() {
        assert_eq!(
            DecodeLimits::default_safe(),
            DecodeLimits {
                max_width: 32_768,
                max_height: 32_768,
                max_pixels: 256_000_000,
                max_decode_bytes: 4_294_967_296,
            }
        );
    }

    #[test]
    fn unlimited_disables_all_limits() {
        assert_eq!(
            DecodeLimits::unlimited(),
            DecodeLimits {
                max_width: u32::MAX,
                max_height: u32::MAX,
                max_pixels: u64::MAX,
                max_decode_bytes: u64::MAX,
            }
        );
    }

    #[test]
    fn validate_accepts_small_images() {
        let limits = DecodeLimits::default_safe();
        assert!(limits.validate(640, 480, 3, 1).is_ok());
    }

    #[test]
    fn validate_rejects_oversized_width() {
        let limits = DecodeLimits::default_safe();

        assert!(matches!(
            limits.validate(40_000, 100, 3, 1),
            Err(ViprsError::ImageTooLarge {
                width: 40_000,
                height: 100,
                bands: 3,
                bytes: 12_000_000,
                limit_bytes: 9_830_400,
                details: "image dimensions exceed decode limits",
            })
        ));
    }

    #[test]
    fn validate_rejects_oversized_pixels() {
        let limits = DecodeLimits {
            max_width: u32::MAX,
            max_height: u32::MAX,
            max_pixels: 100,
            max_decode_bytes: u64::MAX,
        };

        assert!(matches!(
            limits.validate(11, 10, 4, 1),
            Err(ViprsError::ImageTooLarge {
                width: 11,
                height: 10,
                bands: 4,
                bytes: 440,
                limit_bytes: 400,
                details: "pixel count exceeds decode limits",
            })
        ));
    }

    #[test]
    fn validate_rejects_oversized_bytes() {
        let limits = DecodeLimits {
            max_width: u32::MAX,
            max_height: u32::MAX,
            max_pixels: u64::MAX,
            max_decode_bytes: 255,
        };

        assert!(matches!(
            limits.validate(8, 8, 4, 1),
            Err(ViprsError::ImageTooLarge {
                width: 8,
                height: 8,
                bands: 4,
                bytes: 256,
                limit_bytes: 255,
                details: "decoded byte size exceeds decode limits",
            })
        ));
    }

    #[test]
    fn resource_limits_default_safe_tracks_decode_defaults() {
        let limits = ResourceLimits::default_safe();

        assert_eq!(limits.max_pixels(), DecodeLimits::default_safe().max_pixels);
        assert_eq!(
            limits.max_memory_bytes(),
            DecodeLimits::default_safe().max_decode_bytes
        );
        assert!(limits.max_concurrent() >= 1);
    }

    #[test]
    fn resource_limits_validate_output_rejects_oversized_bytes() {
        let limits = ResourceLimits::new(16, 3, 1);

        assert!(matches!(
            limits.validate_output(2, 2, 1, 1),
            Err(ViprsError::ImageTooLarge {
                width: 2,
                height: 2,
                bands: 1,
                bytes: 4,
                limit_bytes: 3,
                details: "output byte size exceeds resource limits",
            })
        ));
    }

    #[test]
    fn resource_limits_clones_share_execution_semaphore() {
        let limits = ResourceLimits::new(16, 16, 1);
        let cloned = limits.clone();
        let execution_limiter = limits.execution_limiter();
        let permit = execution_limiter.acquire();

        assert_eq!(cloned.execution_limiter().in_flight(), 1);

        drop(permit);

        assert_eq!(cloned.execution_limiter().in_flight(), 0);
    }
}
