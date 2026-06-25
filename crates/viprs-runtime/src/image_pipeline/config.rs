use std::num::NonZeroUsize;

use crate::scheduler::rayon_scheduler::RayonScheduler;
use viprs_core::{error::ViprsError, limits::ResourceLimits};

/// Runtime policy used when executing an `ImagePipeline`.
///
/// This skeleton exposes thread count and optional resource limits. Later work
/// will add temp backing, scheduler selection, and observability hooks here.
///
/// # Examples
///
/// ```rust
/// use viprs_runtime::image_pipeline::ProcessingConfig;
///
/// let config = ProcessingConfig::default().with_threads(1);
/// assert_eq!(config.threads(), Some(1));
/// ```
#[derive(Clone, Debug, Default)]
pub struct ProcessingConfig {
    threads: Option<NonZeroUsize>,
    resource_limits: Option<ResourceLimits>,
}

impl ProcessingConfig {
    /// Set the number of scheduler worker threads.
    ///
    /// Values below one are clamped to one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::ProcessingConfig;
    ///
    /// assert_eq!(ProcessingConfig::default().with_threads(0).threads(), Some(1));
    /// ```
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = NonZeroUsize::new(threads.max(1));
        self
    }

    /// Attach request-scoped resource limits to execution.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_core::limits::ResourceLimits;
    /// use viprs_runtime::image_pipeline::ProcessingConfig;
    ///
    /// let config = ProcessingConfig::default()
    ///     .with_resource_limits(ResourceLimits::new(10, 10, 1));
    /// assert!(config.resource_limits().is_some());
    /// ```
    #[must_use]
    pub fn with_resource_limits(mut self, resource_limits: ResourceLimits) -> Self {
        self.resource_limits = Some(resource_limits);
        self
    }

    /// Return the configured worker thread count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::ProcessingConfig;
    ///
    /// assert_eq!(ProcessingConfig::default().threads(), None);
    /// ```
    #[must_use]
    pub const fn threads(&self) -> Option<usize> {
        match self.threads {
            Some(threads) => Some(threads.get()),
            None => None,
        }
    }

    /// Return configured resource limits, when present.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::ProcessingConfig;
    ///
    /// assert!(ProcessingConfig::default().resource_limits().is_none());
    /// ```
    #[must_use]
    pub const fn resource_limits(&self) -> Option<&ResourceLimits> {
        self.resource_limits.as_ref()
    }

    pub(in crate::image_pipeline) fn validate_output(
        &self,
        width: u32,
        height: u32,
        bands: u32,
        bytes_per_sample: u32,
    ) -> Result<(), ViprsError> {
        self.resource_limits.as_ref().map_or(Ok(()), |limits| {
            limits.validate_output(width, height, bands, bytes_per_sample)
        })
    }

    pub(in crate::image_pipeline) fn into_scheduler(self) -> Result<RayonScheduler, ViprsError> {
        let threads = self
            .threads()
            .unwrap_or_else(RayonScheduler::default_threads);
        let scheduler = RayonScheduler::new(threads)?;
        Ok(match self.resource_limits {
            Some(limits) => scheduler.with_max_concurrent_pipelines(limits.max_concurrent()),
            None => scheduler,
        })
    }
}
