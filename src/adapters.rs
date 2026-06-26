//! Compatibility re-exports for extracted runtime adapters.

pub use viprs_codecs as codecs;
/// Shared adapter-side caches and eviction policies.
pub use viprs_runtime::cache;
pub use viprs_runtime::foreign;
#[cfg(feature = "fft")]
pub use viprs_runtime::freqfilt;
pub use viprs_runtime::image_api;
pub use viprs_runtime::image_pipeline;
pub use viprs_runtime::process;
pub use viprs_runtime::scheduler;
/// Concrete image sinks for files, memory buffers, and concurrent writers.
pub use viprs_runtime::sinks;
/// Concrete image sources, generators, and cache-backed source adapters.
pub use viprs_runtime::sources;

pub(crate) mod concretized_bridge {
    #![allow(unused_imports)]
    pub use viprs_runtime::concretized_bridge::*;
}

pub(crate) mod instrumentation {
    #![allow(unused_imports)]
    pub use viprs_runtime::instrumentation::*;
}

pub(crate) mod lock_instrumentation {
    #![allow(unused_imports)]
    pub use viprs_runtime::lock_instrumentation::*;
}
