//! Runtime adapters, schedulers, and pipeline implementations for viprs.

pub mod cache;
/// Runtime routing for dynamic colorspace-conversion graphs.
pub mod colour_dispatcher;
pub mod concretized_bridge;
pub mod foreign;
#[cfg(feature = "fft")]
pub mod freqfilt;
pub mod image_api;
pub mod image_pipeline;
/// Tracing and event macros used by runtime adapters.
pub mod instrumentation;
#[cfg(all(test, feature = "_integration"))]
mod integration_tests;
/// Optional lock profiling helpers for scheduler contention analysis.
pub mod lock_instrumentation;
#[doc(hidden)]
pub mod pipeline;
pub mod process;
pub mod scheduler;
/// Concrete image sinks for files, memory buffers, and concurrent writers.
pub mod sinks;
pub mod sources;

#[doc(hidden)]
pub mod adapters {
    pub use crate::cache;
    pub use crate::concretized_bridge;
    pub use crate::foreign;
    #[cfg(feature = "fft")]
    pub use crate::freqfilt;
    pub use crate::image_api;
    pub use crate::image_pipeline;
    pub use crate::instrumentation;
    pub use crate::lock_instrumentation;
    pub(crate) use crate::pipeline;
    pub use crate::process;
    pub use crate::scheduler;
    pub use crate::sinks;
    pub use crate::sources;

    pub mod codecs {
        pub use viprs_codecs::*;
    }
}

#[doc(hidden)]
pub mod domain {
    pub use viprs_core::cancel;
    pub use viprs_core::codec_options;
    pub use viprs_core::colorspace;
    pub use viprs_core::colour;
    pub mod colour_dispatcher {
        pub use crate::colour_dispatcher::*;
    }
    pub use viprs_core::concretize;
    pub use viprs_core::draw;
    pub use viprs_core::error;
    pub use viprs_core::format;
    pub use viprs_core::image;
    pub use viprs_core::kernel;
    pub use viprs_core::limits;
    pub use viprs_core::op;
    pub mod ops {
        pub use viprs_ops_colour::*;
        pub use viprs_ops_composite::*;
        pub use viprs_ops_pixel::*;
        pub use viprs_ops_spatial::*;
    }
    pub use viprs_core::reducer;
    pub use viprs_core::reorder;
    pub use viprs_core::resample;
}

#[doc(hidden)]
pub mod ports {
    pub use viprs_ports::codec;
    pub use viprs_ports::scheduler;
    pub use viprs_ports::sink;
    pub use viprs_ports::source;
}
