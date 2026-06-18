//! Frequency-domain filtering, FFT, and spectrum operations.
/// Provides the `freqmult` module for this domain area.
pub mod freqmult;
#[cfg(feature = "fft")]
/// Forward FFT operation.
pub mod fwfft;
#[cfg(feature = "fft")]
/// Inverse FFT operation.
pub mod invfft;
/// Provides the `phasecor` module for this domain area.
pub mod phasecor;
/// Provides the `spectrum` module for this domain area.
pub mod spectrum;

pub use freqmult::FreqMultOp;
#[cfg(feature = "fft")]
pub use fwfft::FwFftOp;
#[cfg(feature = "fft")]
pub use invfft::InvFftOp;
pub use phasecor::PhasecorOp;
pub use spectrum::SpectrumOp;

/// Constant value for complex bands.
pub const COMPLEX_BANDS: u32 = 2;
