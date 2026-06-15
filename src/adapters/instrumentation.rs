/// Span creation macro — compiles to nothing without `feature = "tracing"`.
macro_rules! viprs_span {
    ($level:expr, $name:expr, $($field:tt)*) => {
        #[cfg(feature = "tracing")]
        let _span = tracing::span!($level, $name, $($field)*).entered();
        #[cfg(not(feature = "tracing"))]
        let _ = ();
    };
    ($level:expr, $name:expr) => {
        #[cfg(feature = "tracing")]
        let _span = tracing::span!($level, $name).entered();
        #[cfg(not(feature = "tracing"))]
        let _ = ();
    };
}

#[allow(unused_macros)]
macro_rules! viprs_event {
    ($level:expr, $($arg:tt)*) => {
        #[cfg(feature = "tracing")]
        tracing::event!($level, $($arg)*);
    };
}

#[allow(unused_imports)]
pub(crate) use viprs_event;
pub(crate) use viprs_span;
