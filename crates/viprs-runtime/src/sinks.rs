/// Discard sink that forces evaluation without retaining output pixels.
pub mod discard;
/// Disk sink that overlaps tile generation with background strip flushes.
pub mod double_buffer;
/// Disk sink primitives that stream completed regions to a writer thread.
pub mod file_sink;
/// In-memory sink used by tests and `run_to_image` style execution.
pub mod memory;
/// Writer adapters that bridge sinks to concrete output targets.
pub mod writer;
