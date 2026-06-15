//! Concrete schedulers that execute compiled pipelines.
//!
//! Scheduler adapters turn the pipeline DAG into parallel tile execution
//! strategies while respecting demand hints, cache configuration, and sink
//! capabilities.

pub mod rayon_scheduler;
