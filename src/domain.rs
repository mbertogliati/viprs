//! Core domain types and traits for viprs.
//!
//! This module is the public entry point for the pure domain layer: image containers,
//! band formats, colorspaces, operation traits, reducers, and supporting runtime metadata.
//! Most callers import domain building blocks from here before assembling pipelines or
//! implementing new operations.

/// Cancellation primitives for stopping long-running pipeline work.
pub mod cancel;
pub mod codec_options;
pub mod coeff;
pub mod colorspace;
/// Colour-conversion traits and helpers shared across colour operations.
pub mod colour;
/// Runtime routing for dynamic colorspace-conversion graphs.
pub mod colour_dispatcher;
pub mod concretize;
pub mod draw;
/// Typed error enums used across the domain and adapter layers.
pub mod error;
/// Band-format traits, identifiers, and sample math helpers.
pub mod format;
/// Core image, region, and tile container types.
pub mod image;
pub mod kernel;
/// Resource-limit types and validation helpers.
pub mod limits;
pub mod op;
pub mod ops;
pub mod reducer;
pub mod reducers;
pub mod reorder;
/// Resampling traits, filters, and high-level resize configuration.
pub mod resample;
/// SIMD abstraction helpers shared by performance-sensitive operations.
pub mod simd;
/// Aggregated image statistics produced by reducers.
pub mod stats;

pub use op::DemandHint;

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    fn collect_rust_files(root: &Path, files: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(root).expect("domain directory must be readable") {
            let entry = entry.expect("directory entry must be readable");
            let path = entry.path();
            if path.is_dir() {
                collect_rust_files(&path, files);
            } else if path.extension().and_then(std::ffi::OsStr::to_str) == Some("rs") {
                files.push(path);
            }
        }
    }

    #[test]
    fn dyn_operation_dispatch_sites_are_documented() {
        let mut files = Vec::new();
        collect_rust_files(
            Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/domain")),
            &mut files,
        );

        for path in files {
            let source = fs::read_to_string(&path).expect("domain source must be readable");
            let lines = source.lines().collect::<Vec<_>>();
            let first_dyn_site = lines.iter().position(|line| {
                let trimmed = line.trim_start();
                trimmed.contains("Box<dyn DynOperation>")
                    && !trimmed.starts_with("//")
                    && !trimmed.starts_with("/*")
            });

            let Some(first_dyn_site) = first_dyn_site else {
                continue;
            };

            let documented = lines[..first_dyn_site].iter().any(|line| {
                let trimmed = line.trim();
                trimmed.starts_with("// NOTE:") || trimmed.starts_with("// SAFETY:")
            });

            assert!(
                documented,
                "undocumented Box<dyn DynOperation> in {}; add a NOTE/SAFETY comment before the first dyn-dispatch site",
                path.display()
            );
        }
    }
}
