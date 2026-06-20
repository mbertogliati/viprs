use crate::format::BandFormatId;

/// Per-band statistics computed from an image or region.
///
/// Produced by `StatsReducer`. All per-band vectors have length == `bands`.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageStats {
    /// Number of image bands.
    pub bands: u32,
    /// Minimum sample value per band (in the native sample range).
    pub min: Vec<f64>,
    /// Maximum sample value per band.
    pub max: Vec<f64>,
    /// Arithmetic mean per band.
    pub mean: Vec<f64>,
    /// Population standard deviation per band.
    pub stddev: Vec<f64>,
}

/// Frequency histogram over quantized sample values for a single image band.
///
/// For `U8` images: 256 bins, one per sample value in [0, 255].
/// For `U16` images: 65536 bins. For float formats, bins cover [0.0, 1.0]
/// with `bin_count` quantization levels (caller-specified at construction).
///
/// Produced by `HistFindReducer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Histogram {
    /// Format of the source image.
    pub format: BandFormatId,
    /// Which band this histogram covers (0-indexed).
    pub band: u32,
    /// Bin counts. `bins[i]` is the number of pixels whose sample value falls
    /// in the i-th quantization bucket.
    pub bins: Vec<u64>,
}

impl Histogram {
    /// Total number of pixels counted (sum of all bins).
    #[must_use]
    pub fn total(&self) -> u64 {
        self.bins.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_total_is_sum_of_bins() {
        let h = Histogram {
            format: BandFormatId::U8,
            band: 0,
            bins: vec![10, 20, 30],
        };
        assert_eq!(h.total(), 60);
    }

    #[test]
    fn image_stats_fields_accessible() {
        let s = ImageStats {
            bands: 1,
            min: vec![0.0],
            max: vec![255.0],
            mean: vec![128.0],
            stddev: vec![64.0],
        };
        assert_eq!(s.bands, 1);
        assert!((s.mean[0] - 128.0).abs() < f64::EPSILON);
    }
}
