use serde::{Deserialize, Serialize};
use viprs::domain::format::{F32, U8, U16};
use viprs::domain::image::InMemoryImage;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScenarioSpec {
    pub key: &'static str,
    pub input: &'static str,
    pub description: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchFixtureSpec {
    pub size: u32,
    pub width: u32,
    pub height: u32,
    pub input: &'static str,
}

pub enum TiffSaveInput {
    U8(InMemoryImage<U8>),
    U16(InMemoryImage<U16>),
}

pub enum BenchImage {
    U8(InMemoryImage<U8>),
    U16(InMemoryImage<U16>),
    F32(InMemoryImage<F32>),
}

#[derive(Serialize, Deserialize)]
pub struct BenchResult {
    pub backend: String,
    pub input: String,
    pub operation: String,
    pub iterations: usize,
    pub wall_ns: Vec<u64>,
    pub peak_rss_kb: u64,
    pub minor_faults: u64,
    pub major_faults: u64,
    pub vol_ctx_switches: u64,
    pub invol_ctx_switches: u64,
}

#[derive(Serialize)]
pub struct Comparison {
    pub libvips: Option<BenchResult>,
    pub viprs: Option<BenchResult>,
    pub ratios: Option<Ratios>,
}

#[derive(Serialize)]
pub struct Ratios {
    pub latency_p50: f64,
    pub latency_p95: f64,
    pub rss: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SummaryRow {
    pub op: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub op_args: Vec<String>,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scenario: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u32>,
    pub viprs_p50_ms: f64,
    pub viprs_p95_ms: f64,
    pub libvips_p50_ms: Option<f64>,
    pub libvips_p95_ms: Option<f64>,
    pub ratio: Option<f64>,
    pub ratio_p95: Option<f64>,
}

/// One row appended to `tools/bench-vs-libvips/results/trend.jsonl`.
///
/// The file is newline-delimited JSON (one record per line) so it can be
/// streamed or queried with `jq` without loading the entire history.
/// Each CI run appends exactly one record per (op, size) pair.
#[derive(Serialize)]
pub struct TrendRecord {
    /// ISO-8601 UTC timestamp of the run.
    pub date: String,
    /// Short git SHA of HEAD at bench time (empty string if unavailable).
    pub git_sha: String,
    /// Operation name, e.g. `"invert"`, `"thumbnail"`, `"resize"`, `"sharpen"`.
    pub op: String,
    /// Operation arguments captured as part of the canonical scenario identity.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub op_args: Vec<String>,
    /// Fixture width key in pixels (for example 512 / 640 / 1920 / 2048 / 8192).
    pub size: u32,
    /// viprs p50 latency in milliseconds.
    pub viprs_p50_ms: f64,
    /// viprs p95 latency in milliseconds.
    pub viprs_p95_ms: f64,
    /// libvips p50 latency in milliseconds (`null` when runner unavailable).
    pub libvips_p50_ms: Option<f64>,
    /// libvips p95 latency in milliseconds (`null` when runner unavailable).
    pub libvips_p95_ms: Option<f64>,
    /// `viprs_p50 / libvips_p50` — below 1.0 means viprs wins; `null` when libvips unavailable.
    pub ratio_p50: Option<f64>,
}
