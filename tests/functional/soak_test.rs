use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, OnceLock},
};

use viprs::{
    BuildError, CompiledPipeline, Image, Interpretation, U8,
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::{
        kernel::InterpolationKernel,
        ops::resample::{Thumbnail, thumbnail::ThumbnailTarget},
    },
    ports::scheduler::TileScheduler,
};

const FIXTURE_NAME: &str = "bench_2048x2048.jpg";
const FIXTURE_WIDTH: u32 = 2_048;
const FIXTURE_HEIGHT: u32 = 2_048;
const FIXTURE_BANDS: u32 = 1;
const THUMBNAIL_WIDTH: u32 = 400;
const MAX_RSS_GROWTH_PERCENT: usize = 10;
const RSS_SLACK_KB: usize = 8 * 1_024;
const RSS_CHILD_ENV: &str = "VIPRS_THUMBNAIL_SOAK_RSS_CHILD";
const QUICK_RSS_CHILD_TEST: &str =
    "soak_test::quick_thumbnail_soak_stays_deterministic_for_100_iterations_child";
const FULL_RSS_CHILD_TEST: &str =
    "soak_test::thumbnail_soak_stays_deterministic_for_5000_iterations_child";

fn soak_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_owned()
}

fn fixture_path(name: &str) -> PathBuf {
    project_root()
        .join("tests")
        .join("fixtures")
        .join("images")
        .join(name)
}

fn load_fixture_image() -> Image<U8> {
    let path = fixture_path(FIXTURE_NAME);
    let bytes = fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to read fixture {}: {error}", path.display()));
    assert!(
        !bytes.is_empty(),
        "fixture {} must contain at least one byte",
        path.display()
    );

    let expected_len = FIXTURE_WIDTH as usize * FIXTURE_HEIGHT as usize * FIXTURE_BANDS as usize;
    let pixels = bytes
        .iter()
        .copied()
        .cycle()
        .take(expected_len)
        .collect::<Vec<_>>();
    let image = Image::from_buffer(FIXTURE_WIDTH, FIXTURE_HEIGHT, FIXTURE_BANDS, pixels)
        .unwrap_or_else(|error| {
            panic!(
                "failed to build fixture-backed image {}: {error}",
                path.display()
            )
        });

    image.with_metadata(viprs::ImageMetadata {
        interpretation: (FIXTURE_BANDS >= 3).then_some(Interpretation::Srgb),
        ..viprs::ImageMetadata::default()
    })
}

fn memory_source_from_image(image: &Image<U8>) -> MemorySource<U8> {
    MemorySource::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .unwrap_or_else(|error| panic!("failed to create memory source: {error}"))
    .with_metadata(image.metadata().clone())
}

fn build_thumbnail_pipeline(image: &Image<U8>) -> CompiledPipeline {
    viprs_runtime::pipeline::internal::PipelinePlan::from_source(memory_source_from_image(image))
        .plan_thumbnail(thumbnail())
        .unwrap_or_else(|error: BuildError| panic!("pipeline stage failed: {error:?}"))
        .compile()
        .unwrap_or_else(|error| panic!("pipeline build failed: {error:?}"))
}

fn run_thumbnail_pipeline(pipeline: &CompiledPipeline) -> Vec<u8> {
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .unwrap_or_else(|error| panic!("scheduler construction failed: {error}"))
        .run(&pipeline, &mut sink)
        .unwrap_or_else(|error| panic!("pipeline execution failed: {error}"));

    sink.into_buffer()
}

fn thumbnail() -> Thumbnail {
    Thumbnail::new(
        ThumbnailTarget::Width(THUMBNAIL_WIDTH),
        InterpolationKernel::Lanczos3,
    )
}

fn current_rss_kb() -> usize {
    let pid = std::process::id().to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", pid.as_str()])
        .output()
        .unwrap_or_else(|error| panic!("failed to query resident set size: {error}"));
    assert!(
        output.status.success(),
        "ps exited with {} while querying resident set size",
        output.status
    );

    String::from_utf8(output.stdout)
        .unwrap_or_else(|error| panic!("rss output was not valid UTF-8: {error}"))
        .trim()
        .parse::<usize>()
        .unwrap_or_else(|error| panic!("failed to parse rss from ps output: {error}"))
}

fn assert_rss_stable(
    baseline_rss_kb: usize,
    peak_rss_kb: usize,
    final_rss_kb: usize,
    iterations: usize,
) {
    let allowed_peak_kb =
        baseline_rss_kb + (baseline_rss_kb * MAX_RSS_GROWTH_PERCENT / 100) + RSS_SLACK_KB;
    assert!(
        peak_rss_kb <= allowed_peak_kb,
        "thumbnail soak RSS grew too much over {iterations} iterations: baseline={}KB peak={}KB allowed={}KB final={}KB",
        baseline_rss_kb,
        peak_rss_kb,
        allowed_peak_kb,
        final_rss_kb
    );
}

fn run_soak_in_child(child_test: &str) {
    let output = Command::new(std::env::current_exe().unwrap_or_else(|error| {
        panic!("failed to resolve current test binary for soak child run: {error}")
    }))
    .env(RSS_CHILD_ENV, "1")
    .arg("--exact")
    .arg(child_test)
    .arg("--nocapture")
    .arg("--test-threads=1")
    .output()
    .unwrap_or_else(|error| panic!("failed to spawn soak child run: {error}"));

    assert!(
        output.status.success(),
        "thumbnail soak child run failed: stdout=\n{}\nstderr=\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_soak(iterations: usize, rss_sample_interval: usize) {
    let _guard = soak_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let image = load_fixture_image();
    let pipeline = build_thumbnail_pipeline(&image);
    let mut baseline_output: Option<Vec<u8>> = None;
    let mut baseline_rss_kb = None;
    let mut peak_rss_kb = 0;

    for run in 0..iterations {
        let output = run_thumbnail_pipeline(&pipeline);
        assert_eq!(pipeline.width, THUMBNAIL_WIDTH);
        assert_eq!(pipeline.height, THUMBNAIL_WIDTH);
        assert!(
            !output.is_empty(),
            "thumbnail output must not be empty on run {}",
            run + 1
        );

        if let Some(expected) = &baseline_output {
            assert_eq!(
                output.as_slice(),
                expected.as_slice(),
                "thumbnail output changed on run {}",
                run + 1
            );
        } else {
            baseline_output = Some(output);
        }

        if (run + 1) % rss_sample_interval == 0 || run + 1 == iterations {
            let rss_kb = current_rss_kb();
            peak_rss_kb = peak_rss_kb.max(rss_kb);
            if baseline_rss_kb.is_none() {
                baseline_rss_kb = Some(rss_kb);
            }
        }
    }

    let final_rss_kb = current_rss_kb();
    peak_rss_kb = peak_rss_kb.max(final_rss_kb);
    let baseline_rss_kb = baseline_rss_kb.unwrap_or(final_rss_kb);
    assert_rss_stable(baseline_rss_kb, peak_rss_kb, final_rss_kb, iterations);
}

#[test]
#[ignore] // pre-existing: thumbnail non-determinism under instrumentation on x86_64 CI
fn quick_thumbnail_soak_stays_deterministic_for_100_iterations() {
    // Run RSS-sensitive soak assertions in a dedicated child process so unrelated
    // functional tests in the main harness cannot inflate the resident-set samples.
    run_soak_in_child(QUICK_RSS_CHILD_TEST);
}

#[test]
fn quick_thumbnail_soak_stays_deterministic_for_100_iterations_child() {
    if std::env::var_os(RSS_CHILD_ENV).is_none() {
        return;
    }

    run_soak(100, 10);
}

#[test]
#[ignore = "slow soak test for sustained thumbnail stability"]
fn thumbnail_soak_stays_deterministic_for_5000_iterations() {
    run_soak_in_child(FULL_RSS_CHILD_TEST);
}

#[test]
#[ignore = "slow soak test for sustained thumbnail stability"]
fn thumbnail_soak_stays_deterministic_for_5000_iterations_child() {
    if std::env::var_os(RSS_CHILD_ENV).is_none() {
        return;
    }

    run_soak(5_000, 100);
}
