use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use viprs::{
    BuildError, Image, ImageMetadata, Interpretation, U8,
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        kernel::InterpolationKernel,
        ops::resample::{Thumbnail, resize::Resize, thumbnail::ThumbnailTarget},
    },
    ports::scheduler::TileScheduler,
};

const QUICK_ITERATIONS: usize = 500;
const FULL_ITERATIONS: usize = 5_000;
const ITERATION_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_RSS_GROWTH_PERCENT: usize = 10;
const RSS_SLACK_KB: usize = 8 * 1_024;
const RSS_SAMPLE_INTERVAL: usize = 50;
const BASELINE_WARMUP_ITERATIONS: usize = 100;
const MAX_SYNTHETIC_DIMENSION: u32 = 192;
const MIN_SYNTHETIC_DIMENSION: u32 = 48;

fn soak_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct FixtureSpec {
    name: String,
    image: Image<U8>,
}

#[derive(Clone, Copy, Debug)]
enum PipelineKind {
    Thumbnail,
    Crop,
    ExtractArea,
    Resize,
    Heavy,
}

#[derive(Clone, Copy)]
struct Lcg {
    state: u64,
}

impl Lcg {
    const MULTIPLIER: u64 = 6_364_136_223_846_793_005;
    const INCREMENT: u64 = 1;

    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::INCREMENT);
        (self.state >> 32) as u32
    }

    fn index(&mut self, upper_bound: usize) -> usize {
        assert!(upper_bound > 0, "upper_bound must be positive");
        (self.next_u32() as usize) % upper_bound
    }

    fn range_u32(&mut self, start: u32, end_inclusive: u32) -> u32 {
        assert!(start <= end_inclusive, "invalid range");
        start + (self.next_u32() % (end_inclusive - start + 1))
    }

    fn pick<T: Copy>(&mut self, values: &[T]) -> T {
        values[self.index(values.len())]
    }
}

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_owned()
}

fn fixture_directory() -> PathBuf {
    project_root().join("tests").join("fixtures").join("images")
}

fn load_fixture_pool() -> Vec<FixtureSpec> {
    let mut fixtures = fs::read_dir(fixture_directory())
        .expect("fixture directory must exist")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_image_fixture(path))
        .collect::<Vec<_>>();
    fixtures.sort();

    let loaded = fixtures
        .into_iter()
        .map(|path| build_fixture_spec(&path))
        .collect::<Vec<_>>();
    assert!(
        !loaded.is_empty(),
        "expected at least one image fixture in tests/fixtures/images"
    );
    loaded
}

fn is_image_fixture(path: &Path) -> bool {
    matches!(
        path.extension().and_then(std::ffi::OsStr::to_str),
        Some("avif" | "exr" | "gif" | "heic" | "jpg" | "jp2" | "jxl" | "png" | "tif" | "webp")
    )
}

fn build_fixture_spec(path: &Path) -> FixtureSpec {
    let bytes = fs::read(path)
        .unwrap_or_else(|error| panic!("failed to read fixture {}: {error}", path.display()));
    assert!(
        !bytes.is_empty(),
        "fixture {} must contain at least one byte",
        path.display()
    );

    let name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or_else(|| {
            panic!(
                "fixture path {} must have a valid UTF-8 name",
                path.display()
            )
        })
        .to_owned();
    let (width, height) = synthetic_dimensions(&name, &bytes);
    let bands = synthetic_bands(&name);
    let pixel_count = (width as usize) * (height as usize) * (bands as usize);
    let pixels = bytes
        .iter()
        .copied()
        .cycle()
        .take(pixel_count)
        .collect::<Vec<_>>();
    let metadata = ImageMetadata {
        interpretation: (bands >= 3).then_some(Interpretation::Srgb),
        ..ImageMetadata::default()
    };
    let image = Image::from_buffer(width, height, bands, pixels)
        .unwrap_or_else(|error| panic!("failed to synthesize image for {name}: {error}"))
        .with_metadata(metadata);

    FixtureSpec { name, image }
}

fn synthetic_dimensions(name: &str, bytes: &[u8]) -> (u32, u32) {
    if let Some((width, height)) = parse_dimensions_from_name(name) {
        return (
            width.clamp(MIN_SYNTHETIC_DIMENSION, MAX_SYNTHETIC_DIMENSION),
            height.clamp(MIN_SYNTHETIC_DIMENSION, MAX_SYNTHETIC_DIMENSION),
        );
    }

    let hash = bytes.iter().fold(0u64, |acc, byte| {
        acc.wrapping_mul(16_777_619)
            .wrapping_add(u64::from(*byte) + 1)
    });
    let span = MAX_SYNTHETIC_DIMENSION - MIN_SYNTHETIC_DIMENSION + 1;
    let width = MIN_SYNTHETIC_DIMENSION + (hash as u32 % span);
    let height = MIN_SYNTHETIC_DIMENSION + ((hash.rotate_left(13) as u32) % span);
    (width, height)
}

fn parse_dimensions_from_name(name: &str) -> Option<(u32, u32)> {
    name.split(|ch: char| !(ch.is_ascii_digit() || ch == 'x'))
        .find_map(|segment| {
            let (width, height) = segment.split_once('x')?;
            Some((width.parse().ok()?, height.parse().ok()?))
        })
}

fn synthetic_bands(name: &str) -> u32 {
    if name.contains("rgba") {
        4
    } else if name.contains("gray") {
        1
    } else {
        3
    }
}

fn memory_source_from_image(image: &Image<U8>) -> MemorySource<U8> {
    MemorySource::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .expect("memory source construction must succeed")
    .with_metadata(image.metadata().clone())
}

fn current_rss_kb() -> usize {
    let pid = std::process::id().to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", pid.as_str()])
        .output()
        .expect("failed to query resident set size");
    assert!(
        output.status.success(),
        "ps exited with {} while querying resident set size",
        output.status
    );

    String::from_utf8(output.stdout)
        .expect("rss output must be valid UTF-8")
        .trim()
        .parse::<usize>()
        .expect("failed to parse rss from ps output")
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
        "soak RSS grew too much over {iterations} iterations: baseline={}KB peak={}KB allowed={}KB final={}KB",
        baseline_rss_kb,
        peak_rss_kb,
        allowed_peak_kb,
        final_rss_kb
    );
}

fn choose_pipeline(rng: &mut Lcg) -> PipelineKind {
    rng.pick(&[
        PipelineKind::Thumbnail,
        PipelineKind::Crop,
        PipelineKind::ExtractArea,
        PipelineKind::Resize,
        PipelineKind::Heavy,
    ])
}

fn thumbnail_target_width(width: u32, rng: &mut Lcg) -> u32 {
    let upper = width.clamp(64, 256);
    let lower = upper.min(128);
    rng.range_u32(lower, upper)
}

fn random_extract(image: &Image<U8>, rng: &mut Lcg, centered: bool) -> (u32, u32, u32, u32) {
    let max_width = image.width();
    let max_height = image.height();
    let min_width = 1.max(max_width / 4);
    let min_height = 1.max(max_height / 4);
    let width = rng.range_u32(min_width, max_width);
    let height = rng.range_u32(min_height, max_height);

    if centered {
        (
            (max_width - width) / 2,
            (max_height - height) / 2,
            width,
            height,
        )
    } else {
        let x = rng.range_u32(0, max_width - width);
        let y = rng.range_u32(0, max_height - height);
        (x, y, width, height)
    }
}

fn random_resize(rng: &mut Lcg) -> Resize {
    let scale = rng.pick(&[0.5, 0.75, 1.25]);
    let kernel = rng.pick(&[
        InterpolationKernel::Nearest,
        InterpolationKernel::Bilinear,
        InterpolationKernel::Bicubic,
        InterpolationKernel::Lanczos3,
    ]);
    Resize::new(scale, scale, kernel)
}

fn configure_pipeline(
    builder: PipelineBuilder,
    image: &Image<U8>,
    pipeline_kind: PipelineKind,
    rng: &mut Lcg,
) -> Result<PipelineBuilder, BuildError> {
    match pipeline_kind {
        PipelineKind::Thumbnail => {
            let target = thumbnail_target_width(image.width(), rng);
            builder.thumbnail(Thumbnail::new(
                ThumbnailTarget::Width(target),
                InterpolationKernel::Lanczos3,
            ))
        }
        PipelineKind::Crop => {
            let (x, y, width, height) = random_extract(image, rng, true);
            builder.extract_area(x, y, width, height)
        }
        PipelineKind::ExtractArea => {
            let (x, y, width, height) = random_extract(image, rng, false);
            builder.extract_area(x, y, width, height)
        }
        PipelineKind::Resize => builder.resize(random_resize(rng)),
        PipelineKind::Heavy => {
            let (x, y, width, height) = random_extract(image, rng, false);
            let target = thumbnail_target_width(width, rng);
            builder
                .extract_area(x, y, width, height)?
                .resize(random_resize(rng))?
                .thumbnail(Thumbnail::new(
                    ThumbnailTarget::Width(target),
                    InterpolationKernel::Lanczos3,
                ))
        }
    }
}

fn execute_iteration(
    fixture: &FixtureSpec,
    pipeline_kind: PipelineKind,
    rng: &mut Lcg,
) -> (u32, u32, usize) {
    let builder = PipelineBuilder::from_source(memory_source_from_image(&fixture.image));
    let pipeline = configure_pipeline(builder, &fixture.image, pipeline_kind, rng)
        .unwrap_or_else(|error| {
            panic!(
                "pipeline construction failed for fixture {} with {pipeline_kind:?}: {error:?}",
                fixture.name
            )
        })
        .build()
        .unwrap_or_else(|error| {
            panic!(
                "pipeline build failed for fixture {} with {pipeline_kind:?}: {error:?}",
                fixture.name
            )
        });

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .expect("scheduler construction must succeed")
        .run(&pipeline, &mut sink)
        .unwrap_or_else(|error| {
            panic!(
                "pipeline execution failed for fixture {} with {pipeline_kind:?}: {error:?}",
                fixture.name
            )
        });

    (pipeline.width, pipeline.height, sink.into_buffer().len())
}

fn run_soak(iterations: usize) {
    let _guard = soak_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let fixtures = load_fixture_pool();
    let mut rng = Lcg::new(0x26_00_5EED);
    let mut baseline_rss_kb = None;
    let mut peak_rss_kb = 0;

    for iteration in 0..iterations {
        let fixture = &fixtures[rng.index(fixtures.len())];
        let pipeline_kind = choose_pipeline(&mut rng);
        let started_at = Instant::now();
        let (width, height, output_len) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                execute_iteration(fixture, pipeline_kind, &mut rng)
            }))
            .unwrap_or_else(|panic_payload| {
                std::panic::resume_unwind(Box::new(format!(
                    "iteration {} panicked for fixture {} with {pipeline_kind:?}: {:?}",
                    iteration + 1,
                    fixture.name,
                    panic_payload
                )))
            });
        let elapsed = started_at.elapsed();

        assert!(
            elapsed <= ITERATION_TIMEOUT,
            "iteration {} exceeded {:?}: fixture={} pipeline={pipeline_kind:?} elapsed={elapsed:?}",
            iteration + 1,
            ITERATION_TIMEOUT,
            fixture.name
        );
        assert!(
            width > 0 && height > 0,
            "iteration {} produced invalid dimensions {}x{} for fixture {} with {pipeline_kind:?}",
            iteration + 1,
            width,
            height,
            fixture.name
        );
        assert!(
            output_len > 0,
            "iteration {} produced empty output for fixture {} with {pipeline_kind:?}",
            iteration + 1,
            fixture.name
        );

        if iteration + 1 == BASELINE_WARMUP_ITERATIONS {
            let rss_kb = current_rss_kb();
            baseline_rss_kb = Some(rss_kb);
            peak_rss_kb = peak_rss_kb.max(rss_kb);
        } else if iteration + 1 > BASELINE_WARMUP_ITERATIONS
            && ((iteration + 1) % RSS_SAMPLE_INTERVAL == 0 || iteration + 1 == iterations)
        {
            peak_rss_kb = peak_rss_kb.max(current_rss_kb());
        }
    }

    let final_rss_kb = current_rss_kb();
    peak_rss_kb = peak_rss_kb.max(final_rss_kb);
    let baseline_rss_kb = baseline_rss_kb.unwrap_or(final_rss_kb);
    assert_rss_stable(baseline_rss_kb, peak_rss_kb, final_rss_kb, iterations);
}

#[test]
fn soak_stability_quick_random_pipelines_stay_stable_for_500_iterations() {
    run_soak(QUICK_ITERATIONS);
}

#[test]
#[ignore = "full soak coverage for sustained stability; CI uses the 500-iteration quick variant"]
fn soak_stability_random_pipelines_stay_stable_for_5000_iterations() {
    run_soak(FULL_ITERATIONS);
}
