use std::{
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::LazyLock,
};

/// Resolves the absolute path of a CLI binary via `which`, falling back to the
/// well-known Homebrew location on macOS.
fn resolve_bin(name: &str, homebrew_fallback: &str) -> String {
    let output = Command::new("which")
        .arg(name)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned());
    output.unwrap_or_else(|| homebrew_fallback.to_owned())
}

static VIPS_BIN: LazyLock<String> = LazyLock::new(|| resolve_bin("vips", "/opt/homebrew/bin/vips"));
static VIPSHEADER_BIN: LazyLock<String> =
    LazyLock::new(|| resolve_bin("vipsheader", "/opt/homebrew/bin/vipsheader"));
const OUTPUT_PLACEHOLDER: &str = "{output}";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VipsBandFormat {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    F64,
}

impl VipsBandFormat {
    fn cli_name(self) -> &'static str {
        match self {
            Self::U8 => "uchar",
            Self::I8 => "char",
            Self::U16 => "ushort",
            Self::I16 => "short",
            Self::U32 => "uint",
            Self::I32 => "int",
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageSpec {
    pub width: u32,
    pub height: u32,
    pub bands: u32,
    pub format: VipsBandFormat,
}

impl ImageSpec {
    pub const fn new(width: u32, height: u32, bands: u32, format: VipsBandFormat) -> Self {
        Self {
            width,
            height,
            bands,
            format,
        }
    }

    fn expected_len(self) -> usize {
        self.width as usize
            * self.height as usize
            * self.bands as usize
            * self.format.bytes_per_sample()
    }
}

struct GeneratedGolden {
    bytes: Vec<u8>,
    format: VipsBandFormat,
}

/// Returns `true` when the libvips CLI tools are available on this machine.
///
/// Tests that compare output against the reference libvips should call this
/// and return early when it is `false` rather than panicking.
pub fn vips_available() -> bool {
    Path::new(VIPS_BIN.as_str()).exists() && Path::new(VIPSHEADER_BIN.as_str()).exists()
}

/// Returns `true` when the current test should return early because the
/// libvips CLI tools are unavailable in this environment.
#[must_use]
pub fn skip_without_vips() -> bool {
    if vips_available() {
        return false;
    }

    eprintln!(
        "skipping: libvips parity test requires the `vips` and `vipsheader` CLIs at {} and {}",
        &*VIPS_BIN, &*VIPSHEADER_BIN
    );
    true
}

/// Panics when libvips CLI tools are not installed.
///
/// Prefer checking [`skip_without_vips`] and returning early from the test instead.
#[track_caller]
pub fn require_vips() {
    assert!(
        !skip_without_vips(),
        "libvips parity test requires the `vips` and `vipsheader` CLIs at {} and {}; install libvips or mark the test #[ignore]",
        &*VIPS_BIN,
        &*VIPSHEADER_BIN
    );
}

fn manifest_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set by cargo test harness");
    PathBuf::from(manifest)
}

fn fixtures_dir() -> PathBuf {
    manifest_dir().join("tests").join("fixtures")
}

fn runtime_dir() -> PathBuf {
    manifest_dir().join("target").join("libvips-golden")
}

pub fn case_dir(op: &str, case: &str) -> PathBuf {
    runtime_dir().join(op).join(case)
}

pub fn fixture_path(op: &str, case: &str) -> PathBuf {
    fixtures_dir().join(op).join(format!("{case}.bin"))
}

fn ensure_parent(path: &Path) {
    let parent = path
        .parent()
        .expect("path used by golden tests must have a parent directory");
    fs::create_dir_all(parent)
        .unwrap_or_else(|e| panic!("failed to create directory {parent:?}: {e}"));
}

fn run_command(binary: &str, args: &[OsString]) {
    let output = Command::new(binary)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {binary}: {e}"));

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "{binary} failed with status {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status.code()
        );
    }
}

fn run_vips<I, S>(args: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let args = args
        .into_iter()
        .map(|arg| arg.as_ref().to_os_string())
        .collect::<Vec<_>>();
    run_command(&VIPS_BIN, &args);
}

fn run_vipsheader<I, S>(args: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let args = args
        .into_iter()
        .map(|arg| arg.as_ref().to_os_string())
        .collect::<Vec<_>>();
    let output = Command::new(VIPSHEADER_BIN.as_str())
        .args(&args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", &*VIPSHEADER_BIN));

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "{} failed with status {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            &*VIPSHEADER_BIN,
            output.status.code()
        );
    }

    String::from_utf8(output.stdout)
        .unwrap_or_else(|e| panic!("vipsheader output was not valid utf-8: {e}"))
        .trim()
        .to_owned()
}

pub fn write_vips_input(
    op: &str,
    case: &str,
    name: &str,
    bytes: &[u8],
    spec: ImageSpec,
) -> PathBuf {
    assert_eq!(
        bytes.len(),
        spec.expected_len(),
        "raw input length mismatch for op={op} case={case} name={name}"
    );

    let dir = case_dir(op, case);
    let raw_path = dir.join(format!("{name}.raw"));
    let image_path = dir.join(format!("{name}.v"));
    ensure_parent(&raw_path);

    fs::write(&raw_path, bytes)
        .unwrap_or_else(|e| panic!("failed to write raw input {raw_path:?}: {e}"));

    run_vips([
        OsString::from("rawload"),
        raw_path.as_os_str().to_os_string(),
        image_path.as_os_str().to_os_string(),
        OsString::from(spec.width.to_string()),
        OsString::from(spec.height.to_string()),
        OsString::from(spec.bands.to_string()),
        OsString::from("--format"),
        OsString::from(spec.format.cli_name()),
    ]);

    image_path
}

fn format_of(image_path: &Path) -> VipsBandFormat {
    let output = run_vipsheader([
        OsString::from("-f"),
        OsString::from("format"),
        image_path.as_os_str().to_os_string(),
    ]);

    if output.contains("VIPS_FORMAT_UCHAR") {
        VipsBandFormat::U8
    } else if output.contains("VIPS_FORMAT_CHAR") {
        VipsBandFormat::I8
    } else if output.contains("VIPS_FORMAT_USHORT") {
        VipsBandFormat::U16
    } else if output.contains("VIPS_FORMAT_SHORT") {
        VipsBandFormat::I16
    } else if output.contains("VIPS_FORMAT_UINT") {
        VipsBandFormat::U32
    } else if output.contains("VIPS_FORMAT_INT") {
        VipsBandFormat::I32
    } else if output.contains("VIPS_FORMAT_FLOAT") {
        VipsBandFormat::F32
    } else if output.contains("VIPS_FORMAT_DOUBLE") {
        VipsBandFormat::F64
    } else {
        panic!("unsupported libvips output format: {output}");
    }
}

fn generate_vips_golden_internal(op: &str, case: &str, vips_cmd: &[&str]) -> GeneratedGolden {
    assert!(
        vips_cmd.contains(&OUTPUT_PLACEHOLDER),
        "vips_cmd for op={op} case={case} must include {OUTPUT_PLACEHOLDER}"
    );

    let dir = case_dir(op, case);
    let output_image = dir.join("expected.v");
    let output_raw = dir.join("expected.raw");
    ensure_parent(&output_image);

    let args = vips_cmd
        .iter()
        .map(|arg| {
            if *arg == OUTPUT_PLACEHOLDER {
                output_image.as_os_str().to_os_string()
            } else {
                OsString::from(arg)
            }
        })
        .collect::<Vec<_>>();
    run_command(&VIPS_BIN, &args);

    let format = format_of(&output_image);
    run_vips([
        OsString::from("rawsave"),
        output_image.as_os_str().to_os_string(),
        output_raw.as_os_str().to_os_string(),
    ]);

    let bytes = fs::read(&output_raw)
        .unwrap_or_else(|e| panic!("failed to read generated golden {output_raw:?}: {e}"));

    GeneratedGolden { bytes, format }
}

pub fn generate_vips_golden(op: &str, case: &str, vips_cmd: &[&str]) -> Vec<u8> {
    generate_vips_golden_internal(op, case, vips_cmd).bytes
}

pub fn assert_golden_approx(actual: &[u8], expected: &[u8], max_diff: u8) {
    assert_eq!(actual.len(), expected.len(), "output length mismatch");

    for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (i16::from(got) - i16::from(want)).unsigned_abs() as u8;
        assert!(
            diff <= max_diff,
            "byte {idx}: viprs={got} libvips={want} diff={diff} > {max_diff}"
        );
    }
}

fn ordered_f32_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

fn ordered_f64_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000_0000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000_0000_0000
    }
}

fn assert_f32_ulp(actual: &[u8], expected: &[u8], max_ulp: u32) {
    assert_eq!(actual.len(), expected.len(), "output length mismatch");
    assert_eq!(
        actual.len() % 4,
        0,
        "f32 golden length must be a multiple of 4"
    );

    for (idx, (got, want)) in actual
        .chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .enumerate()
    {
        let got = f32::from_le_bytes(got.try_into().expect("f32 bytes"));
        let want = f32::from_le_bytes(want.try_into().expect("f32 bytes"));

        if got.is_nan() && want.is_nan() {
            continue;
        }

        let diff = ordered_f32_bits(got).abs_diff(ordered_f32_bits(want));
        assert!(
            diff <= max_ulp,
            "sample {idx}: viprs={got} libvips={want} ulp={diff} > {max_ulp}"
        );
    }
}

fn assert_f64_ulp(actual: &[u8], expected: &[u8], max_ulp: u64) {
    assert_eq!(actual.len(), expected.len(), "output length mismatch");
    assert_eq!(
        actual.len() % 8,
        0,
        "f64 golden length must be a multiple of 8"
    );

    for (idx, (got, want)) in actual
        .chunks_exact(8)
        .zip(expected.chunks_exact(8))
        .enumerate()
    {
        let got = f64::from_le_bytes(got.try_into().expect("f64 bytes"));
        let want = f64::from_le_bytes(want.try_into().expect("f64 bytes"));

        if got.is_nan() && want.is_nan() {
            continue;
        }

        let diff = ordered_f64_bits(got).abs_diff(ordered_f64_bits(want));
        assert!(
            diff <= max_ulp,
            "sample {idx}: viprs={got} libvips={want} ulp={diff} > {max_ulp}"
        );
    }
}

pub fn assert_golden(op: &str, case: &str, actual: &[u8]) {
    let path = fixture_path(op, case);
    let stored = fs::read(&path).unwrap_or_else(|e| panic!("failed to read fixture {path:?}: {e}"));
    assert_eq!(
        actual,
        stored.as_slice(),
        "golden mismatch for op={op} case={case}\n  fixture: {path:?}\n  actual length: {}\n  stored length: {}",
        actual.len(),
        stored.len()
    );
}

pub fn assert_golden_libvips(op: &str, case: &str, actual: &[u8], vips_cmd: &[&str]) {
    let expected = generate_vips_golden_internal(op, case, vips_cmd);

    if expected.format.is_float() {
        match expected.format {
            VipsBandFormat::F32 => assert_f32_ulp(actual, &expected.bytes, 1),
            VipsBandFormat::F64 => assert_f64_ulp(actual, &expected.bytes, 1),
            _ => unreachable!("only float formats reach this branch"),
        }
    } else {
        assert_golden_approx(actual, &expected.bytes, 0);
    }
}
