//! samply-based CPU flame graph profiling for viprs and libvips.
//!
//! Requires `samply` to be installed: `cargo install samply`
//! On macOS it uses the system DTrace/Instruments backend.
//! On Linux it uses perf.
//!
//! Produces two JSON profiles (one per binary) and can optionally print an
//! AI-readable top-functions summary extracted from the Firefox Profiler JSON.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

use serde::Deserialize;

use crate::common::repo_root;
use crate::profile::baseline_op_name;

const AI_SUMMARY_LIMIT: usize = 20;
const MIN_SELF_PERCENT: f64 = 1.0;
const MIN_FUNCTION_NAME_LEN: usize = 4;

pub fn run_samply(input: &Path, op: &str, op_args: &[String], iterations: usize, ai_output: bool) {
    if !ai_output {
        println!("--- CPU flame graph via samply ---");
        println!("  Profiles both viprs and libvips separately.");
        println!("  View at: https://profiler.firefox.com (load both files)");
        println!();
    }

    if !samply_available() {
        eprintln!("  samply not found. Install with:");
        eprintln!("    cargo install samply");
        return;
    }

    let repo = repo_root();
    let threads = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(4);
    let libvips_runner = repo.join("tools/bench-vs-libvips/libvips-runner");
    let viprs_runner =
        std::env::current_exe().unwrap_or_else(|_| repo.join("target/release/xtask"));

    // Build libvips-runner if needed
    if !libvips_runner.exists() {
        if !ai_output {
            println!("  Building libvips-runner...");
        }
        let status = Command::new("make")
            .current_dir(repo.join("tools/bench-vs-libvips"))
            .arg("libvips-runner")
            .status();
        match status {
            Ok(s) if s.success() => {
                if !ai_output {
                    println!("  ✓ libvips-runner built");
                }
            }
            _ => {
                eprintln!("  Failed to build libvips-runner.");
                eprintln!("  Run: make -C tools/bench-vs-libvips libvips-runner");
                return;
            }
        }
    }

    let output_dir = repo.join("tmp");
    if let Err(error) = std::fs::create_dir_all(&output_dir) {
        eprintln!(
            "  Failed to create profile output dir {}: {error}",
            output_dir.display()
        );
        return;
    }
    let viprs_out = output_dir.join(format!("viprs_profile_{op}.json"));
    let libvips_out = output_dir.join(format!("libvips_profile_{op}.json"));
    let viprs_out_str = viprs_out.to_string_lossy().into_owned();
    let libvips_out_str = libvips_out.to_string_lossy().into_owned();

    // Profile viprs
    if !ai_output {
        println!("  [1/2] Profiling viprs...");
    }
    let mut viprs_cmd = build_samply_cmd(&viprs_out_str);
    viprs_cmd.arg(&viprs_runner);
    viprs_cmd.arg("bench");
    viprs_cmd.arg(input);
    viprs_cmd.arg(op);
    for a in op_args {
        viprs_cmd.arg(a);
    }
    append_viprs_profile_only_args(&mut viprs_cmd, iterations, threads);

    match viprs_cmd.status() {
        Ok(s) if s.success() => {
            if !ai_output {
                println!("  ✓ viprs profile saved: {}", viprs_out.display());
            }
        }
        Ok(_) => eprintln!("  ✗ viprs samply run failed"),
        Err(e) => eprintln!("  ✗ Failed to run samply: {e}"),
    }

    // Profile libvips
    if !ai_output {
        println!("  [2/2] Profiling libvips...");
    }
    let mut libvips_cmd = build_samply_cmd(&libvips_out_str);
    libvips_cmd.arg(&libvips_runner);
    libvips_cmd.arg(input);
    libvips_cmd.arg(baseline_op_name(op));
    for a in op_args {
        libvips_cmd.arg(a);
    }
    libvips_cmd.args(["--iterations", &iterations.to_string()]);
    libvips_cmd.args(["--threads", &threads.to_string(), "--quiet"]);

    match libvips_cmd.status() {
        Ok(s) if s.success() => {
            if !ai_output {
                println!("  ✓ libvips profile saved: {}", libvips_out.display());
            }
        }
        Ok(_) => eprintln!("  ✗ libvips samply run failed"),
        Err(e) => eprintln!("  ✗ Failed to run samply for libvips: {e}"),
    }

    if ai_output {
        print_ai_summary("viprs", &viprs_out);
        if libvips_out.exists() {
            print_ai_summary("libvips", &libvips_out);
        }
        return;
    }

    println!();
    println!("  === How to view profiles ===");
    println!();
    println!("  Use `samply load` (NOT drag-and-drop into Firefox Profiler):");
    println!("  The .json files do NOT contain symbols — samply load starts a local");
    println!("  symbolication server that Firefox Profiler queries in real time.");
    println!();
    println!("    samply load {}", viprs_out.display());
    println!("    samply load {}", libvips_out.display());
    println!();
    println!("  Each command opens Firefox Profiler with full function names.");
    println!("  Run them in separate terminals to compare side-by-side.");
    println!();
    println!("  What to look for:");
    println!("    - Functions that appear wide in viprs but thin/absent in libvips");
    println!("      → those are the bottlenecks to investigate");
    println!("    - Missing SIMD intrinsics in viprs where libvips has them");
}

fn build_samply_cmd(output_file: &str) -> Command {
    let mut cmd = Command::new("samply");
    cmd.args([
        "record",
        "--save-only",
        "--unstable-presymbolicate",
        "--output",
        output_file,
    ]);
    cmd
}

fn append_viprs_profile_only_args(cmd: &mut Command, iterations: usize, threads: usize) {
    cmd.args(["--iterations", &iterations.to_string()]);
    cmd.args(["--threads", &threads.to_string()]);
    cmd.arg("--profile-only");
}

fn samply_available() -> bool {
    Command::new("samply")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn print_ai_summary(label: &str, path: &Path) {
    match summarize_profile(path) {
        Ok(entries) if entries.is_empty() => {
            eprintln!(
                "  [{label}] No resolved symbols found in {}",
                path.display()
            );
            eprintln!("  The profile likely contains only raw hex addresses.");
            eprintln!("  This happens when --unstable-presymbolicate fails (e.g. missing dSYM,");
            eprintln!("  lack of debug info, or macOS SIP restrictions).");
            eprintln!();
            eprintln!("  Fix: use `samply load` to view the profile with live symbolication:");
            eprintln!("    samply load {}", path.display());
        }
        Ok(entries) => {
            println!("--- {label} top functions (self-time) ---");
            println!("  #  self%  function");
            for (index, entry) in entries.iter().enumerate() {
                println!("{:>3}  {:>5.1}%  {}", index + 1, entry.percent, entry.name);
            }
        }
        Err(error) => eprintln!("  Failed to summarize {}: {error}", path.display()),
    }
}

fn summarize_profile(path: &Path) -> Result<Vec<FunctionSelfTime>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let profile: SamplyProfile = serde_json::from_str(&contents)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;

    // Load the `.syms.json` sidecar written by `samply record --unstable-presymbolicate`.
    // samply derives the sidecar path by replacing `.json` with `.syms.json`.
    let syms_path = path.with_extension("syms.json");
    let resolver = if syms_path.exists() {
        match fs::read_to_string(&syms_path)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<SymsFile>(&s).map_err(|e| e.to_string()))
        {
            Ok(syms) => Some(SymbolResolver::build(&syms)),
            Err(err) => {
                eprintln!(
                    "  Warning: failed to load symbol sidecar {}: {err}",
                    syms_path.display()
                );
                None
            }
        }
    } else {
        None
    };

    Ok(extract_top_functions(&profile, resolver.as_ref()))
}

fn extract_top_functions(
    profile: &SamplyProfile,
    resolver: Option<&SymbolResolver>,
) -> Vec<FunctionSelfTime> {
    let mut function_weights: HashMap<String, f64> = HashMap::new();
    let mut total_weight = 0.0;

    for thread in &profile.threads {
        let strings = thread.strings();
        if strings.is_empty() {
            continue;
        }

        for (sample_index, stack_index) in thread.samples.stack.iter().enumerate() {
            let Some(stack_index) = *stack_index else {
                continue;
            };

            let weight = thread
                .samples
                .weight
                .get(sample_index)
                .copied()
                .unwrap_or(1.0);
            if weight <= 0.0 {
                continue;
            }

            let Some(frame_index) = thread
                .stack_table
                .frame
                .get(stack_index)
                .and_then(|value| *value)
            else {
                continue;
            };
            let Some(func_index) = thread
                .frame_table
                .func
                .get(frame_index)
                .and_then(|value| *value)
            else {
                continue;
            };
            let Some(name_index) = thread
                .func_table
                .name
                .get(func_index)
                .and_then(|value| *value)
            else {
                continue;
            };
            let Some(raw_name) = strings.get(name_index).map(|name| name.trim()) else {
                continue;
            };
            if raw_name.is_empty() {
                continue;
            }

            // Resolve hex addresses via the .syms.json sidecar when available.
            let function_name = if is_hex_address(raw_name) {
                resolver
                    .and_then(|r| {
                        resolve_hex_name(
                            raw_name,
                            func_index,
                            &thread.func_table,
                            &thread.resource_table,
                            &profile.libs,
                            r,
                        )
                    })
                    .unwrap_or_else(|| raw_name.to_owned())
            } else {
                raw_name.to_owned()
            };

            *function_weights.entry(function_name).or_insert(0.0) += weight;
            total_weight += weight;
        }
    }

    if total_weight <= 0.0 {
        return Vec::new();
    }

    let mut entries = function_weights
        .into_iter()
        .map(|(name, weight)| FunctionSelfTime {
            percent: (weight * 100.0) / total_weight,
            weight,
            name,
        })
        .filter(|entry| should_include_function(&entry.name, entry.percent))
        .collect::<Vec<_>>();

    entries.sort_by(|left, right| {
        right
            .weight
            .total_cmp(&left.weight)
            .then_with(|| left.name.cmp(&right.name))
    });
    entries.truncate(AI_SUMMARY_LIMIT);
    entries
}

/// Resolves a hex-address function name to a real symbol name using the syms sidecar.
///
/// The resolution chain is:
///   funcTable.resource[func_index] → resourceTable.lib[resource_index] → libs[lib_index].code_id
///   → SymbolResolver::resolve(code_id, rva)
fn resolve_hex_name(
    raw_name: &str,
    func_index: usize,
    func_table: &FuncTable,
    resource_table: &ResourceTable,
    libs: &[SamplyLib],
    resolver: &SymbolResolver,
) -> Option<String> {
    let resource_index = func_table.resource.get(func_index).copied()?;
    if resource_index < 0 {
        return None;
    }
    let lib_index = resource_table.lib.get(resource_index as usize).copied()?;
    if lib_index < 0 {
        return None;
    }
    let lib = libs.get(lib_index as usize)?;
    let hex = raw_name
        .strip_prefix("0x")
        .or_else(|| raw_name.strip_prefix("0X"))?;
    let rva = u64::from_str_radix(hex, 16).ok()?;
    resolver.resolve(&lib.code_id, rva).map(ToOwned::to_owned)
}

fn should_include_function(function_name: &str, self_percent: f64) -> bool {
    self_percent >= MIN_SELF_PERCENT
        && function_name.chars().count() >= MIN_FUNCTION_NAME_LEN
        && !is_system_function(function_name)
        && !is_hex_address(function_name)
}

/// Returns true when the name is an unresolved address like `0x00007fff5fbff960`.
/// samply emits these when presymbolication fails (missing dSYM / SIP / no debug info).
fn is_hex_address(name: &str) -> bool {
    let s = name.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit() || c == '_')
    } else {
        false
    }
}

fn is_system_function(function_name: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "__",
        "_dispatch",
        "_pthread",
        "clone",
        "dyld",
        "libc_",
        "libdyld",
        "libsystem_",
        "mach_",
        "pthread_",
        "start",
        "thread_start",
    ];

    PREFIXES
        .iter()
        .any(|prefix| function_name.starts_with(prefix))
        || function_name.starts_with("std::sys::")
}

#[derive(Debug, PartialEq)]
struct FunctionSelfTime {
    percent: f64,
    weight: f64,
    name: String,
}

// ── Firefox Profiler JSON structs ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct SamplyProfile {
    /// Library descriptors — one per shared object / binary in the process.
    #[serde(default)]
    libs: Vec<SamplyLib>,
    #[serde(default)]
    threads: Vec<SamplyThread>,
}

/// Metadata for one library / binary loaded in the profiled process.
#[derive(Debug, Deserialize)]
struct SamplyLib {
    /// Uppercase hex build-ID used as key into the `.syms.json` sidecar.
    #[serde(default, rename = "codeId")]
    code_id: String,
}

#[derive(Debug, Deserialize)]
struct SamplyThread {
    samples: SampleTable,
    #[serde(rename = "stackTable")]
    stack_table: StackTable,
    #[serde(rename = "frameTable")]
    frame_table: FrameTable,
    #[serde(rename = "funcTable")]
    func_table: FuncTable,
    /// Maps resource index → library index into `SamplyProfile::libs`.
    /// Firefox Profiler uses -1 as the null sentinel for integer columns.
    #[serde(default, rename = "resourceTable")]
    resource_table: ResourceTable,
    #[serde(default, rename = "stringArray")]
    string_array: Vec<String>,
    #[serde(default, rename = "stringTable")]
    string_table: Vec<String>,
}

impl SamplyThread {
    fn strings(&self) -> &[String] {
        if self.string_array.is_empty() {
            &self.string_table
        } else {
            &self.string_array
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct ResourceTable {
    /// `lib[r]` = index into `SamplyProfile::libs`; -1 means no library.
    #[serde(default)]
    lib: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct SampleTable {
    #[serde(default)]
    stack: Vec<Option<usize>>,
    #[serde(default)]
    weight: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct StackTable {
    #[serde(default)]
    frame: Vec<Option<usize>>,
}

#[derive(Debug, Deserialize)]
struct FrameTable {
    #[serde(default)]
    func: Vec<Option<usize>>,
}

#[derive(Debug, Deserialize)]
struct FuncTable {
    #[serde(default)]
    name: Vec<Option<usize>>,
    /// `resource[f]` = resource index; -1 means unknown origin.
    #[serde(default)]
    resource: Vec<i64>,
}

// ── `.syms.json` sidecar (written by `samply record --unstable-presymbolicate`) ──

#[derive(Debug, Deserialize)]
struct SymsFile {
    /// Interned string pool shared by all symbol entries.
    #[serde(default)]
    string_table: Vec<String>,
    #[serde(default)]
    data: Vec<SymsEntry>,
}

/// Symbol table for one library / binary.
#[derive(Debug, Deserialize)]
struct SymsEntry {
    /// Build-ID, uppercase hex — matches `SamplyLib::code_id`.
    #[serde(default)]
    code_id: String,
    #[serde(default)]
    symbol_table: Vec<SymsSymbol>,
}

#[derive(Debug, Deserialize)]
struct SymsSymbol {
    /// Relative virtual address (offset from image load address).
    rva: u64,
    /// Index into `SymsFile::string_table`.
    symbol: usize,
}

/// Resolves RVA → function name using the data from a `.syms.json` sidecar.
struct SymbolResolver {
    /// code_id (uppercase) → RVA-sorted list of (rva, function_name).
    tables: HashMap<String, Vec<(u64, String)>>,
}

impl SymbolResolver {
    fn build(syms: &SymsFile) -> Self {
        let mut tables = HashMap::with_capacity(syms.data.len());
        for entry in &syms.data {
            let code_id = entry.code_id.to_ascii_uppercase();
            let mut tbl: Vec<(u64, String)> = entry
                .symbol_table
                .iter()
                .filter_map(|s| {
                    syms.string_table
                        .get(s.symbol)
                        .map(|name| (s.rva, name.clone()))
                })
                .collect();
            tbl.sort_by_key(|(rva, _)| *rva);
            tables.insert(code_id, tbl);
        }
        Self { tables }
    }

    /// Returns the name of the symbol whose RVA is the largest one ≤ `rva`.
    fn resolve(&self, code_id: &str, rva: u64) -> Option<&str> {
        let tbl = self.tables.get(&code_id.to_ascii_uppercase())?;
        // partition_point gives the first index where rva > target
        let idx = tbl.partition_point(|(r, _)| *r <= rva);
        idx.checked_sub(1).map(|i| tbl[i].1.as_str())
    }
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use super::{
        FuncTable, ResourceTable, SamplyLib, SamplyProfile, SymbolResolver, SymsEntry, SymsFile,
        SymsSymbol, append_viprs_profile_only_args, extract_top_functions, is_hex_address,
        resolve_hex_name,
    };

    #[test]
    fn profile_only_args_do_not_include_stale_cache_flag() {
        let mut cmd = Command::new("echo");
        append_viprs_profile_only_args(&mut cmd, 7, 3);

        let args = cmd
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert!(args.contains(&"--profile-only".to_owned()));
        assert!(!args.contains(&"--no-cache-comparison".to_owned()));
    }

    #[test]
    fn extracts_top_functions_from_string_array_profiles() {
        // Thread 1: stacks [0,1,1,2], weights [3,2,2,1]
        //   name 0 "viprs::hot"                     w=3
        //   name 1 "core::ptr::copy_nonoverlapping"  w=2+2=4
        //   name 2 "tin" (3 chars, filtered)         w=1 — still in total
        // Thread 2: stacks [0], weights [2]
        //   name 1 "core::ptr::copy_nonoverlapping"  w=2
        // total_weight = 10
        // "core::ptr::copy_nonoverlapping" = 6/10 = 60%
        // "viprs::hot"                     = 3/10 = 30%
        let profile: SamplyProfile = serde_json::from_str(
            r#"{
                "threads": [
                    {
                        "samples": { "stack": [0, 1, 1, 2], "weight": [3, 2, 2, 1] },
                        "stackTable": { "frame": [0, 1, 2] },
                        "frameTable": { "func": [0, 1, 2] },
                        "funcTable": { "name": [0, 1, 2] },
                        "stringArray": ["viprs::hot", "core::ptr::copy_nonoverlapping", "tin"]
                    },
                    {
                        "samples": { "stack": [0], "weight": [2] },
                        "stackTable": { "frame": [0] },
                        "frameTable": { "func": [0] },
                        "funcTable": { "name": [1] },
                        "stringArray": ["unused", "core::ptr::copy_nonoverlapping"]
                    }
                ]
            }"#,
        )
        .expect("profile should deserialize");

        let entries = extract_top_functions(&profile, None);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "core::ptr::copy_nonoverlapping");
        assert!((entries[0].percent - 60.0).abs() < 0.001);
        assert_eq!(entries[1].name, "viprs::hot");
        assert!((entries[1].percent - 30.0).abs() < 0.001);
    }

    #[test]
    fn extracts_top_functions_from_legacy_string_table_without_weights() {
        let profile: SamplyProfile = serde_json::from_str(
            r#"{
                "threads": [
                    {
                        "samples": { "stack": [0, 1, 1, 2] },
                        "stackTable": { "frame": [0, 1, 2] },
                        "frameTable": { "func": [0, 1, 2] },
                        "funcTable": { "name": [0, 1, 2] },
                        "stringTable": ["viprs::leaf", "abcd", "__kernel_rt_sigreturn"]
                    }
                ]
            }"#,
        )
        .expect("profile should deserialize");

        let entries = extract_top_functions(&profile, None);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "abcd");
        assert_eq!(entries[1].name, "viprs::leaf");
    }

    #[test]
    fn is_hex_address_detects_unresolved_samply_entries() {
        assert!(is_hex_address("0x00007fff5fbff960"));
        assert!(is_hex_address("0x1a2b3c4d"));
        assert!(is_hex_address("0XDEADBEEF"));
        assert!(!is_hex_address("viprs::domain::ops::resize::run"));
        assert!(!is_hex_address("core::ptr::copy_nonoverlapping"));
        assert!(!is_hex_address("abcd"));
        assert!(!is_hex_address("0x")); // prefix with no digits
    }

    #[test]
    fn hex_addresses_without_resolver_are_filtered_from_output() {
        // Without a syms sidecar, hex addresses stay as-is and are filtered.
        let profile: SamplyProfile = serde_json::from_str(
            r#"{
                "threads": [
                    {
                        "samples": { "stack": [0, 1, 2], "weight": [5, 5, 5] },
                        "stackTable": { "frame": [0, 1, 2] },
                        "frameTable": { "func": [0, 1, 2] },
                        "funcTable": { "name": [0, 1, 2] },
                        "stringArray": ["0x00007fff5fbff960", "0x1a2b3c4d", "viprs::hot"]
                    }
                ]
            }"#,
        )
        .expect("profile should deserialize");

        let entries = extract_top_functions(&profile, None);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "viprs::hot");
    }

    #[test]
    fn hex_addresses_resolve_to_symbol_names_via_syms_sidecar() {
        // Simulates a real samply profile where funcTable.name contains hex RVAs,
        // and the .syms.json sidecar has the symbol table to resolve them.
        //
        // libs[0].codeId = "AABBCCDD"
        // funcTable.resource[0] = 0 (resource 0)
        // resourceTable.lib[0] = 0 (lib 0)
        // name[0] = "0x1000" → rva 0x1000 → resolves to "viprs::pixel_path"
        let profile: SamplyProfile = serde_json::from_str(
            r#"{
                "libs": [{ "codeId": "AABBCCDD" }],
                "threads": [
                    {
                        "samples": { "stack": [0], "weight": [10] },
                        "stackTable": { "frame": [0] },
                        "frameTable": { "func": [0] },
                        "funcTable": { "name": [0], "resource": [0] },
                        "resourceTable": { "lib": [0] },
                        "stringArray": ["0x1000"]
                    }
                ]
            }"#,
        )
        .expect("profile should deserialize");

        let syms = SymsFile {
            string_table: vec!["viprs::pixel_path".to_owned()],
            data: vec![SymsEntry {
                code_id: "AABBCCDD".to_owned(),
                symbol_table: vec![SymsSymbol {
                    rva: 0x1000,
                    symbol: 0,
                }],
            }],
        };
        let resolver = SymbolResolver::build(&syms);

        let entries = extract_top_functions(&profile, Some(&resolver));

        assert_eq!(entries.len(), 1, "resolved symbol must appear in output");
        assert_eq!(entries[0].name, "viprs::pixel_path");
    }

    #[test]
    fn resolver_picks_largest_rva_not_exceeding_target() {
        // Two symbols at rva 0x1000 and 0x2000.
        // Address 0x1500 should resolve to the symbol at 0x1000.
        let syms = SymsFile {
            string_table: vec!["func_a".to_owned(), "func_b".to_owned()],
            data: vec![SymsEntry {
                code_id: "DEADBEEF".to_owned(),
                symbol_table: vec![
                    SymsSymbol {
                        rva: 0x1000,
                        symbol: 0,
                    },
                    SymsSymbol {
                        rva: 0x2000,
                        symbol: 1,
                    },
                ],
            }],
        };
        let resolver = SymbolResolver::build(&syms);

        assert_eq!(resolver.resolve("DEADBEEF", 0x1000), Some("func_a"));
        assert_eq!(resolver.resolve("DEADBEEF", 0x1500), Some("func_a")); // falls inside func_a
        assert_eq!(resolver.resolve("DEADBEEF", 0x2000), Some("func_b"));
        assert_eq!(resolver.resolve("DEADBEEF", 0x0FFF), None); // before first symbol
        assert_eq!(resolver.resolve("DEADBEEF", 0x9999), Some("func_b")); // after last
        // code_id comparison is case-insensitive
        assert_eq!(resolver.resolve("deadbeef", 0x1000), Some("func_a"));
    }

    #[test]
    fn resolve_hex_name_returns_none_for_unknown_resource() {
        let libs = vec![SamplyLib {
            code_id: "AABB".to_owned(),
        }];
        let func_table = FuncTable {
            name: vec![],
            resource: vec![-1],
        }; // -1 = no resource
        let resource_table = ResourceTable { lib: vec![] };
        let syms = SymsFile {
            string_table: vec![],
            data: vec![],
        };
        let resolver = SymbolResolver::build(&syms);

        let result = resolve_hex_name("0x1000", 0, &func_table, &resource_table, &libs, &resolver);
        assert!(result.is_none());
    }
}
