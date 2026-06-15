use std::path::{Path, PathBuf};
use std::process::Command;

use super::args::Arch;
use crate::common::repo_root;

pub fn run_simd_analysis(arch: Arch, op: &str) {
    println!("--- SIMD instruction analysis ---");
    println!("  Target arch: {:?}", arch);
    println!();

    let scope = AnalysisScope::for_op(op);
    let artifacts = if arch.is_native() {
        find_native_artifacts()
    } else {
        cross_compile_for_simd(arch)
    };

    let artifacts = match artifacts {
        Some(b) => b,
        None => return,
    };

    // Use llvm-objdump which can disassemble any architecture
    let objdump = find_objdump();

    let mut counts = Counts::default();
    for artifact in &artifacts {
        match analyze_artifact(artifact, &objdump, arch, &scope) {
            Ok(artifact_counts) => counts.merge(artifact_counts),
            Err(err) => {
                eprintln!("  objdump failed for {}: {err}", artifact.display());
                return;
            }
        }
    }

    if counts.scoped_symbols == 0 {
        eprintln!(
            "  No viprs symbols matched scope '{}'.",
            scope.symbol_scope_description()
        );
        return;
    }

    let simd_total = counts.simd_arm + counts.simd_x86;
    let simd_pct = if counts.eligible_total > 0 {
        simd_total as f64 / counts.eligible_total as f64 * 100.0
    } else {
        0.0
    };
    let fp_pct = if counts.eligible_total > 0 {
        counts.scalar_fp as f64 / counts.eligible_total as f64 * 100.0
    } else {
        0.0
    };

    println!("  Operation scope: {}", scope.symbol_scope_description());
    println!("  Artifacts scanned: {}", artifacts.len());
    println!("  Matching symbols:  {}", counts.scoped_symbols);
    println!(
        "  Eligible datapath instructions: {}",
        counts.eligible_total
    );
    println!("  SIMD instructions:             {simd_total} ({simd_pct:.2}%)");
    if counts.simd_arm > 0 {
        println!("    NEON/ASIMD/SVE: {}", counts.simd_arm);
    }
    if counts.simd_x86 > 0 {
        println!("    SSE/AVX:        {}", counts.simd_x86);
    }
    println!(
        "  Scalar datapath instructions: {} ({fp_pct:.2}%)",
        counts.scalar_fp
    );
    println!(
        "  Note: branches, address math, and other control instructions are excluded from the denominator."
    );
}

#[derive(Clone, Debug)]
struct AnalysisScope {
    op_symbol: String,
}

impl AnalysisScope {
    fn for_op(op: &str) -> Self {
        let normalized = op.replace('-', "_");
        Self {
            op_symbol: format!("::{normalized}::"),
        }
    }

    fn matches_symbol(&self, symbol: &str) -> bool {
        symbol.contains(&self.op_symbol)
    }

    fn symbol_scope_description(&self) -> String {
        format!("symbols containing '{}'", self.op_symbol)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    eligible_total: u64,
    simd_arm: u64,
    simd_x86: u64,
    scalar_fp: u64,
    scoped_symbols: u64,
}

impl Counts {
    fn merge(&mut self, other: Self) {
        self.eligible_total += other.eligible_total;
        self.simd_arm += other.simd_arm;
        self.simd_x86 += other.simd_x86;
        self.scalar_fp += other.scalar_fp;
        self.scoped_symbols += other.scoped_symbols;
    }
}

/// Find native viprs object files for SIMD analysis.
fn find_native_artifacts() -> Option<Vec<PathBuf>> {
    let repo = repo_root();
    let release_deps = repo.join("target/release/deps");
    find_viprs_objects(&release_deps).or_else(|| {
        eprintln!("  No viprs release object files found. Run `cargo build --release` first.");
        None
    })
}

/// Cross-compile for a non-native architecture and return the object files.
fn cross_compile_for_simd(arch: Arch) -> Option<Vec<PathBuf>> {
    let triple = arch.rust_target_triple();
    println!("  Cross-compiling for {triple}...");

    // Check target is installed
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(triple))
        .unwrap_or(false);

    if !installed {
        eprintln!("  Target '{triple}' not installed. Run:");
        eprintln!("    rustup target add {triple}");
        eprintln!();
        eprintln!("  For x86_64-unknown-linux-gnu you also need a cross-linker.");
        eprintln!("  Alternatively, use --arch with the Docker hw metrics which");
        eprintln!("  builds inside a container for the target arch:");
        eprintln!("    cargo xtask perf <input> <op> --metrics hw --arch amd64");
        return None;
    }

    let repo = repo_root();
    let status = Command::new("cargo")
        .current_dir(&repo)
        .args(["build", "--lib", "--release", "--target", triple])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(_) => {
            eprintln!("  Cross-compilation failed. You may need a linker for {triple}.");
            eprintln!("  For static analysis only, try building as rlib (no link step):");
            eprintln!("    cargo rustc --lib --release --target {triple} -- --emit=obj");
            // Try obj-only fallback
            let obj_status = Command::new("cargo")
                .current_dir(&repo)
                .args([
                    "rustc",
                    "--lib",
                    "--release",
                    "--target",
                    triple,
                    "--",
                    "--emit=obj",
                ])
                .status();
            if obj_status.map(|s| s.success()).unwrap_or(false) {
                // Find the .o file
                let obj_dir = repo.join(format!("target/{triple}/release/deps"));
                if let Some(objects) = find_viprs_objects(&obj_dir) {
                    println!("  Using {} object file(s).", objects.len());
                    return Some(objects);
                }
            }
            eprintln!("  Could not produce any binary for {triple}.");
            return None;
        }
        Err(e) => {
            eprintln!("  cargo build failed: {e}");
            return None;
        }
    }

    let deps_dir = repo.join(format!("target/{triple}/release/deps"));
    if let Some(objects) = find_viprs_objects(&deps_dir) {
        Some(objects)
    } else {
        eprintln!("  Built successfully but cannot find viprs object files for {triple}.");
        None
    }
}

/// Find the best objdump tool — prefer llvm-objdump (handles all architectures).
fn find_objdump() -> String {
    // llvm-objdump from homebrew (macOS)
    let llvm_brew = "/opt/homebrew/opt/llvm/bin/llvm-objdump";
    if Path::new(llvm_brew).exists() {
        return llvm_brew.to_string();
    }
    // llvm-objdump in PATH
    if Command::new("llvm-objdump")
        .arg("--version")
        .output()
        .is_ok()
    {
        return "llvm-objdump".to_string();
    }
    // Fallback to system objdump (only works for native arch)
    "objdump".to_string()
}

fn find_viprs_objects(dir: &Path) -> Option<Vec<PathBuf>> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut objects: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "o"))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("viprs"))
        })
        .collect();
    objects.sort();
    (!objects.is_empty()).then_some(objects)
}

fn analyze_artifact(
    artifact: &Path,
    objdump: &str,
    arch: Arch,
    scope: &AnalysisScope,
) -> Result<Counts, String> {
    let output = Command::new(objdump)
        .args(["-d", "--demangle", "--no-show-raw-insn"])
        .arg(artifact)
        .output()
        .map_err(|err| err.to_string())?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    let output = String::from_utf8_lossy(&output.stdout);
    let mut counts = Counts::default();
    let mut in_scope = false;

    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(symbol) = parse_symbol_header(trimmed) {
            in_scope = scope.matches_symbol(symbol);
            if in_scope {
                counts.scoped_symbols += 1;
            }
            continue;
        }

        if !in_scope {
            continue;
        }

        let Some((mnemonic, operands)) = parse_instruction(trimmed) else {
            continue;
        };

        match classify_instruction(arch, mnemonic, operands) {
            InstructionClass::SimdArm => {
                counts.eligible_total += 1;
                counts.simd_arm += 1;
            }
            InstructionClass::SimdX86 => {
                counts.eligible_total += 1;
                counts.simd_x86 += 1;
            }
            InstructionClass::ScalarDatapath => {
                counts.eligible_total += 1;
                counts.scalar_fp += 1;
            }
            InstructionClass::ControlOrAddressMath => {}
        }
    }

    Ok(counts)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InstructionClass {
    SimdArm,
    SimdX86,
    ScalarDatapath,
    ControlOrAddressMath,
}

fn parse_symbol_header(line: &str) -> Option<&str> {
    let start = line.find('<')?;
    let end = line.rfind(">:")?;
    (start < end).then_some(&line[start + 1..end])
}

fn parse_instruction(line: &str) -> Option<(&str, &str)> {
    let (address, rest) = line.split_once(':')?;
    if address.is_empty() || !address.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }

    let rest = rest.trim();
    if rest.is_empty() {
        return None;
    }

    let mut parts = rest.splitn(2, char::is_whitespace);
    let mnemonic = parts.next()?;
    let operands = parts.next().unwrap_or("").trim();
    Some((mnemonic, operands))
}

fn classify_instruction(arch: Arch, mnemonic: &str, operands: &str) -> InstructionClass {
    match arch {
        Arch::Arm64 => classify_arm_instruction(mnemonic, operands),
        Arch::Amd64 => classify_x86_instruction(mnemonic, operands),
    }
}

fn classify_arm_instruction(mnemonic: &str, operands: &str) -> InstructionClass {
    if uses_arm_vector_registers(operands)
        || mnemonic.starts_with("ld1")
        || mnemonic.starts_with("ld2")
        || mnemonic.starts_with("ld3")
        || mnemonic.starts_with("ld4")
        || mnemonic.starts_with("st1")
        || mnemonic.starts_with("st2")
        || mnemonic.starts_with("st3")
        || mnemonic.starts_with("st4")
        || mnemonic.starts_with("zip")
        || mnemonic.starts_with("uzp")
        || mnemonic.starts_with("trn")
        || mnemonic.starts_with("tbl")
        || mnemonic.starts_with("tbx")
        || mnemonic.starts_with("ext")
        || mnemonic.starts_with("ptrue")
        || mnemonic.starts_with("whilelt")
    {
        return InstructionClass::SimdArm;
    }

    if uses_arm_scalar_fp_registers(operands)
        || mnemonic.starts_with("ldrb")
        || mnemonic.starts_with("strb")
        || mnemonic.starts_with("ldrh")
        || mnemonic.starts_with("strh")
    {
        return InstructionClass::ScalarDatapath;
    }

    InstructionClass::ControlOrAddressMath
}

fn classify_x86_instruction(mnemonic: &str, operands: &str) -> InstructionClass {
    if uses_x86_vector_registers(operands) || is_x86_simd_mnemonic(mnemonic) {
        return InstructionClass::SimdX86;
    }

    if is_x86_scalar_fp(mnemonic, operands) {
        return InstructionClass::ScalarDatapath;
    }

    InstructionClass::ControlOrAddressMath
}

fn uses_arm_vector_registers(operands: &str) -> bool {
    operand_has_prefixed_register(operands, &['v', 'q', 'p'])
}

fn uses_arm_scalar_fp_registers(operands: &str) -> bool {
    operand_has_prefixed_register(operands, &['s', 'd']) && !uses_arm_vector_registers(operands)
}

fn uses_x86_vector_registers(operands: &str) -> bool {
    operands.contains("xmm")
        || operands.contains("ymm")
        || operands.contains("zmm")
        || operands.contains("mm")
}

fn operand_has_prefixed_register(operands: &str, prefixes: &[char]) -> bool {
    operands
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|token| !token.is_empty())
        .any(|token| {
            let mut chars = token.chars();
            let Some(prefix) = chars.next() else {
                return false;
            };
            prefixes.contains(&prefix) && chars.all(|c| c.is_ascii_digit())
        })
}

fn is_x86_simd_mnemonic(mnemonic: &str) -> bool {
    mnemonic.starts_with("vmov")
        || mnemonic.starts_with("vadd")
        || mnemonic.starts_with("vsub")
        || mnemonic.starts_with("vmul")
        || mnemonic.starts_with("vdiv")
        || mnemonic.starts_with("vfma")
        || mnemonic.starts_with("vfnma")
        || mnemonic.starts_with("vpack")
        || mnemonic.starts_with("vpunpck")
        || mnemonic.starts_with("vshuf")
        || mnemonic.starts_with("vperm")
        || mnemonic.starts_with("vcvt")
        || mnemonic.starts_with("vmin")
        || mnemonic.starts_with("vmax")
        || mnemonic.starts_with("vpand")
        || mnemonic.starts_with("vpor")
        || mnemonic.starts_with("vpxor")
        || mnemonic.starts_with("vpcmp")
        || mnemonic.starts_with("vbroadcast")
        || mnemonic.starts_with("vinsert")
        || mnemonic.starts_with("vextract")
        || mnemonic.starts_with("movaps")
        || mnemonic.starts_with("movups")
        || mnemonic.starts_with("movdqa")
        || mnemonic.starts_with("movdqu")
        || mnemonic.starts_with("addps")
        || mnemonic.starts_with("mulps")
        || mnemonic.starts_with("addpd")
        || mnemonic.starts_with("mulpd")
        || mnemonic.starts_with("paddb")
        || mnemonic.starts_with("paddw")
        || mnemonic.starts_with("paddd")
        || mnemonic.starts_with("pmull")
        || mnemonic.starts_with("pshufd")
        || mnemonic.starts_with("pshufb")
        || mnemonic.starts_with("punpck")
}

fn is_x86_scalar_fp(mnemonic: &str, operands: &str) -> bool {
    (operands.contains("xmm") && (mnemonic.ends_with("ss") || mnemonic.ends_with("sd")))
        || mnemonic.starts_with("addss")
        || mnemonic.starts_with("mulss")
        || mnemonic.starts_with("addsd")
        || mnemonic.starts_with("mulsd")
        || mnemonic.starts_with("divss")
        || mnemonic.starts_with("divsd")
}

#[cfg(test)]
mod tests {
    use super::{
        InstructionClass, classify_arm_instruction, operand_has_prefixed_register,
        parse_instruction, parse_symbol_header,
    };

    #[test]
    fn parses_symbol_headers() {
        let line = "0000000000062d2c <viprs::domain::ops::arithmetic::linear::neon_linear_u8_f32::hda095076c6f6a1f6>:";
        assert_eq!(
            parse_symbol_header(line),
            Some("viprs::domain::ops::arithmetic::linear::neon_linear_u8_f32::hda095076c6f6a1f6")
        );
    }

    #[test]
    fn parses_instruction_lines() {
        let line = "62d80:\t\tfmul.4s\tv7, v7, v0[0]";
        assert_eq!(parse_instruction(line), Some(("fmul.4s", "v7, v7, v0[0]")));
    }

    #[test]
    fn detects_arm_vector_registers_in_operands() {
        assert!(operand_has_prefixed_register(
            "v7, v7, v0[0]",
            &['v', 'q', 'p']
        ));
        assert!(operand_has_prefixed_register(
            "q5, [x11], #0x10",
            &['v', 'q', 'p']
        ));
        assert!(!operand_has_prefixed_register(
            "s4, s0, s4",
            &['v', 'q', 'p']
        ));
    }

    #[test]
    fn classifies_arm_vector_and_scalar_datapath_instructions() {
        assert_eq!(
            classify_arm_instruction("fmul.4s", "v7, v7, v0[0]"),
            InstructionClass::SimdArm
        );
        assert_eq!(
            classify_arm_instruction("ldr", "q5, [x11], #0x10"),
            InstructionClass::SimdArm
        );
        assert_eq!(
            classify_arm_instruction("fadd", "s4, s1, s4"),
            InstructionClass::ScalarDatapath
        );
        assert_eq!(
            classify_arm_instruction("add", "x8, x0, x10"),
            InstructionClass::ControlOrAddressMath
        );
    }
}
