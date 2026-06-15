#!/usr/bin/env bash
# tools/check-parity.sh — Viprs ↔ libvips parity checker
#
# Usage:
#   tools/check-parity.sh              # print full parity report to stdout
#   tools/check-parity.sh --diff       # only print ❌ missing and 🔶 stub ops
#   tools/check-parity.sh --summary    # print only the per-module summary table
#
# The script compares:
#   .libvips_repo/libvips/<module>/*.c  — libvips user-facing ops (internal helpers filtered)
#   src/domain/ops/<module>/            — viprs op files (*.rs)
#   src/adapters/codecs/                — viprs codec files
#
# Exit code 0 = full parity (no ❌ or 🔶). Exit code 1 = gaps found.
# Exit code 2 = .libvips_repo not found.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# .libvips_repo is gitignored and lives in the main worktree.
# Resolve the git common directory to find it regardless of which worktree we're in.
GIT_COMMON_DIR="$(git -C "$ROOT" rev-parse --git-common-dir 2>/dev/null || true)"
MAIN_WORKTREE="$(cd "$GIT_COMMON_DIR/.." && pwd 2>/dev/null || echo "$ROOT")"
LIBVIPS_ROOT="$MAIN_WORKTREE/.libvips_repo/libvips"

# Fallback: try the current root if main worktree does not have it
if [[ ! -d "$LIBVIPS_ROOT" ]]; then
  LIBVIPS_ROOT="${LIBVIPS_ROOT_OVERRIDE:-$ROOT/.libvips_repo/libvips}"
fi

if [[ ! -d "$LIBVIPS_ROOT" ]]; then
  echo "ERROR: .libvips_repo not found at $LIBVIPS_ROOT" >&2
  echo "       Clone libvips into .libvips_repo/ at the project root, or" >&2
  echo "       set LIBVIPS_ROOT_OVERRIDE env var to the libvips source directory." >&2
  exit 2
fi

OPS_ROOT="$ROOT/src/domain/ops"
CODECS_ROOT="$ROOT/src/adapters/codecs"
SRC_ROOT="$ROOT/src"

DIFF_ONLY=0
SUMMARY_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --diff)    DIFF_ONLY=1 ;;
    --summary) SUMMARY_ONLY=1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Internal helpers that are not user-facing operations in libvips.
# ---------------------------------------------------------------------------
INTERNAL_FILES=(
  arithmetic binary nary unary unaryconst statistic
  colour colourspace profiles profile_load
  conversion
  convolution
  create mask
  draw drawink
  histogram hist_unary
  morphology
  mosaicing im_avgdxdy im_clinear im_improve im_initialize im_lrcalcon
  freqfilt
  foreign tiff vips2jpeg vips2magick vips2tiff jpeg2vips webp2vips
  openexr2vips archive exif fits magic magick6load magick7load
  spngload spngsave vipspng dcrawload nsgifload quantise
  interpolate transform resample sample_conv
)

is_internal() {
  local name="$1"
  for skip in "${INTERNAL_FILES[@]}"; do
    [[ "$name" == "$skip" ]] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# Explicit cross-module mapping: libvips_op → path relative to src/
# Used when an op is in a different module or has a different name.
# ---------------------------------------------------------------------------
declare -A CROSS_MODULE_MAP

# arithmetic.c ops implemented in other viprs modules
CROSS_MODULE_MAP["boolean"]="domain/ops/boolean"
CROSS_MODULE_MAP["complex"]="domain/ops/arithmetic/complex_real.rs"
CROSS_MODULE_MAP["relational"]="domain/ops/relational"
CROSS_MODULE_MAP["math"]="domain/ops/arithmetic/sin.rs"
CROSS_MODULE_MAP["math2"]="domain/ops/arithmetic/power.rs"
CROSS_MODULE_MAP["hist_find"]="domain/ops/histogram/hist_find_facades.rs"
CROSS_MODULE_MAP["hist_find_indexed"]="domain/ops/histogram/hist_find_indexed.rs"
CROSS_MODULE_MAP["avg"]="domain/ops/arithmetic/reduce_facades.rs"
CROSS_MODULE_MAP["max"]="domain/ops/arithmetic/reduce_facades.rs"
CROSS_MODULE_MAP["min"]="domain/ops/arithmetic/reduce_facades.rs"
CROSS_MODULE_MAP["deviate"]="domain/ops/arithmetic/reduce_facades.rs"
CROSS_MODULE_MAP["stats"]="domain/ops/arithmetic/reduce_facades.rs"
# colour.c ops with non-obvious name translations
CROSS_MODULE_MAP["float2rad"]="domain/ops/colour/float_to_radiance.rs"
CROSS_MODULE_MAP["rad2float"]="domain/ops/colour/radiance_to_float.rs"
CROSS_MODULE_MAP["icc_transform"]="domain/ops/colour/icc.rs"
# convolution.c variants unified in conv.rs
CROSS_MODULE_MAP["convf"]="domain/ops/convolution/conv.rs"
CROSS_MODULE_MAP["convi"]="domain/ops/convolution/conv.rs"
CROSS_MODULE_MAP["spcor"]="domain/ops/convolution/correlation.rs"
# resample.c
CROSS_MODULE_MAP["shrink"]="domain/ops/resample/shrinkh.rs"
CROSS_MODULE_MAP["quadratic"]="MISSING"
# conversion.c ops in structural/
CROSS_MODULE_MAP["flatten"]="domain/ops/structural/flatten.rs"
CROSS_MODULE_MAP["insert"]="domain/ops/structural/insert.rs"
CROSS_MODULE_MAP["join"]="domain/ops/structural/join.rs"
CROSS_MODULE_MAP["premultiply"]="domain/ops/structural/premultiply.rs"
CROSS_MODULE_MAP["unpremultiply"]="domain/ops/structural/unpremultiply.rs"
CROSS_MODULE_MAP["recomb"]="domain/ops/lut/recomb.rs"
CROSS_MODULE_MAP["extract"]="domain/ops/structural/extract_area.rs"
CROSS_MODULE_MAP["cache"]="MISSING"
CROSS_MODULE_MAP["sequential"]="MISSING"
CROSS_MODULE_MAP["tilecache"]="MISSING"
# histogram.c ops
CROSS_MODULE_MAP["maplut"]="domain/ops/lut/map_lut.rs"
CROSS_MODULE_MAP["percent"]="domain/ops/histogram/hist_percent.rs"
CROSS_MODULE_MAP["hist_local"]="domain/ops/histogram/clahe.rs"
CROSS_MODULE_MAP["case"]="MISSING"
CROSS_MODULE_MAP["hist_plot"]="MISSING"
# morphology.c ops
CROSS_MODULE_MAP["morph"]="domain/ops/morphology/erode.rs"
CROSS_MODULE_MAP["countlines"]="domain/ops/morphology/count_lines.rs"
# mosaicing.c ops
CROSS_MODULE_MAP["match"]="domain/ops/mosaicing/match_op.rs"
CROSS_MODULE_MAP["lrmerge"]="domain/ops/mosaicing/merge.rs"
CROSS_MODULE_MAP["tbmerge"]="domain/ops/mosaicing/merge.rs"
CROSS_MODULE_MAP["lrmosaic"]="domain/ops/mosaicing/mosaic.rs"
CROSS_MODULE_MAP["tbmosaic"]="domain/ops/mosaicing/mosaic.rs"
CROSS_MODULE_MAP["matrixinvert"]="MISSING"
CROSS_MODULE_MAP["matrixmultiply"]="MISSING"
CROSS_MODULE_MAP["mosaic1"]="MISSING"
CROSS_MODULE_MAP["im_tbcalcon"]="MISSING"
# colour.c ops under oklab submodule
CROSS_MODULE_MAP["Oklab2Oklch"]="domain/ops/colour/oklab/oklab_to_oklch.rs"
CROSS_MODULE_MAP["Oklab2XYZ"]="domain/ops/colour/oklab/oklab_to_xyz.rs"
CROSS_MODULE_MAP["Oklch2Oklab"]="domain/ops/colour/oklab/oklch_to_oklab.rs"
CROSS_MODULE_MAP["XYZ2Oklab"]="domain/ops/colour/oklab/xyz_to_oklab.rs"
CROSS_MODULE_MAP["LabQ2LabS"]="domain/ops/colour/labq_to_lab.rs"
CROSS_MODULE_MAP["LabS2LabQ"]="domain/ops/colour/labs_to_lab.rs"
CROSS_MODULE_MAP["CICP2scRGB"]="MISSING"
CROSS_MODULE_MAP["uhdr2scRGB"]="MISSING"
# create.c mask variants unified in frequency_mask.rs
CROSS_MODULE_MAP["mask_butterworth"]="domain/ops/create/frequency_mask.rs"
CROSS_MODULE_MAP["mask_butterworth_band"]="domain/ops/create/frequency_mask.rs"
CROSS_MODULE_MAP["mask_butterworth_ring"]="domain/ops/create/frequency_mask.rs"
CROSS_MODULE_MAP["mask_gaussian"]="domain/ops/create/frequency_mask.rs"
CROSS_MODULE_MAP["mask_gaussian_band"]="domain/ops/create/frequency_mask.rs"
CROSS_MODULE_MAP["mask_gaussian_ring"]="domain/ops/create/frequency_mask.rs"
CROSS_MODULE_MAP["mask_ideal"]="domain/ops/create/frequency_mask.rs"
CROSS_MODULE_MAP["mask_ideal_band"]="domain/ops/create/frequency_mask.rs"
CROSS_MODULE_MAP["mask_ideal_ring"]="domain/ops/create/frequency_mask.rs"
CROSS_MODULE_MAP["mask_fractal"]="MISSING"
CROSS_MODULE_MAP["perlin"]="MISSING"
CROSS_MODULE_MAP["point"]="MISSING"
CROSS_MODULE_MAP["sdf"]="MISSING"
CROSS_MODULE_MAP["text"]="MISSING"
CROSS_MODULE_MAP["worley"]="MISSING"

# ---------------------------------------------------------------------------
# Check if a viprs module directory has a file matching an op name.
# ---------------------------------------------------------------------------
viprs_has_op() {
  local op_name="$1"
  local module_dir="$2"

  # Check cross-module explicit mapping first
  if [[ -n "${CROSS_MODULE_MAP[$op_name]+_}" ]]; then
    local mapped_path="${CROSS_MODULE_MAP[$op_name]}"
    [[ "$mapped_path" == "MISSING" ]] && return 1
    local full_path="$SRC_ROOT/$mapped_path"
    if [[ -f "$full_path" ]] || [[ -d "$full_path" ]]; then
      return 0
    fi
    return 1
  fi

  [[ -d "$module_dir" ]] || return 1

  # Exact match
  [[ -f "$module_dir/${op_name}.rs" ]] && return 0

  # Common libvips → viprs name translations
  local translated
  translated=$(echo "$op_name" | sed \
    -e 's/2/_to_/g' \
    -e 's/sRGB/srgb/g' \
    -e 's/XYZ/xyz/g' \
    -e 's/Lab/lab/g' \
    -e 's/CMYK/cmyk/g' \
    -e 's/HSV/hsv/g' \
    -e 's/LCh/lch/g' \
    -e 's/Oklab/oklab/g' \
    -e 's/Oklch/oklch/g' \
    -e 's/UCS/ucs/g' \
    -e 's/BW/bw/g' \
    -e 's/Yxy/yxy/g' \
    -e 's/scRGB/scrgb/g' \
    -e 's/LabQ/labq/g' \
    -e 's/LabS/labs/g' \
    -e 's/CICP/cicp/g' \
    -e 's/dE/de/g' \
    -e 's/dECMC/decmc/g' \
    -e 's/float2rad/float_to_radiance/g' \
    -e 's/rad2float/radiance_to_float/g' \
    -e 's/gaussblur/gauss_blur/g' \
  )

  [[ -f "$module_dir/${translated}.rs" ]] && return 0

  local lower
  lower=$(echo "$op_name" | tr '[:upper:]' '[:lower:]')
  [[ -f "$module_dir/${lower}.rs" ]] && return 0

  local translated_lower
  translated_lower=$(echo "$translated" | tr '[:upper:]' '[:lower:]')
  [[ -f "$module_dir/${translated_lower}.rs" ]] && return 0

  # Search recursively (handles subdirs like oklab/)
  local found
  found=$(find "$module_dir" -name "${translated}.rs" -o -name "${translated_lower}.rs" 2>/dev/null | head -1 || true)
  [[ -n "$found" ]] && return 0

  return 1
}

# Check if op file contains todo!/unimplemented!
viprs_op_is_stub() {
  local op_name="$1"
  local module_dir="$2"

  # Check cross-module path for stubs too
  local check_dir="$module_dir"
  if [[ -n "${CROSS_MODULE_MAP[$op_name]+_}" ]]; then
    local mapped_path="${CROSS_MODULE_MAP[$op_name]}"
    [[ "$mapped_path" == "MISSING" ]] && return 1
    check_dir="$(dirname "$SRC_ROOT/$mapped_path")"
  fi

  local lower
  lower=$(echo "$op_name" | tr '[:upper:]' '[:lower:]')

  while IFS= read -r f; do
    [[ -f "$f" ]] || continue
    local base
    base=$(basename "$f" .rs)
    if [[ "$base" == "$op_name" ]] || [[ "$base" == "$lower" ]]; then
      if grep -q 'todo!\|unimplemented!' "$f" 2>/dev/null; then
        return 0
      fi
    fi
  done < <(find "$check_dir" -name "*.rs" 2>/dev/null)

  return 1
}

# ---------------------------------------------------------------------------
# Process one libvips module
# ---------------------------------------------------------------------------
process_module() {
  local module="$1"
  local libvips_dir="$2"
  local viprs_dir="$3"

  local total=0 implemented=0 stubs=0 missing=0

  while IFS= read -r cfile; do
    local op_name
    op_name=$(basename "$cfile" .c)

    is_internal "$op_name" && continue

    total=$((total + 1))

    if viprs_has_op "$op_name" "$viprs_dir"; then
      if viprs_op_is_stub "$op_name" "$viprs_dir"; then
        stubs=$((stubs + 1))
        if [[ $SUMMARY_ONLY -eq 0 ]]; then
          echo "| $op_name | 🔶 stub | |"
        fi
      else
        implemented=$((implemented + 1))
        if [[ $SUMMARY_ONLY -eq 0 && $DIFF_ONLY -eq 0 ]]; then
          echo "| $op_name | ✅ implemented | |"
        fi
      fi
    else
      missing=$((missing + 1))
      if [[ $SUMMARY_ONLY -eq 0 ]]; then
        echo "| $op_name | ❌ missing | |"
      fi
    fi
  done < <(find "$libvips_dir" -maxdepth 1 -name "*.c" 2>/dev/null | sort)

  echo "COUNTS:$total:$implemented:$stubs:$missing"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODULES=(
  "arithmetic:$LIBVIPS_ROOT/arithmetic:$OPS_ROOT/arithmetic"
  "colour:$LIBVIPS_ROOT/colour:$OPS_ROOT/colour"
  "convolution:$LIBVIPS_ROOT/convolution:$OPS_ROOT/convolution"
  "resample:$LIBVIPS_ROOT/resample:$OPS_ROOT/resample"
  "conversion:$LIBVIPS_ROOT/conversion:$OPS_ROOT/conversion"
  "create:$LIBVIPS_ROOT/create:$OPS_ROOT/create"
  "morphology:$LIBVIPS_ROOT/morphology:$OPS_ROOT/morphology"
  "draw:$LIBVIPS_ROOT/draw:$OPS_ROOT/draw"
  "histogram:$LIBVIPS_ROOT/histogram:$OPS_ROOT/histogram"
  "mosaicing:$LIBVIPS_ROOT/mosaicing:$OPS_ROOT/mosaicing"
  "freqfilt:$LIBVIPS_ROOT/freqfilt:$OPS_ROOT/freqfilt"
)

declare -A MOD_TOTAL MOD_IMPL MOD_STUBS MOD_MISS

GRAND_TOTAL=0 GRAND_IMPL=0 GRAND_STUBS=0 GRAND_MISS=0

if [[ $SUMMARY_ONLY -eq 0 ]]; then
  echo "# Viprs ↔ libvips Parity Matrix"
  echo ""
  echo "Generated: $(date '+%Y-%m-%d')"
  echo ""
  echo "**Status legend:**"
  echo "- ✅ implemented — concrete implementation"
  echo "- 🔶 stub — \`todo!()\` / \`unimplemented!()\` present"
  echo "- ❌ missing — no corresponding file in viprs"
  echo ""
  echo "---"
  echo ""
fi

for entry in "${MODULES[@]}"; do
  IFS=':' read -r mod libvips_dir viprs_dir <<< "$entry"

  if [[ $SUMMARY_ONLY -eq 0 ]]; then
    echo "## $mod/"
    echo ""
    echo "| libvips op | viprs status | notes |"
    echo "|---|---|---|"
  fi

  output=$(process_module "$mod" "$libvips_dir" "$viprs_dir")
  counts_line=$(echo "$output" | grep "^COUNTS:")
  table_lines=$(echo "$output" | grep -v "^COUNTS:" || true)

  if [[ $SUMMARY_ONLY -eq 0 ]]; then
    echo "$table_lines"
  fi

  IFS=':' read -r _ total impl stubs miss <<< "$counts_line"

  MOD_TOTAL[$mod]=$total
  MOD_IMPL[$mod]=$impl
  MOD_STUBS[$mod]=$stubs
  MOD_MISS[$mod]=$miss

  GRAND_TOTAL=$((GRAND_TOTAL + total))
  GRAND_IMPL=$((GRAND_IMPL + impl))
  GRAND_STUBS=$((GRAND_STUBS + stubs))
  GRAND_MISS=$((GRAND_MISS + miss))

  if [[ $SUMMARY_ONLY -eq 0 ]]; then
    local_pct=0
    if [[ $total -gt 0 ]]; then local_pct=$(( (impl * 100) / total )); fi
    echo ""
    echo "_Module parity: ${impl}/${total} (${local_pct}%)_"
    echo ""
    echo "---"
    echo ""
  fi
done

# Summary table
echo ""
echo "## Summary"
echo ""
echo "| Module | libvips ops | ✅ implemented | 🔶 stub | ❌ missing | Parity % |"
echo "|---|---|---|---|---|---|"
for entry in "${MODULES[@]}"; do
  IFS=':' read -r mod _ _ <<< "$entry"
  t=${MOD_TOTAL[$mod]}
  i=${MOD_IMPL[$mod]}
  s=${MOD_STUBS[$mod]}
  m=${MOD_MISS[$mod]}
  pct=0
  if [[ $t -gt 0 ]]; then pct=$(( (i * 100) / t )); fi
  echo "| $mod | $t | $i | $s | $m | ${pct}% |"
done

grand_pct=0
if [[ $GRAND_TOTAL -gt 0 ]]; then grand_pct=$(( (GRAND_IMPL * 100) / GRAND_TOTAL )); fi
echo "| **TOTAL** | **$GRAND_TOTAL** | **$GRAND_IMPL** | **$GRAND_STUBS** | **$GRAND_MISS** | **${grand_pct}%** |"
echo ""
echo "> Note: counts exclude internal libvips helper files (not user-facing ops)."
echo ""

# Exit with error if gaps exist
if [[ $((GRAND_MISS + GRAND_STUBS)) -gt 0 ]]; then
  exit 1
fi
exit 0
