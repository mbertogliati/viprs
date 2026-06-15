#!/bin/bash
# profile.sh — cachegrind per-function cache-miss profiling for viprs vs libvips.
#
# Runs both binaries under valgrind --tool=cachegrind, then calls cg_annotate
# on each output and prints a side-by-side ranked table sorted by DLmr
# (last-level data cache misses — the expensive ones).
#
# Usage (via Docker entrypoint override):
#   docker run --entrypoint /opt/bench/profile.sh <image> \
#     <input> <op> [op_args...] --iterations N

set -euo pipefail

INPUT=""
OP=""
OP_ARGS=()
ITERATIONS=5  # fewer needed for cachegrind — it's deterministic

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iterations) ITERATIONS="$2"; shift 2 ;;
        *)
            if [[ -z "$INPUT" ]]; then
                INPUT="$1"
            elif [[ -z "$OP" ]]; then
                OP="$1"
            else
                OP_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "$INPUT" || -z "$OP" ]]; then
    echo "Usage: profile.sh <input> <op> [args...] --iterations N" >&2
    exit 1
fi

LIBVIPS_ARGS=("$INPUT" "$OP" "${OP_ARGS[@]+"${OP_ARGS[@]}"}" "--iterations" "$ITERATIONS")
VIPRS_ARGS=("bench" "$INPUT" "$OP" "${OP_ARGS[@]+"${OP_ARGS[@]}"}" "--iterations" "$ITERATIONS")

CG_LIBVIPS="/tmp/cg_libvips_$$.out"
CG_VIPRS="/tmp/cg_viprs_$$.out"

echo "=== cachegrind profile: viprs vs libvips ==="
echo "  op:         $OP ${OP_ARGS[*]+"${OP_ARGS[*]}"}"
echo "  input:      $INPUT"
echo "  iterations: $ITERATIONS"
echo "  arch:       $(uname -m)"
echo ""
echo "  Running cachegrind on libvips... (20-50x slower — deterministic)"
valgrind --tool=cachegrind \
         --cachegrind-out-file="$CG_LIBVIPS" \
         --cache-sim=yes \
         --branch-sim=yes \
         libvips-runner "${LIBVIPS_ARGS[@]}" \
         >/dev/null 2>&1 || true

echo "  Running cachegrind on viprs..."
valgrind --tool=cachegrind \
         --cachegrind-out-file="$CG_VIPRS" \
         --cache-sim=yes \
         --branch-sim=yes \
         viprs-xtask "${VIPRS_ARGS[@]}" \
         >/dev/null 2>&1 || true

echo ""

# ── Per-function table ──────────────────────────────────────────────────────
# Extract function-level DLmr (LL data cache read misses) from cg_annotate.
# cg_annotate --auto=yes produces a table like:
#   Ir  I1mr ILmr  Dr  D1mr DLmr  Dw  D1mw DLmw  file:function
#
# We parse both files and join on function name to produce a comparison.

parse_cg_annotate() {
    local cg_file="$1"
    cg_annotate --auto=yes "$cg_file" 2>/dev/null \
        | grep -E '^\s+[0-9,]' \
        | awk '
        {
            # strip leading whitespace, split on whitespace
            # columns: Ir I1mr ILmr Dr D1mr DLmr Dw D1mw DLmw function
            # positions may vary — find function name (last field containing :)
            dlmr = 0
            func = ""
            for (i = 1; i <= NF; i++) {
                # DLmr is the 6th numeric column
                if (i == 6) { dlmr = $i; gsub(",","",dlmr) }
                # function is after the last number
                if ($i ~ /[a-zA-Z_]/ && $i !~ /^[0-9,]+$/) {
                    func = $i
                    break
                }
            }
            if (func != "" && dlmr+0 > 0) {
                print dlmr "\t" func
            }
        }' \
        | sort -rn \
        | head -30
}

echo "--- Per-function LL cache misses (DLmr) — top 30 ---"
echo ""

# Write parsed tables to temp files
PARSED_LV="/tmp/parsed_lv_$$.txt"
PARSED_VP="/tmp/parsed_vp_$$.txt"
parse_cg_annotate "$CG_LIBVIPS" > "$PARSED_LV"
parse_cg_annotate "$CG_VIPRS"   > "$PARSED_VP"

# Print comparison table
printf "%-60s  %12s  %12s  %8s\n" "Function" "libvips DLmr" "viprs DLmr" "ratio"
printf "%-60s  %12s  %12s  %8s\n" "$(printf '%0.s-' {1..60})" "$(printf '%0.s-' {1..12})" "$(printf '%0.s-' {1..12})" "$(printf '%0.s-' {1..8})"

# Merge both lists, deduplicate by function name
{
    awk '{print $2}' "$PARSED_LV"
    awk '{print $2}' "$PARSED_VP"
} | sort -u | while read -r func; do
    lv_val=$(grep -F "$func" "$PARSED_LV" | awk '{print $1}' | head -1)
    vp_val=$(grep -F "$func" "$PARSED_VP" | awk '{print $1}' | head -1)
    lv_val="${lv_val:-0}"
    vp_val="${vp_val:-0}"

    if [[ "$lv_val" -eq 0 && "$vp_val" -eq 0 ]]; then
        continue
    fi

    if [[ "$lv_val" -gt 0 ]]; then
        ratio=$(awk "BEGIN { printf \"%.1fx\", $vp_val / $lv_val }")
    else
        ratio="inf"
    fi

    # Truncate function name if too long
    short_func="${func:0:58}"
    printf "%-60s  %12d  %12d  %8s\n" "$short_func" "$lv_val" "$vp_val" "$ratio"
done | sort -t$'\t' -k4 -rn 2>/dev/null || \
# Fallback: just print raw tables side by side
{
    echo ""
    echo "  libvips top misses:"
    head -15 "$PARSED_LV" | awk '{printf "  %12d  %s\n", $1, $2}'
    echo ""
    echo "  viprs top misses:"
    head -15 "$PARSED_VP" | awk '{printf "  %12d  %s\n", $1, $2}'
}

echo ""
echo "--- Aggregate summary ---"
echo ""

print_summary() {
    local cg_file="$1"
    local label="$2"
    if [[ ! -f "$cg_file" ]]; then
        echo "  $label: no cachegrind output"
        return
    fi
    local events data
    events=$(grep '^events:' "$cg_file" | sed 's/events: //')
    data=$(grep '^summary:' "$cg_file" | sed 's/summary: //')

    echo "  $label:"
    local i=0
    for event in $events; do
        val=$(echo "$data" | awk -v idx=$((i+1)) '{print $idx}')
        printf "    %-10s %s\n" "$event" "${val:-0}"
        i=$((i+1))
    done
    echo ""
}

print_summary "$CG_LIBVIPS" "libvips"
print_summary "$CG_VIPRS"   "viprs"

echo "--- How to read this ---"
echo ""
echo "  DLmr = Last-Level data cache read misses (the expensive ones)"
echo "  D1mr = L1 data cache read misses"
echo "  Ir   = Total instructions executed"
echo ""
echo "  A function with high DLmr in viprs but low in libvips → bad memory access"
echo "  pattern: wrong stride, tile too large, or sub-optimal data layout."
echo "  Cross-reference with src/domain/ops/ vs .libvips_repo/ to find the fix."

# Cleanup
rm -f "$CG_LIBVIPS" "$CG_VIPRS" "$PARSED_LV" "$PARSED_VP"
