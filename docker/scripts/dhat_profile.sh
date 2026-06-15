#!/bin/bash
# dhat_profile.sh — heap allocation profiling for viprs and libvips via dhat.
#
# Runs libvips-runner under valgrind --tool=dhat and viprs-xtask with the
# dhat feature enabled, then copies JSON profiles to /data/output/ so they
# survive container exit.
#
# Usage (via Docker --entrypoint override):
#   docker run -v /tmp:/data/output --entrypoint /opt/bench/dhat_profile.sh <image> \
#     <input> <op> [op_args...] --iterations N

set -euo pipefail

INPUT=""
OP=""
OP_ARGS=()
ITERATIONS=5  # dhat is slow under valgrind — fewer iterations needed

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
    echo "Usage: dhat_profile.sh <input> <op> [args...] --iterations N" >&2
    exit 1
fi

LIBVIPS_ARGS=("$INPUT" "$OP" "${OP_ARGS[@]+"${OP_ARGS[@]}"}" "--iterations" "$ITERATIONS")
VIPRS_ARGS=("bench" "$INPUT" "$OP" "${OP_ARGS[@]+"${OP_ARGS[@]}"}" "--iterations" "$ITERATIONS" "--no-e2e")

DHAT_LIBVIPS="/tmp/libvips_dhat_${OP}.json"
DHAT_VIPRS="/tmp/viprs_dhat_${OP}.json"
DHAT_VIPRS_RAW="/tmp/dhat-heap.json"

echo "=== dhat allocation profiling: viprs vs libvips ==="
echo "  op:         $OP ${OP_ARGS[*]+"${OP_ARGS[*]}"}"
echo "  input:      $INPUT"
echo "  iterations: $ITERATIONS"
echo "  arch:       $(uname -m)"
echo ""

# ── libvips: valgrind dhat ─────────────────────────────────────────────────
echo "  [1/2] Profiling libvips with valgrind dhat..."
valgrind --tool=dhat \
         --dhat-out-file="$DHAT_LIBVIPS" \
         libvips-runner "${LIBVIPS_ARGS[@]}" \
         >/dev/null 2>&1 || true

if [[ -f "$DHAT_LIBVIPS" ]]; then
    echo "  ✓ libvips dhat profile: $DHAT_LIBVIPS"
else
    echo "  ✗ libvips dhat profile not produced"
fi

# ── viprs: dhat crate (dhat feature must be enabled in the binary) ──────────
echo "  [2/2] Profiling viprs with dhat crate..."
# The xtask binary in the Docker image is built with dhat support.
# It writes dhat-heap.json in the working directory.
cd /tmp
viprs-xtask "${VIPRS_ARGS[@]}" >/dev/null 2>&1 || true

if [[ -f "$DHAT_VIPRS_RAW" ]]; then
    mv "$DHAT_VIPRS_RAW" "$DHAT_VIPRS"
    echo "  ✓ viprs dhat profile:   $DHAT_VIPRS"
else
    echo "  ✗ viprs dhat profile not produced (dhat feature may not be enabled)"
fi

# ── Copy to output mount ───────────────────────────────────────────────────
if [[ -d "/data/output" ]]; then
    [[ -f "$DHAT_LIBVIPS" ]] && cp "$DHAT_LIBVIPS" "/data/output/" && echo "  → copied to /data/output/$(basename $DHAT_LIBVIPS)"
    [[ -f "$DHAT_VIPRS"   ]] && cp "$DHAT_VIPRS"   "/data/output/" && echo "  → copied to /data/output/$(basename $DHAT_VIPRS)"
fi

# ── Quick summary from dhat JSON ──────────────────────────────────────────
print_dhat_summary() {
    local json_file="$1"
    local label="$2"
    if [[ ! -f "$json_file" ]]; then return; fi

    # dhat JSON has "total-blocks", "total-bytes", "peak-bytes" at top level
    local total_blocks total_bytes peak_bytes
    total_blocks=$(grep -o '"total-blocks":[0-9]*' "$json_file" | head -1 | cut -d: -f2 || echo "?")
    total_bytes=$(grep -o '"total-bytes":[0-9]*' "$json_file" | head -1 | cut -d: -f2 || echo "?")
    peak_bytes=$(grep -o '"peak-bytes":[0-9]*' "$json_file" | head -1 | cut -d: -f2 || echo "?")

    echo "  $label:"
    echo "    total allocations: $total_blocks"
    echo "    total bytes:       $total_bytes"
    echo "    peak live bytes:   $peak_bytes"
}

echo ""
echo "--- Aggregate summary ---"
echo ""
print_dhat_summary "$DHAT_LIBVIPS" "libvips"
echo ""
print_dhat_summary "$DHAT_VIPRS"   "viprs"

echo ""
echo "--- How to read dhat profiles ---"
echo ""
echo "  1. Open https://nnethercote.github.io/dh_view/dh_view.html"
echo "  2. Load $(basename "$DHAT_VIPRS") → inspect viprs call stacks"
echo "  3. Reload and load $(basename "$DHAT_LIBVIPS") → compare with libvips"
echo ""
echo "  Red flags in viprs:"
echo "    - Any frame containing 'process_region' or 'domain/ops/' → zero-alloc violation"
echo "    - Alloc count grows linearly with image size → per-tile Vec in the hot path"
echo "    - Alloc sites present in viprs but absent in libvips → unnecessary copies"
