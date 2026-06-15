#!/usr/bin/env bash
# bench-vs-libvips/run.sh — Orchestrates both runners on same input and compares.
#
# Usage: ./run.sh <input_image> <operation> [op_args...] [--iterations N]
#
# Examples:
#   ./run.sh tests/fixtures/images/sample.jpg thumbnail 800 --iterations 50
#   ./run.sh tests/fixtures/images/sample.png invert --iterations 100
#   ./run.sh tests/fixtures/images/sample.jpg resize 0.5 --iterations 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

# Build libvips runner if needed
if [ ! -f "$SCRIPT_DIR/libvips-runner" ]; then
    echo "Building libvips runner..."
    make -C "$SCRIPT_DIR" libvips-runner
fi

# Build viprs runner if needed
if ! cargo build --release --bin viprs-bench-runner -q 2>/dev/null; then
    echo "WARNING: viprs-bench-runner not yet implemented, skipping viprs side."
    VIPRS_AVAILABLE=0
else
    VIPRS_AVAILABLE=1
fi

# Parse arguments
INPUT="$1"
shift

# Resolve relative input paths
if [[ "$INPUT" != /* ]]; then
    INPUT="$REPO_ROOT/$INPUT"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OP_NAME="${1:-unknown}"
RESULT_FILE="$RESULTS_DIR/${OP_NAME}_${TIMESTAMP}.json"

echo "=== bench-vs-libvips ==="
echo "Input: $INPUT"
echo "Operation: $*"
echo "Results: $RESULT_FILE"
echo ""

# Run libvips
echo "--- libvips C runner ---"
LIBVIPS_JSON=$("$SCRIPT_DIR/libvips-runner" "$INPUT" "$@")
echo "$LIBVIPS_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ns = sorted(d['wall_ns'])
n = len(ns)
p50 = ns[n//2]
p95 = ns[int(n*0.95)]
print(f'  p50: {p50/1e6:.2f} ms  p95: {p95/1e6:.2f} ms  RSS: {d[\"peak_rss_kb\"]} KB')
"

# Run viprs (when available)
if [ "$VIPRS_AVAILABLE" = "1" ]; then
    echo "--- viprs runner ---"
    VIPRS_JSON=$("$REPO_ROOT/target/release/viprs-bench-runner" "$INPUT" "$@")
    echo "$VIPRS_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ns = sorted(d['wall_ns'])
n = len(ns)
p50 = ns[n//2]
p95 = ns[int(n*0.95)]
print(f'  p50: {p50/1e6:.2f} ms  p95: {p95/1e6:.2f} ms  RSS: {d[\"peak_rss_kb\"]} KB')
"
else
    VIPRS_JSON="{}"
fi

# Write combined result
python3 -c "
import json, sys
libvips = json.loads('''$LIBVIPS_JSON''')
viprs = json.loads('''$VIPRS_JSON''')
result = {
    'libvips': libvips,
    'viprs': viprs if viprs else None,
}
if viprs and 'wall_ns' in viprs and 'wall_ns' in libvips:
    lv_ns = sorted(libvips['wall_ns'])
    vp_ns = sorted(viprs['wall_ns'])
    n_lv = len(lv_ns)
    n_vp = len(vp_ns)
    result['comparison'] = {
        'latency_ratio_p50': vp_ns[n_vp//2] / max(lv_ns[n_lv//2], 1),
        'rss_ratio': (viprs.get('peak_rss_kb', 0) / max(libvips.get('peak_rss_kb', 1), 1)),
    }
    ratio = result['comparison']['latency_ratio_p50']
    if ratio < 1.0:
        print(f'  >>> viprs is {1/ratio:.2f}x FASTER <<<')
    else:
        print(f'  >>> viprs is {ratio:.2f}x slower <<<')
with open('$RESULT_FILE', 'w') as f:
    json.dump(result, f, indent=2)
print(f'Results saved to: $RESULT_FILE')
" 2>/dev/null || true

echo ""
echo "Done."
