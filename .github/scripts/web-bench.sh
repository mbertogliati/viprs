#!/usr/bin/env bash
set -euo pipefail

# Web-service workload simulation: thumbnail, pipeline, concurrent, large-upload.

# Use pre-built binary if available (CI), otherwise compile via cargo.
XTASK="${XTASK_BIN:-cargo xtask}"

FIXTURE="tests/fixtures/images/bench_2048x2048.jpg"
[ -f "$FIXTURE" ] || FIXTURE="tests/fixtures/images/sample.jpg"

echo "=== thumbnail-bytes (decode → thumbnail → encode, bytes-in/bytes-out) ==="
$XTASK web-bench "$FIXTURE" -s thumbnail-bytes -n 20 --json \
  > /tmp/bench-results/web_thumbnail_bytes.json 2>/dev/null || true

echo "=== pipeline-bytes (thumbnail + sharpen + linear → JPEG) ==="
$XTASK web-bench "$FIXTURE" -s pipeline-bytes -n 20 --json \
  > /tmp/bench-results/web_pipeline_bytes.json 2>/dev/null || true

echo "=== concurrent (parallel requests: 2, 4, 8, 16 threads) ==="
$XTASK web-bench "$FIXTURE" -s concurrent --concurrency 2,4,8,16 -n 20 --json \
  > /tmp/bench-results/web_concurrent.json 2>/dev/null || true

echo "=== large-upload (8192px → thumbnail, simulates user upload) ==="
LARGE="tests/fixtures/images/bench_8192x8192.jpg"
if [ -f "$LARGE" ]; then
  $XTASK web-bench "$LARGE" -s large-upload -n 10 --json \
    > /tmp/bench-results/web_large_upload.json 2>/dev/null || true
fi

echo "✅ Web-service workload simulation completed."
