#!/usr/bin/env bash
set -euo pipefail

# Web-service workload simulation: thumbnail, pipeline, concurrent, large-upload.
# Failures are collected and reported at the end.

FIXTURE="tests/fixtures/images/bench_2048x2048.jpg"
[ -f "$FIXTURE" ] || FIXTURE="tests/fixtures/images/sample.jpg"

FAILED=()

echo "=== thumbnail-bytes (decode → thumbnail → encode, bytes-in/bytes-out) ==="
if ! cargo xtask web-bench "$FIXTURE" -s thumbnail-bytes -n 20 --json \
  > /tmp/bench-results/web_thumbnail_bytes.json 2>&1; then
  echo "::warning::web-bench failed: thumbnail-bytes"
  FAILED+=("thumbnail-bytes")
fi

echo "=== pipeline-bytes (thumbnail + sharpen + linear → JPEG) ==="
if ! cargo xtask web-bench "$FIXTURE" -s pipeline-bytes -n 20 --json \
  > /tmp/bench-results/web_pipeline_bytes.json 2>&1; then
  echo "::warning::web-bench failed: pipeline-bytes"
  FAILED+=("pipeline-bytes")
fi

echo "=== concurrent (parallel requests: 2, 4, 8, 16 threads) ==="
if ! cargo xtask web-bench "$FIXTURE" -s concurrent --concurrency 2,4,8,16 -n 20 --json \
  > /tmp/bench-results/web_concurrent.json 2>&1; then
  echo "::warning::web-bench failed: concurrent"
  FAILED+=("concurrent")
fi

echo "=== large-upload (8192px → thumbnail, simulates user upload) ==="
LARGE="tests/fixtures/images/bench_8192x8192.jpg"
if [ -f "$LARGE" ]; then
  if ! cargo xtask web-bench "$LARGE" -s large-upload -n 10 --json \
    > /tmp/bench-results/web_large_upload.json 2>&1; then
    echo "::warning::web-bench failed: large-upload"
    FAILED+=("large-upload")
  fi
else
  echo "::warning::Missing fixture: $LARGE — skipping large-upload"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "::error::${#FAILED[@]} web-bench scenario(s) failed: ${FAILED[*]}"
  exit 1
fi

echo "✅ Web-service workload simulation completed — all scenarios succeeded."
