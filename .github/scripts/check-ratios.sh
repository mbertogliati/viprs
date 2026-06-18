#!/usr/bin/env bash
set -euo pipefail

# Check viprs/libvips ratios — fail if any operation exceeds 1.05.

FAILED=0
for f in /tmp/bench-results/*.json; do
  [ -f "$f" ] || continue
  ratio=$(jq -r '.ratio_p50 // .ratio // empty' "$f" 2>/dev/null || true)
  [ -z "$ratio" ] && continue
  name=$(basename "$f" .json)
  # Skip non-numeric ratios (corrupt JSON or null values)
  if ! echo "$ratio" | grep -qE '^[0-9]+\.?[0-9]*$'; then
    echo "::warning::$name: skipped — non-numeric ratio '$ratio'"
    continue
  fi
  echo "$name: ratio=$ratio"
  if [ "$(echo "$ratio > 1.05" | bc -l)" = "1" ]; then
    echo "::warning::$name: viprs/libvips ratio $ratio > 1.05"
    FAILED=$((FAILED + 1))
  fi
done

if [ "$FAILED" -gt 0 ]; then
  echo "::error::$FAILED operation(s) slower than libvips (ratio > 1.05)"
  exit 1
fi
echo "✓ All operations within target ratio (≤ 1.05)"
